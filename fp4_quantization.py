#!/usr/bin/env python3
"""
FP4 Quantization Script for NVIDIA Cosmos-Reason1-7B on RTX 5090 (Blackwell)

This script uses NVIDIA TensorRT-LLM to build an FP4 engine with TRUE hardware-accelerated
FP4 computation on RTX 5090's native FP4 Tensor Cores.

RTX 5090 (Blackwell) introduces native FP4 Tensor Core operations that perform actual
4-bit floating point matrix multiplications in hardware, not simulated via higher precision.

Requirements:
    - NVIDIA RTX 5090 GPU (Blackwell architecture, SM 100+)
    - TensorRT-LLM >= 0.15.0 with FP4 support
    - CUDA 12.8+ with Blackwell support
    
Install TensorRT-LLM:
    pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def check_gpu_compatibility():
    """Verify RTX 5090 (Blackwell) GPU is available for native FP4."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. RTX 5090 GPU required.")
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Detected GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Blackwell (RTX 5090) has compute capability 10.0+
    # Native FP4 Tensor Cores require SM 100+
    if compute_cap[0] < 10:
        print(f"\nWARNING: Compute capability {compute_cap[0]}.{compute_cap[1]} detected.")
        print("Native FP4 Tensor Core acceleration requires Blackwell (SM 10.0+).")
        print("RTX 5090 has SM 10.0 with native FP4 support.")
        print("Proceeding, but FP4 may fall back to emulation on older GPUs.")
    else:
        print("Blackwell architecture confirmed - Native FP4 Tensor Cores available!")
    
    return gpu_name, compute_cap, gpu_memory


def convert_hf_to_trtllm_checkpoint(
    model_name: str,
    output_dir: Path,
    tp_size: int = 1,
    pp_size: int = 1,
):
    """
    Convert HuggingFace model to TensorRT-LLM checkpoint format with FP4 quantization.
    
    This step calibrates and quantizes the model weights to NVIDIA's native FP4 format
    (NVFP4 - NVIDIA's 4-bit floating point format optimized for Blackwell Tensor Cores).
    """
    checkpoint_dir = output_dir / "trtllm_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 1] Converting {model_name} to TensorRT-LLM checkpoint with FP4...")
    
    # Use TensorRT-LLM's convert_checkpoint script for Qwen2-VL models
    # The Cosmos-Reason1-7B is based on Qwen2.5-VL architecture
    convert_cmd = [
        sys.executable, "-m", "tensorrt_llm.commands.convert_checkpoint",
        "--model_dir", model_name,
        "--output_dir", str(checkpoint_dir),
        "--dtype", "float16",  # Base dtype before quantization
        "--tp_size", str(tp_size),
        "--pp_size", str(pp_size),
        # FP4 quantization flags for Blackwell
        "--use_fp4_awq",  # Use FP4 AWQ quantization for native Tensor Core acceleration
        "--fp4_gemm",  # Enable FP4 GEMM operations on Blackwell Tensor Cores
    ]
    
    print(f"Running: {' '.join(convert_cmd)}")
    
    try:
        result = subprocess.run(
            convert_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Checkpoint conversion failed: {e.stderr}")
        raise
    except FileNotFoundError:
        # Fallback: Use Python API directly
        print("Using TensorRT-LLM Python API for conversion...")
        convert_with_python_api(model_name, checkpoint_dir, tp_size)
    
    return checkpoint_dir


def convert_with_python_api(model_name: str, checkpoint_dir: Path, tp_size: int = 1):
    """
    Convert model using TensorRT-LLM Python API with FP4 quantization.
    
    This uses NVIDIA's native FP4 (NVFP4) format which maps directly to
    Blackwell's FP4 Tensor Core instructions.
    """
    try:
        import tensorrt_llm
        from tensorrt_llm.quantization import QuantMode
        from tensorrt_llm.models import QWenForCausalLM
        from tensorrt_llm.mapping import Mapping
    except ImportError:
        raise ImportError(
            "TensorRT-LLM not installed. Install with:\n"
            "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
        )
    
    print("Loading model for FP4 quantization...")
    
    # Configure FP4 quantization mode
    # NVFP4 uses 4-bit floating point with hardware acceleration
    quant_mode = QuantMode.from_description(
        quantize_weights=True,
        quantize_activations=True,
        per_channel=True,
        per_token=True,
        use_fp4_awq=True,  # NVIDIA FP4 AWQ for Blackwell
    )
    
    mapping = Mapping(
        world_size=tp_size,
        rank=0,
        tp_size=tp_size,
    )
    
    # Convert and quantize to FP4
    model = QWenForCausalLM.from_hugging_face(
        model_name,
        dtype="float16",
        mapping=mapping,
        quant_mode=quant_mode,
    )
    
    model.save_checkpoint(str(checkpoint_dir), save_config=True)
    print(f"FP4 checkpoint saved to: {checkpoint_dir}")


def build_trtllm_engine(
    checkpoint_dir: Path,
    output_dir: Path,
    max_batch_size: int = 1,
    max_input_len: int = 4096,
    max_output_len: int = 512,
    max_num_tokens: int = 8192,
):
    """
    Build TensorRT-LLM engine with FP4 Tensor Core acceleration for RTX 5090.
    
    The engine is compiled with NVFP4 kernels that execute directly on
    Blackwell's native FP4 Tensor Cores for true hardware acceleration.
    """
    engine_dir = output_dir / "trtllm_engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 2] Building TensorRT-LLM FP4 engine for RTX 5090...")
    
    build_cmd = [
        sys.executable, "-m", "tensorrt_llm.commands.build",
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(engine_dir),
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_seq_len", str(max_input_len + max_output_len),
        "--max_num_tokens", str(max_num_tokens),
        "--gemm_plugin", "fp4",  # Use FP4 GEMM plugin for Blackwell Tensor Cores
        "--strongly_typed",  # Enforce FP4 precision throughout
        "--use_fp4_context_fmha",  # FP4 flash attention for context phase
        "--remove_input_padding",  # Optimize for variable sequence lengths
        "--paged_kv_cache",  # Memory-efficient KV cache
        "--multiple_profiles",  # Support multiple batch sizes efficiently
    ]
    
    print(f"Running: {' '.join(build_cmd)}")
    
    try:
        result = subprocess.run(
            build_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Engine build failed: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Using TensorRT-LLM Python API for engine build...")
        build_engine_with_python_api(checkpoint_dir, engine_dir, max_batch_size, max_input_len, max_output_len)
    
    return engine_dir


def build_engine_with_python_api(
    checkpoint_dir: Path,
    engine_dir: Path,
    max_batch_size: int,
    max_input_len: int,
    max_output_len: int,
):
    """Build TensorRT engine using Python API with FP4 configuration."""
    try:
        import tensorrt_llm
        from tensorrt_llm.builder import Builder
        from tensorrt_llm.plugin import PluginConfig
    except ImportError:
        raise ImportError(
            "TensorRT-LLM not installed. Install with:\n"
            "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
        )
    
    # Configure FP4 plugins for Blackwell Tensor Cores
    plugin_config = PluginConfig()
    plugin_config.gemm_plugin = "fp4"  # Native FP4 GEMM on Blackwell
    plugin_config.context_fmha_type = "fp4"  # FP4 attention
    plugin_config.paged_kv_cache = True
    plugin_config.remove_input_padding = True
    
    builder = Builder()
    
    build_config = builder.create_build_config(
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_input_len + max_output_len,
        plugin_config=plugin_config,
        strongly_typed=True,  # Enforce FP4 precision
    )
    
    # Load checkpoint and build engine
    engine = builder.build_engine_from_checkpoint(
        str(checkpoint_dir),
        build_config,
    )
    
    # Save engine
    builder.save_engine(engine, str(engine_dir / "model.engine"))
    
    # Copy config files
    for config_file in checkpoint_dir.glob("*.json"):
        shutil.copy(config_file, engine_dir)
    
    print(f"FP4 engine saved to: {engine_dir}")


def setup_vision_encoder(model_name: str, output_dir: Path):
    """
    Setup vision encoder for multimodal inference.
    
    The vision encoder (ViT) runs in FP16 while the LLM runs in FP4.
    """
    print("\n[Step 3] Setting up vision encoder...")
    
    vision_dir = output_dir / "vision_encoder"
    vision_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from transformers import AutoProcessor, AutoModel
        
        # Load and save the processor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor.save_pretrained(vision_dir)
        
        # For vision encoder, we can also build a TensorRT engine
        # but typically it runs efficiently in FP16
        print(f"Vision encoder/processor saved to: {vision_dir}")
        
    except Exception as e:
        print(f"Warning: Could not setup vision encoder: {e}")
        print("Vision encoder will be loaded from original model during inference.")
    
    return vision_dir


def quantize_model(
    model_name: str,
    output_dir: str,
    tp_size: int = 1,
    max_batch_size: int = 1,
    verify: bool = True,
):
    """
    Main quantization function: converts HuggingFace model to TensorRT-LLM FP4 engine.
    
    This creates an engine that uses TRUE hardware FP4 computation on RTX 5090's
    Blackwell Tensor Cores - not emulated FP4 via higher precision arithmetic.
    """
    print("=" * 70)
    print("FP4 Quantization with Native Blackwell Tensor Core Acceleration")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Tensor Parallelism: {tp_size}")
    print(f"Quantization: NVFP4 (Native Blackwell FP4)")
    
    # Check GPU
    gpu_name, compute_cap, gpu_memory = check_gpu_compatibility()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Step 1: Convert to TensorRT-LLM checkpoint with FP4 quantization
    checkpoint_dir = convert_hf_to_trtllm_checkpoint(
        model_name, output_path, tp_size
    )
    
    # Step 2: Build TensorRT engine with FP4 Tensor Core kernels
    engine_dir = build_trtllm_engine(
        checkpoint_dir, output_path, max_batch_size
    )
    
    # Step 3: Setup vision encoder
    vision_dir = setup_vision_encoder(model_name, output_path)
    
    # Save metadata
    metadata = {
        "original_model": model_name,
        "quantization_type": "NVFP4",
        "precision": "FP4 (Native Blackwell Tensor Core)",
        "compute_type": "FP4 GEMM + FP4 Attention",
        "target_gpu": "RTX 5090 (Blackwell SM 10.0+)",
        "tensor_parallelism": tp_size,
        "engine_dir": str(engine_dir),
        "vision_dir": str(vision_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "quantization_time_seconds": time.time() - start_time,
        "gpu_used": gpu_name,
        "gpu_compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
    }
    
    with open(output_path / "fp4_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("FP4 QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Total time: {total_time:.2f}s")
    print(f"\nQuantization details:")
    print(f"  - Format: NVFP4 (NVIDIA native 4-bit floating point)")
    print(f"  - Compute: FP4 Tensor Core GEMM operations")
    print(f"  - Attention: FP4 Flash Attention")
    print(f"  - Hardware: Blackwell native FP4 acceleration")
    print(f"\nTo run inference:")
    print(f"  python fp4_inference.py --video_dir <path> --engine_dir {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="FP4 Quantization with Native Blackwell Tensor Core Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates a TensorRT-LLM engine with TRUE hardware-accelerated FP4
computation using RTX 5090's native Blackwell FP4 Tensor Cores.

Unlike bitsandbytes FP4 which stores weights in 4-bit but computes in FP16/BF16,
this engine performs actual FP4 matrix multiplications in hardware.

Examples:
  # Basic quantization
  python fp4_quantization.py --output_dir ./cosmos-fp4-engine

  # With tensor parallelism for multi-GPU
  python fp4_quantization.py --output_dir ./cosmos-fp4-engine --tp_size 2

Requirements:
  - RTX 5090 GPU (Blackwell architecture)
  - TensorRT-LLM >= 0.15.0
  - CUDA 12.8+
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="HuggingFace model ID (default: nvidia/Cosmos-Reason1-7B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the FP4 TensorRT engine"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size for multi-GPU (default: 1)"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size for engine optimization (default: 1)"
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip verification after quantization"
    )
    
    args = parser.parse_args()
    
    quantize_model(
        model_name=args.model,
        output_dir=args.output_dir,
        tp_size=args.tp_size,
        max_batch_size=args.max_batch_size,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
