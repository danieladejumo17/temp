#!/bin/bash
# =============================================================================
# FP4 Environment Activation Script
# =============================================================================
#
# Source this file to set up the environment for FP4 quantization on Blackwell
#
# Usage:
#   source activate_fp4.sh           # Quick activation (no verification)
#   source activate_fp4.sh --verify  # Activate with package verification
#
# This script:
#   1. Adds CUDA 13 libraries to LD_LIBRARY_PATH (required for TensorRT-LLM)
#   2. Activates virtual environment if present
#   3. Optionally verifies the environment (--verify flag)
#
# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for --verify flag
VERIFY_ENV=false
for arg in "$@"; do
    if [ "$arg" == "--verify" ]; then
        VERIFY_ENV=true
    fi
done

# =============================================================================
# Find and set CUDA 13 library path
# =============================================================================

# Try to find the CUDA 13 library path
PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

if [ -n "$PYTHON_SITE_PACKAGES" ]; then
    CUDA13_LIB_PATH="$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"
    
    if [ -d "$CUDA13_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$CUDA13_LIB_PATH:${LD_LIBRARY_PATH}"
        echo -e "${GREEN}[FP4]${NC} Added CUDA 13 libraries to LD_LIBRARY_PATH"
    else
        # Try alternate location
        CUDA13_LIB_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
        if [ -d "$CUDA13_LIB_PATH" ]; then
            export LD_LIBRARY_PATH="$CUDA13_LIB_PATH:${LD_LIBRARY_PATH}"
            echo -e "${GREEN}[FP4]${NC} Added CUDA 13 libraries to LD_LIBRARY_PATH"
        else
            echo -e "${YELLOW}[FP4]${NC} Warning: CUDA 13 library path not found"
            echo "       You may need to install nvidia-cublas>=13.0.0"
        fi
    fi
else
    echo -e "${YELLOW}[FP4]${NC} Warning: Could not determine Python site-packages path"
fi

# =============================================================================
# Activate virtual environment if it exists
# =============================================================================

VENV_PATHS=(
    "$SCRIPT_DIR/fp4_env/bin/activate"
    "$SCRIPT_DIR/venv/bin/activate"
    "$SCRIPT_DIR/.venv/bin/activate"
)

for venv_path in "${VENV_PATHS[@]}"; do
    if [ -f "$venv_path" ]; then
        source "$venv_path"
        echo -e "${GREEN}[FP4]${NC} Activated virtual environment: $(dirname $(dirname $venv_path))"
        break
    fi
done

# =============================================================================
# Verify environment (optional, use --verify flag)
# =============================================================================

if $VERIFY_ENV; then
    echo ""
    echo "FP4 Environment Status (verifying packages...):"
    
    # Check TensorRT-LLM
    TRTLLM_VERSION=$(python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>/dev/null)
    if [ -n "$TRTLLM_VERSION" ]; then
        echo "  ✓ TensorRT-LLM: $TRTLLM_VERSION"
    else
        echo "  ✗ TensorRT-LLM: Not available (check LD_LIBRARY_PATH)"
    fi
    
    # Check ModelOpt
    MODELOPT_VERSION=$(python3 -c "import modelopt; print(modelopt.__version__)" 2>/dev/null)
    if [ -n "$MODELOPT_VERSION" ]; then
        echo "  ✓ ModelOpt: $MODELOPT_VERSION"
    else
        echo "  ✗ ModelOpt: Not available"
    fi
    
    # Check PyTorch and CUDA
    python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'  ✓ PyTorch: {torch.__version__}')
    print(f'  ✓ CUDA GPU: {gpu_name}')
    print(f'  ✓ Compute Capability: {compute_cap[0]}.{compute_cap[1]}')
    if compute_cap[0] >= 10:
        print('  ✓ Blackwell FP4 Tensor Cores: Available')
    else:
        print('  ⚠ Blackwell FP4 Tensor Cores: Not available (SM 10.0+ required)')
else:
    print('  ✗ CUDA: Not available')
" 2>/dev/null || echo "  ✗ PyTorch/CUDA: Check failed"
fi

echo ""
echo -e "${GREEN}[FP4]${NC} Environment activated. Ready for FP4 quantization."
echo "       Run 'source activate_fp4.sh --verify' to check all packages."
