#!/usr/bin/env bash
set -Eeuo pipefail

echo "========================================================"
echo "   VTRACE: COMPLETE SYSTEM & DATA SETUP"
echo "========================================================"

# Ensure we are in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 1. Check for uv command
if ! command -v uv &> /dev/null; then
    echo "Không tìm thấy lệnh 'uv'. Đang tự động cài đặt uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "${HOME}/.local/bin/env"
fi

# 2. Run 3DGS and environment setup
echo ""
echo ">>> [1/3] Khởi tạo môi trường VTRACE và 3DGS legacy..."

mkdir -p src/vtrace

if [ ! -d "src/vtrace/gaussian-splatting" ]; then
    echo "Cloning gaussian-splatting repository..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting src/vtrace/gaussian-splatting
else
    echo "gaussian-splatting repository already exists."
fi

echo "Applying custom patches to 3DGS repository..."
if [ -f "src/vtrace/patches/dataset_readers.py" ]; then
    cp src/vtrace/patches/dataset_readers.py src/vtrace/gaussian-splatting/scene/dataset_readers.py
fi
if [ -f "src/vtrace/patches/train.py" ]; then
    cp src/vtrace/patches/train.py src/vtrace/gaussian-splatting/train.py
fi
if [ -f "src/vtrace/patches/arguments_init.py" ]; then
    cp src/vtrace/patches/arguments_init.py src/vtrace/gaussian-splatting/arguments/__init__.py
fi
if [ -f "src/vtrace/patches/cameras.py" ]; then
    cp src/vtrace/patches/cameras.py src/vtrace/gaussian-splatting/scene/cameras.py
fi
if [ -f "src/vtrace/patches/gaussian_model.py" ]; then
    cp src/vtrace/patches/gaussian_model.py src/vtrace/gaussian-splatting/scene/gaussian_model.py
fi
if [ -f "src/vtrace/patches/rasterizer_impl.h" ]; then
    cp src/vtrace/patches/rasterizer_impl.h src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
fi
if [ -f "src/vtrace/patches/camera_utils.py" ]; then
    cp src/vtrace/patches/camera_utils.py src/vtrace/gaussian-splatting/utils/camera_utils.py
fi

echo "Syncing Python environment with uv (installing PyTorch, OpenCV, etc.)..."
uv sync

echo "Checking CUDA version and aligning environment..."

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-12.1" ]; then
        export CUDA_HOME="/usr/local/cuda-12.1"
    fi
fi

if command -v nvcc &> /dev/null; then
    SYS_CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "Detected system NVCC version: $SYS_CUDA_VERSION"
    
    if [ "$SYS_CUDA_VERSION" != "12.1" ] && [ -d "/usr/local/cuda-12.1" ]; then
        echo "Found CUDA 12.1 toolkit. Switching build tools to CUDA 12.1 to match default PyTorch..."
        export CUDA_HOME="/usr/local/cuda-12.1"
        export PATH="/usr/local/cuda-12.1/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
        SYS_CUDA_VERSION="12.1"
    fi
    
    if [ "$SYS_CUDA_VERSION" != "12.1" ]; then
        if [ "$SYS_CUDA_VERSION" = "12.4" ]; then
            echo "System CUDA is 12.4. Reinstalling PyTorch and gsplat to match CUDA 12.4..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
            uv pip install --force-reinstall gsplat
        elif [ "$SYS_CUDA_VERSION" = "12.6" ]; then
            echo "System CUDA is 12.6. Reinstalling PyTorch and gsplat to match CUDA 12.6..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
            uv pip install --force-reinstall gsplat
        elif [ "$SYS_CUDA_VERSION" = "11.8" ]; then
            echo "System CUDA is 11.8. Reinstalling PyTorch and gsplat to match CUDA 11.8..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            uv pip install --force-reinstall gsplat
        else
            echo "Warning: System CUDA version is $SYS_CUDA_VERSION. If build fails, manually run matching PyTorch install."
        fi
    fi
else
    echo "Warning: nvcc not found in PATH. Ensure CUDA Toolkit is installed."
fi

echo "Patching PyTorch's cpp_extension.py to bypass CUDA mismatch verification..."
.venv/bin/python -c '
import torch.utils.cpp_extension as m
path = m.__file__
with open(path, "r") as f:
    code = f.read()
target = "raise RuntimeError(CUDA_MISMATCH_MESSAGE"
if target in code:
    print("Found CUDA mismatch check, patching it...")
    code = code.replace(target, "print(\"Warning: CUDA mismatch warning bypassed\"); # raise RuntimeError(CUDA_MISMATCH_MESSAGE")
    with open(path, "w") as f:
        f.write(code)
    print("Patch applied successfully.")
else:
    print("CUDA mismatch check not found or already patched.")
'

echo "Installing remaining submodules using uv..."
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/simple-knn
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/fused-ssim

# 3. Install the pinned max-quality NVIDIA 3DGRUT + NHT environment.
echo ""
echo ">>> [2/3] Cài NVIDIA 3DGRUT + 3DGUT-MCMC + Neural Harmonic Textures..."

FRAMEWORK_DIR="${VTRACE_3DGRUT_DIR:-${PROJECT_ROOT}/external/3dgrut}"
FRAMEWORK_PARENT="$(dirname "${FRAMEWORK_DIR}")"
FRAMEWORK_NAME="$(basename "${FRAMEWORK_DIR}")"
FRAMEWORK_READY="${FRAMEWORK_PARENT}/.${FRAMEWORK_NAME}.vtrace-nht-ready"
FRAMEWORK_ENV="${FRAMEWORK_PARENT}/.${FRAMEWORK_NAME}.vtrace-nht-env"
THREEDGRUT_REPOSITORY="https://github.com/nv-tlabs/3dgrut.git"
THREEDGRUT_COMMIT="a37ef721012dea0f29c0fcfff2d525023b4e854a"

mkdir -p "${FRAMEWORK_PARENT}"
if [ ! -d "${FRAMEWORK_DIR}/.git" ]; then
    echo "Cloning pinned 3DGRUT/NHT repository..."
    git clone --recursive "${THREEDGRUT_REPOSITORY}" "${FRAMEWORK_DIR}"
else
    echo "3DGRUT repository already exists at ${FRAMEWORK_DIR}."
fi

FRAMEWORK_DIRTY="$(git -C "${FRAMEWORK_DIR}" status --porcelain)"
if [ -n "${FRAMEWORK_DIRTY}" ]; then
    echo "ERROR: 3DGRUT checkout có thay đổi cục bộ; không thể checkout an toàn: ${FRAMEWORK_DIR}" >&2
    exit 20
fi

if ! git -C "${FRAMEWORK_DIR}" cat-file -e "${THREEDGRUT_COMMIT}^{commit}" 2>/dev/null; then
    echo "Fetching pinned 3DGRUT commit ${THREEDGRUT_COMMIT}..."
    git -C "${FRAMEWORK_DIR}" fetch origin "${THREEDGRUT_COMMIT}"
fi
git -C "${FRAMEWORK_DIR}" checkout --detach "${THREEDGRUT_COMMIT}"
git -C "${FRAMEWORK_DIR}" submodule update --init --recursive

NHT_CUDA_HOME=""
for candidate in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-11.8 /usr/local/cuda-13.0 /usr/local/cuda; do
    if [ ! -x "${candidate}/bin/nvcc" ]; then
        continue
    fi
    candidate_version="$("${candidate}/bin/nvcc" --version | awk '/release/{print $5}' | cut -d, -f1)"
    case "${candidate_version}" in
        12.8|12.6|12.4|11.8|13.0)
            NHT_CUDA_HOME="${candidate}"
            break
            ;;
    esac
done

if [ -z "${NHT_CUDA_HOME}" ]; then
    echo "ERROR: NHT cần CUDA Toolkit 11.8, 12.4, 12.6, 12.8 hoặc 13.0." >&2
    echo "Khuyên dùng image CUDA 12.8 devel trên server 48/80GB." >&2
    exit 21
fi

NHT_CUDA_VERSION="$("${NHT_CUDA_HOME}/bin/nvcc" --version | awk '/release/{print $5}' | cut -d, -f1)"
echo "3DGRUT/NHT will build with CUDA ${NHT_CUDA_VERSION} at ${NHT_CUDA_HOME}."

READY_COMMIT=""
if [ -f "${FRAMEWORK_READY}" ]; then
    READY_COMMIT="$(tr -d '[:space:]' < "${FRAMEWORK_READY}")"
fi

if [ "${READY_COMMIT}" = "${THREEDGRUT_COMMIT}" ] && [ -x "${FRAMEWORK_DIR}/.venv/bin/python" ]; then
    echo "Pinned 3DGRUT/NHT environment is already installed; verifying it..."
else
    echo "Installing isolated 3DGRUT/NHT environment (this compiles CUDA extensions)..."
    (
        cd "${FRAMEWORK_DIR}"
        unset VIRTUAL_ENV || true
        export CUDA_HOME="${NHT_CUDA_HOME}"
        export PATH="${NHT_CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${NHT_CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        ./install_env_uv.sh
    )
fi

ACTUAL_3DGRUT_COMMIT="$(git -C "${FRAMEWORK_DIR}" rev-parse HEAD)"
if [ "${ACTUAL_3DGRUT_COMMIT}" != "${THREEDGRUT_COMMIT}" ]; then
    echo "ERROR: 3DGRUT commit mismatch: ${ACTUAL_3DGRUT_COMMIT}" >&2
    exit 22
fi
if [ ! -f "${FRAMEWORK_DIR}/configs/apps/colmap_3dgut_mcmc_nht.yaml" ]; then
    echo "ERROR: NHT config is missing from the pinned 3DGRUT checkout." >&2
    exit 23
fi
(
    export CUDA_HOME="${NHT_CUDA_HOME}"
    export PATH="${NHT_CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${NHT_CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    "${FRAMEWORK_DIR}/.venv/bin/python" -c 'import torch, ncore, omegaconf; print("3DGRUT environment OK; torch", torch.__version__)'
)
printf 'CUDA_HOME=%s\n' "${NHT_CUDA_HOME}" > "${FRAMEWORK_ENV}"
printf '%s\n' "${THREEDGRUT_COMMIT}" > "${FRAMEWORK_READY}"

# 4. Download data
echo ""
echo ">>> [3/3] Tải dữ liệu VAI_NVS_DATA từ Google Drive..."
FILE_ID="12vOrYdBT_0yrvV48pf--yXaSzXD5QONV"

download_and_extract() {
    echo "Đang tải VAI_NVS_DATA.zip từ Google Drive..."
    uvx gdown "${FILE_ID}" -O VAI_NVS_DATA.zip
    echo "Đang giải nén VAI_NVS_DATA.zip..."
    if command -v unzip &> /dev/null; then
        unzip -q VAI_NVS_DATA.zip
    else
        python3 -m zipfile -e VAI_NVS_DATA.zip .
    fi
    rm -f VAI_NVS_DATA.zip
    
    # Wrap phase1 in VAI_NVS_DATA if extracted directly
    if [ -d "phase1" ]; then
        mkdir -p VAI_NVS_DATA
        mv phase1 VAI_NVS_DATA/
    fi
    rm -rf __MACOSX
}

if [ -d "VAI_NVS_DATA" ]; then
    RE_DOWNLOAD="${VTRACE_REDOWNLOAD_DATA:-n}"
    if [ "$RE_DOWNLOAD" = "y" ] || [ "$RE_DOWNLOAD" = "Y" ]; then
        download_and_extract
    else
        echo "VAI_NVS_DATA đã tồn tại; tự động bỏ qua tải lại. Đặt VTRACE_REDOWNLOAD_DATA=y nếu cần tải lại."
    fi
else
    download_and_extract
fi

echo ""
echo "========================================================"
echo "   Hệ thống VTRACE đã thiết lập thành công!"
echo "   Max-quality framework: 3DGRUT/NHT ${THREEDGRUT_COMMIT}"
echo "   Bước kế tiếp: ./scripts/launch_nht_max.sh --smoke-test --data-dir VAI_NVS_DATA/phase1/public_set --scene HCM0181"
echo "========================================================"
