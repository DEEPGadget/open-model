#!/usr/bin/env bash
# ============================================================================
# SGLang 설치 스크립트 (DeepSeek-V3.2 / DSA 지원)
#
# 충돌 회피 핵심:
#   - sglang[all] 을 그냥 깔면 pip이 flashinfer 구버전(0.2.x, torch 2.12 요구)을
#     끌어와 충돌함. → torch 와 flashinfer 를 "먼저, 정확한 버전"으로 고정 설치한 뒤
#     sglang 본체를 깔아 pip resolver 가 이미 충족된 것으로 인식하게 한다.
#   - 버전 핀(SGLANG_VER/TORCH_VER/FLASHINFER_VER/SGL_KERNEL_VER)은 env.sh 에 있음.
#     (sglang v0.5.9 pyproject 기준: torch 2.9.1 / flashinfer 0.6.3 / sgl-kernel 0.3.21)
#
# 사용법:  bash scripts/sglang/setup_sglang.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

# CUDA 휠 인덱스 (driver CUDA 13.2 → cu12 하위호환 휠 사용. x86_64는 pypi 기본 휠로 충분)
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"

source "$CONDA_HOME/etc/profile.d/conda.sh"

# env 없으면 생성 (python 3.11)
if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  echo "--- conda env '$CONDA_ENV' 생성 (python 3.11) ---"
  conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"

echo "--- pip 업그레이드 ---"
python -m pip install --upgrade pip

# 1) torch 먼저 고정 설치 (sglang 0.5.9 핀과 일치)
echo "--- [1/4] torch==$TORCH_VER (cu129 인덱스) ---"
pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" torchvision \
  --index-url "$TORCH_INDEX"

# 2) flashinfer 정확한 버전을 --no-deps 로 (torch 재해석 못하게)
echo "--- [2/4] flashinfer-python==$FLASHINFER_VER (--no-deps) ---"
pip install --no-deps "flashinfer-python==$FLASHINFER_VER" || \
  pip install --no-deps "flashinfer_python==$FLASHINFER_VER"

# 3) sgl-kernel 고정
echo "--- [3/4] sgl-kernel==$SGL_KERNEL_VER ---"
pip install "sgl-kernel==$SGL_KERNEL_VER" || echo "⚠️ sgl-kernel 핀 설치 실패 (sglang 의존성으로 재시도됨)"

# 4) sglang[all] 본체 (이미 충족된 torch/flashinfer 는 그대로 사용)
echo "--- [4/4] sglang[all]==$SGLANG_VER ---"
pip install "sglang[all]==$SGLANG_VER"

# 보조: 빠른 다운로드
pip install hf_transfer

# DSA(DeepSeek Sparse Attention) prefill 백엔드 FlashMLA
# 대부분 flashinfer/sgl-kernel 에 포함되나, 없으면 별도 설치 안내
echo "--- FlashMLA(DSA) 확인 ---"
if python -c "import flash_mla" 2>/dev/null; then
  echo "  flash_mla OK"
else
  echo "  ⚠️ flash_mla 미탐지. V3.2 기동 시 백엔드 에러가 나면 아래로 설치:"
  echo "     pip install flash-mla   # 또는 https://github.com/deepseek-ai/FlashMLA 소스빌드"
fi

echo "=== 검증 ==="
python - <<'PY'
import importlib
for m in ["torch","sglang","flashinfer"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:12s} IMPORT 실패: {e}")
import torch
print("  cuda runtime", torch.version.cuda, "| gpus", torch.cuda.device_count())
PY
echo "→ 완료. 기동:  bash $SCRIPTS_DIR/sglang/restart_server.sh"
