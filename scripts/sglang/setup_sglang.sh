#!/usr/bin/env bash
# ============================================================================
# SGLang 설치 스크립트 (채널/대상 env 선택 가능)
#
# 핀/CUDA/설치방식은 SGLANG_CHANNEL 로 결정 (env.sh 참조):
#   stable(0.5.9)  : DeepSeek-V3.2(H200) 검증. cu129. flashinfer 수동(--no-deps) 설치
#                    — 구버전 resolver 가 flashinfer 구버전을 끌어와 충돌하는 것 회피.
#   next(0.5.13)   : SM120 GDN/FP8 수정 포함. cu130. flashinfer[cu13] 정확핀 → resolver 위임.
#
# 사용법:
#   bash scripts/sglang/setup_sglang.sh                                  # stable → serving
#   CONDA_ENV=serving-next SGLANG_CHANNEL=next bash scripts/sglang/setup_sglang.sh   # next → serving-next
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/$TORCH_CUDA}"

echo "============================================================"
echo " SGLang 설치"
echo "   channel : $SGLANG_CHANNEL  →  sglang==$SGLANG_VER"
echo "   env     : $CONDA_ENV"
echo "   torch   : $TORCH_VER ($TORCH_CUDA)"
echo "   flashinfer : $FLASHINFER_VER (수동설치=$SGLANG_MANUAL_FLASHINFER)"
echo "============================================================"

source "$CONDA_HOME/etc/profile.d/conda.sh"

# env 없으면 생성 (python 3.11)
if ! conda env list | grep -qE "^[[:space:]*]*${CONDA_ENV}[[:space:]]"; then
  echo "--- conda env '$CONDA_ENV' 생성 (python 3.11) ---"
  conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"

echo "--- pip 업그레이드 ---"
python -m pip install --upgrade pip

# 1) torch 먼저 고정 설치 (채널 핀 + CUDA 인덱스)
echo "--- [torch] torch==$TORCH_VER ($TORCH_CUDA) ---"
pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" torchvision \
  --index-url "$TORCH_INDEX"

# 2) (stable 만) flashinfer/sgl-kernel 정확버전 선설치 → 구버전 끌어오기 방지
if [ "$SGLANG_MANUAL_FLASHINFER" = "1" ]; then
  echo "--- [flashinfer] $FLASHINFER_VER (--no-deps) ---"
  pip install --no-deps "flashinfer-python==$FLASHINFER_VER" || \
    pip install --no-deps "flashinfer_python==$FLASHINFER_VER"
  if [ -n "${SGL_KERNEL_VER:-}" ]; then
    echo "--- [sgl-kernel] $SGL_KERNEL_VER ---"
    pip install "sgl-kernel==$SGL_KERNEL_VER" || echo "⚠️ sgl-kernel 핀 설치 실패 (sglang 의존성으로 재시도됨)"
  fi
else
  echo "--- [flashinfer] resolver 위임 (sglang[all] 의 정확핀 사용) ---"
fi

# 3) sglang[all] 본체
echo "--- [sglang] sglang[all]==$SGLANG_VER ---"
pip install "sglang[all]==$SGLANG_VER"

# 보조: 빠른 다운로드
pip install hf_transfer

# DSA(DeepSeek Sparse Attention) prefill 백엔드 FlashMLA (H200 DeepSeek 용)
echo "--- FlashMLA(DSA) 확인 ---"
if python -c "import flash_mla" 2>/dev/null; then
  echo "  flash_mla OK"
else
  echo "  ⚠️ flash_mla 미탐지. H200 에서 DeepSeek-V3.2 기동 시 백엔드 에러가 나면:"
  echo "     pip install flash-mla   # 또는 deepseek-ai/FlashMLA 소스빌드"
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
echo "→ 완료. 기동:  CONDA_ENV=$CONDA_ENV SGLANG_CHANNEL=$SGLANG_CHANNEL bash $SCRIPTS_DIR/sglang/restart_server.sh <모델키>"
