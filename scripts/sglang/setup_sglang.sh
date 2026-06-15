#!/usr/bin/env bash
# ============================================================================
# SGLang 설치 (sglang 서빙 env, 0.5.13 단일)
#
#   대상 env  : $CONDA_ENV (기본 serving)
#   버전      : sglang 0.5.13 / torch 2.11.0 (cu130) / flashinfer[cu13] 0.6.12
#   설치순서  : torch 먼저(정확버전·CUDA 인덱스) → sglang[all] (resolver 가 flashinfer 정확핀 사용)
#
# 사용법:
#   bash scripts/sglang/setup_sglang.sh              # 기본 serving env
#   RECREATE=1 bash scripts/sglang/setup_sglang.sh   # 기존 env 삭제 후 재생성(클린)
#   CONDA_ENV=other bash scripts/sglang/setup_sglang.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/$TORCH_CUDA}"

echo "============================================================"
echo " SGLang 설치  (env=$CONDA_ENV)"
echo "   sglang     : $SGLANG_VER"
echo "   torch      : $TORCH_VER ($TORCH_CUDA)"
echo "   flashinfer : $FLASHINFER_VER (resolver 가 sglang 핀으로 설치)"
echo "============================================================"

source "$CONDA_HOME/etc/profile.d/conda.sh"

# 클린 재설치 옵션
if [ "${RECREATE:-0}" = "1" ] && conda env list | grep -qE "^[[:space:]*]*${CONDA_ENV}[[:space:]]"; then
  echo "--- RECREATE=1 → 기존 env '$CONDA_ENV' 삭제 ---"
  conda deactivate 2>/dev/null || true
  conda env remove -y -n "$CONDA_ENV"
fi

# env 없으면 생성 (python 3.11)
if ! conda env list | grep -qE "^[[:space:]*]*${CONDA_ENV}[[:space:]]"; then
  echo "--- conda env '$CONDA_ENV' 생성 (python 3.11) ---"
  conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"

echo "--- pip 업그레이드 ---"
python -m pip install --upgrade pip

# 1) torch 먼저 고정 설치 (정확버전 + CUDA 휠 인덱스)
echo "--- [torch] torch==$TORCH_VER ($TORCH_CUDA) ---"
pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" torchvision \
  --index-url "$TORCH_INDEX"

# 2) sglang[all] 본체 (flashinfer[cu13]==$FLASHINFER_VER 등은 sglang 의 정확핀으로 resolver 가 설치)
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
echo "→ 완료. 기동:  bash $SCRIPTS_DIR/sglang/restart_server.sh <모델키>"
