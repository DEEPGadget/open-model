#!/usr/bin/env bash
# ============================================================================
# LiteLLM 설치 (전용 env)
#
#   LiteLLM 은 sglang 과 openai 버전 핀이 충돌하므로 별도 env 에 격리한다.
#   (proxy 는 HTTP 호출만 하므로 GPU/torch/sglang 불필요)
#
#   대상 env : $LITELLM_ENV (기본 litellm)
#
# 사용법:
#   bash scripts/litellm/setup_litellm.sh
#   RECREATE=1 bash scripts/litellm/setup_litellm.sh   # 기존 env 삭제 후 재생성
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "============================================================"
echo " LiteLLM 설치  (env=$LITELLM_ENV)"
echo "============================================================"

source "$CONDA_HOME/etc/profile.d/conda.sh"

if [ "${RECREATE:-0}" = "1" ] && conda env list | grep -qE "^[[:space:]*]*${LITELLM_ENV}[[:space:]]"; then
  echo "--- RECREATE=1 → 기존 env '$LITELLM_ENV' 삭제 ---"
  conda deactivate 2>/dev/null || true
  conda env remove -y -n "$LITELLM_ENV"
fi

if ! conda env list | grep -qE "^[[:space:]*]*${LITELLM_ENV}[[:space:]]"; then
  echo "--- conda env '$LITELLM_ENV' 생성 (python 3.11) ---"
  conda create -y -n "$LITELLM_ENV" python=3.11
fi
conda activate "$LITELLM_ENV"

echo "--- pip 업그레이드 ---"
python -m pip install --upgrade pip

echo "--- litellm[proxy] 설치 ---"
pip install "litellm[proxy]"

echo "=== 검증 ==="
python - <<'PY'
import litellm
from importlib.metadata import version
print("  litellm", version("litellm"))
PY
echo "  litellm CLI: $(which litellm 2>/dev/null || echo '없음')"
echo "→ 완료. 기동:  bash $SCRIPTS_DIR/litellm/start_litellm.sh"
