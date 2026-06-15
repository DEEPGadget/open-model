#!/usr/bin/env bash
# ============================================================================
# 전체 환경 셋업 — sglang 서빙 env + litellm proxy env 를 각각 구성
#
#   1) sglang  → $CONDA_ENV (기본 sglang)   : sglang 0.5.13 + torch/flashinfer
#   2) litellm → $LITELLM_ENV (기본 litellm) : litellm[proxy]
#   두 env 를 분리하는 이유: openai 버전 핀이 서로 충돌하기 때문.
#
# 사용법:
#   bash scripts/setup.sh                 # 둘 다 설치(없으면 생성)
#   RECREATE=1 bash scripts/setup.sh      # 둘 다 삭제 후 재생성(클린)
#   bash scripts/setup.sh sglang          # sglang env 만
#   bash scripts/setup.sh litellm         # litellm env 만
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

TARGET="${1:-all}"

run_sglang()  { echo; echo "########## sglang env 셋업 ($CONDA_ENV) ##########"; bash "$SCRIPTS_DIR/sglang/setup_sglang.sh"; }
run_litellm() { echo; echo "########## litellm env 셋업 ($LITELLM_ENV) ##########"; bash "$SCRIPTS_DIR/litellm/setup_litellm.sh"; }

case "$TARGET" in
  all)     run_sglang; run_litellm ;;
  sglang)  run_sglang ;;
  litellm) run_litellm ;;
  *) echo "사용: setup.sh [all|sglang|litellm]" >&2; exit 1 ;;
esac

echo
echo "=== 셋업 완료 ==="
echo "  sglang  env : $CONDA_ENV"
echo "  litellm env : $LITELLM_ENV"
echo "  다음:  bash $SCRIPTS_DIR/sglang/restart_server.sh <모델키> --bg"
echo "         bash $SCRIPTS_DIR/litellm/start_litellm.sh"
