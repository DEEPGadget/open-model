#!/usr/bin/env bash
# ============================================================================
# 전체 서버 종료: LiteLLM proxy → SGLang
#   (proxy를 먼저 내려 백엔드 종료 중 들어오는 요청을 막음)
#   실제 종료 로직은 컴포넌트별 stop 스크립트에 위임.
#
# 사용법:  bash scripts/stop_all.sh
#   FORCE=1 bash scripts/stop_all.sh   # SIGKILL 강제 종료(하위로 전달)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "########## 1/2  LiteLLM 종료 ##########"
bash "$SCRIPTS_DIR/litellm/stop_litellm.sh" || true

echo "########## 2/2  SGLang 종료 ##########"
bash "$SCRIPTS_DIR/sglang/stop_sglang.sh" || true

echo "✅ 전체 종료 절차 완료"
