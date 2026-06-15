#!/usr/bin/env bash
# ============================================================================
# 전체 서버 기동: SGLang → (헬스 대기) → LiteLLM proxy
#
# 사용법:  bash scripts/start_all.sh
#   SGLANG_WAIT=600 bash scripts/start_all.sh   # SGLang 로딩 대기 최대 600초
# 파라미터: scripts/env.sh 참조
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SGLANG_WAIT="${SGLANG_WAIT:-900}"          # SGLang health 대기 한도(초). 가중치 로딩이 오래걸림
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"

echo "########## 1/3  SGLang 기동 ##########"
bash "$SCRIPTS_DIR/sglang/restart_server.sh"

echo ""
echo "########## 2/3  SGLang health 대기 (최대 ${SGLANG_WAIT}s) ##########"
deadline=$((SECONDS + SGLANG_WAIT))
until curl -fsS "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "❌ ${SGLANG_WAIT}s 내 SGLang이 준비되지 않음. 로그 확인: tail -f $LOG_DIR/sglang_*.log" >&2
    exit 1
  fi
  sleep 5
  printf '.'
done
echo ""
echo "✅ SGLang ready"

echo ""
echo "########## 3/3  LiteLLM proxy 기동 ##########"
bash "$SCRIPTS_DIR/litellm/start_litellm.sh"

echo ""
echo "=== 전체 기동 완료 ==="
echo "→ 상태:  bash $SCRIPTS_DIR/status.sh"
echo "→ 종료:  bash $SCRIPTS_DIR/stop_all.sh"
