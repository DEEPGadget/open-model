#!/usr/bin/env bash
# ============================================================================
# LiteLLM Proxy 기동 - SGLang 을 OpenAI/Anthropic API로 노출
#
# 사용법:  bash scripts/litellm/start_litellm.sh
# 엔드포인트(기동 후):
#   OpenAI    : http://<host>:<LL_PORT>/v1/chat/completions
#   Anthropic : http://<host>:<LL_PORT>/v1/messages
#   인증      : Authorization: Bearer $LITELLM_MASTER_KEY
# 파라미터: scripts/env.sh 참조
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

activate_conda

# litellm 설치 확인
if ! python -c "import litellm" 2>/dev/null; then
  echo "litellm 미설치 → 설치합니다 (litellm[proxy])..."
  pip install "litellm[proxy]"
fi

# 백엔드(SGLang) 살아있는지 먼저 확인
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "⚠️ SGLang 백엔드(:${PORT})가 응답하지 않습니다. 먼저 restart_server.sh 로 기동/로딩 완료 후 실행하세요." >&2
fi

echo "--- 기존 litellm 정리 ---"
pkill -f "litellm.*--port ${LL_PORT}" 2>/dev/null && sleep 2 || echo "(기존 없음)"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/litellm_$(date +%Y%m%d_%H%M%S).log"

echo "--- LiteLLM proxy 기동 (config: $LITELLM_CONFIG, 로그: $LOG_FILE) ---"
nohup litellm --config "$LITELLM_CONFIG" --host "$LL_HOST" --port "$LL_PORT" \
  > "$LOG_FILE" 2>&1 &

echo "$!" > "$LOG_DIR/litellm.pid"
echo "→ PID $! 로 기동. 확인: tail -f $LOG_FILE"
echo ""
echo "테스트 (OpenAI 스타일):"
echo "  curl http://127.0.0.1:${LL_PORT}/v1/chat/completions \\"
echo "    -H \"Authorization: Bearer \$LITELLM_MASTER_KEY\" -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
echo ""
echo "테스트 (Anthropic 스타일):"
echo "  curl http://127.0.0.1:${LL_PORT}/v1/messages \\"
echo "    -H \"x-api-key: \$LITELLM_MASTER_KEY\" -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVED_NAME}\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
