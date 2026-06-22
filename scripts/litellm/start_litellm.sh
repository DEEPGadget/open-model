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

# 전용 litellm env 활성화 (sglang env 와 분리 — openai 핀 충돌 방지)
activate_conda "$LITELLM_ENV"

# --- 기동 시 외부 인터넷 의존성 제거 -----------------------------------------
# litellm 은 import 시 GitHub 에서 모델 코스트맵을 httpx 로 가져오는데(5s timeout),
# 오프라인/DNS 지연 상태로 기동하면 이 페치가 init 을 오염시켜 이후 백엔드 호출이
# "APIConnectionError: Connection error." 로 떨어지는 현상이 있었다(기동 시점 상태가
# 프로세스 수명 내내 고정됨). 아래 3개로 startup 시 외부 호출을 전부 차단한다.
export LITELLM_LOCAL_MODEL_COST_MAP="True"          # GitHub 코스트맵 페치 끔 → 번들 백업만
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"   # 백엔드(localhost)는 프록시 우회
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0}"

# litellm 설치 확인
if ! python -c "import litellm" 2>/dev/null; then
  echo "❌ '$LITELLM_ENV' env 에 litellm 미설치." >&2
  echo "   설치:  conda create -y -n $LITELLM_ENV python=3.11 && conda activate $LITELLM_ENV && pip install 'litellm[proxy]'" >&2
  exit 1
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
# litellm config 의 model_name 은 서빙명에 -oai / -anth 접미사 (litellm_config.yaml 참조)
echo "테스트 (OpenAI 스타일):"
echo "  curl http://127.0.0.1:${LL_PORT}/v1/chat/completions \\"
echo "    -H \"Authorization: Bearer \$LITELLM_MASTER_KEY\" -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVED_NAME}-oai\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
echo ""
echo "테스트 (Anthropic 스타일):"
echo "  curl http://127.0.0.1:${LL_PORT}/v1/messages \\"
echo "    -H \"x-api-key: \$LITELLM_MASTER_KEY\" -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVED_NAME}-anth\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
