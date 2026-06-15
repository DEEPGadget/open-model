#!/usr/bin/env bash
# ============================================================================
# SGLang 서버 헬스체크 + 스모크 테스트
# 사용법:  bash scripts/healthcheck.sh
# 파라미터: PORT / SERVED_NAME / HEALTH_HOST  (scripts/env.sh 참조)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# 접속 대상 호스트: 0.0.0.0(바인드 주소)이면 127.0.0.1 로 대체
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
BASE="http://${HEALTH_HOST}:${PORT}"

echo "=== 1) /health ($BASE) ==="
if curl -fsS "${BASE}/health" >/dev/null 2>&1; then
  echo "✅ health OK"
else
  echo "⏳ 아직 준비 안 됨 (로딩 중일 수 있음). 로그 확인: tail -f $LOG_DIR/*.log"
  exit 1
fi

echo "=== 2) /v1/models ==="
curl -fsS "${BASE}/v1/models" | python -m json.tool 2>/dev/null || echo "(models 조회 실패)"

echo "=== 3) 단일 completion 스모크 테스트 ==="
curl -fsS "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say 'pong' and nothing else.\"}],
    \"max_tokens\": 16,
    \"temperature\": 0
  }" | python -m json.tool 2>/dev/null || echo "❌ completion 실패"
