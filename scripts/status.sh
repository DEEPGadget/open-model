#!/usr/bin/env bash
# ============================================================================
# 서버 상태 확인: 프로세스 / 포트 / HTTP health / GPU
#
# 사용법:  bash scripts/status.sh
# 파라미터: scripts/env.sh 참조
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"

# 한 줄 상태 출력 헬퍼
line() {  # $1=label  $2=ok(0/1)  $3=detail
  if [ "$2" = "0" ]; then echo "  ✅ $1  $3"; else echo "  ❌ $1  $3"; fi
}

proc_pid() { pgrep -f "$1" 2>/dev/null | head -1; }

echo "============================================================"
echo " 상태 점검   (profile=$PROFILE  served=$SERVED_NAME)"
echo "============================================================"

echo "[프로세스]"
sg_pid="$(proc_pid "sglang.launch_server.*--port ${PORT}")"
[ -n "$sg_pid" ] && line "SGLang   :${PORT}" 0 "PID $sg_pid" || line "SGLang   :${PORT}" 1 "미실행"
ll_pid="$(proc_pid "litellm.*--port ${LL_PORT}")"
[ -n "$ll_pid" ] && line "LiteLLM  :${LL_PORT}" 0 "PID $ll_pid" || line "LiteLLM  :${LL_PORT}" 1 "미실행"

echo "[HTTP health]"
if curl -fsS "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; then
  line "SGLang  /health" 0 "http://${HEALTH_HOST}:${PORT}"
else
  line "SGLang  /health" 1 "응답없음(로딩중이거나 미실행)"
fi
if curl -fsS "http://${HEALTH_HOST}:${LL_PORT}/health" >/dev/null 2>&1; then
  line "LiteLLM /health" 0 "http://${HEALTH_HOST}:${LL_PORT}"
else
  line "LiteLLM /health" 1 "응답없음"
fi

echo "[모델 (SGLang /v1/models)]"
models="$(curl -fsS "http://${HEALTH_HOST}:${PORT}/v1/models" 2>/dev/null \
  | python -c 'import sys,json; d=json.load(sys.stdin); print(", ".join(m["id"] for m in d.get("data",[])))' 2>/dev/null || true)"
[ -n "$models" ] && echo "  • $models" || echo "  (조회 불가)"

echo "[GPU]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "  GPU%s  util %3s%%   mem %6s / %6s MiB\n",$1,$2,$3,$4}'
else
  echo "  (nvidia-smi 없음)"
fi

echo "[최근 로그]"
latest_sg="$(ls -t "$LOG_DIR"/sglang_*.log 2>/dev/null | head -1 || true)"
latest_ll="$(ls -t "$LOG_DIR"/litellm_*.log 2>/dev/null | head -1 || true)"
[ -n "$latest_sg" ] && echo "  sglang : $latest_sg" || echo "  sglang : (없음)"
[ -n "$latest_ll" ] && echo "  litellm: $latest_ll" || echo "  litellm: (없음)"
echo "============================================================"
