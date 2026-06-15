#!/usr/bin/env bash
# ============================================================================
# LiteLLM proxy 만 종료
#
# 사용법:  bash scripts/litellm/stop_litellm.sh
#   FORCE=1 bash scripts/litellm/stop_litellm.sh   # SIGKILL 강제 종료
# 파라미터: scripts/env.sh 참조 (LL_PORT / LOG_DIR)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

SIG="TERM"; [ "${FORCE:-0}" = "1" ] && SIG="KILL"
PIDFILE="$LOG_DIR/litellm.pid"

echo "--- LiteLLM 종료 (port=$LL_PORT, SIG$SIG) ---"

# 1) PID 파일 기반
if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "  - PID파일 $pid → SIG$SIG"
    kill "-$SIG" "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

# 2) 포트 패턴 기반
pids="$(pgrep -f "litellm.*--port ${LL_PORT}" 2>/dev/null || true)"
if [ -n "$pids" ]; then
  echo "  - 패턴매칭 PID: $pids → SIG$SIG"
  # shellcheck disable=SC2086
  kill "-$SIG" $pids 2>/dev/null || true
fi

sleep 2
remaining="$(pgrep -f "litellm.*--port ${LL_PORT}" 2>/dev/null || true)"
if [ -n "$remaining" ]; then
  echo "⚠️ 아직 살아있음: $remaining   →  FORCE=1 bash $SCRIPTS_DIR/litellm/stop_litellm.sh"
  exit 1
else
  echo "✅ LiteLLM 종료 완료"
fi
