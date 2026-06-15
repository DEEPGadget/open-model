#!/usr/bin/env bash
# ============================================================================
# SGLang 프로세스 제어 헬퍼 (restart_server.sh / stop_sglang.sh 공용)
# ============================================================================

# 해당 포트의 sglang 메인 프로세스 PID들
_sglang_pids() { pgrep -f "sglang.launch_server.*--port[= ]${1}" 2>/dev/null || true; }
# sglang 워커/서브프로세스(이름 sglang::...) 까지 포함한 잔여 PID
_sglang_all_pids() {
  { pgrep -f "sglang.launch_server.*--port[= ]${1}" 2>/dev/null; pgrep -f "sglang::" 2>/dev/null; } | sort -u || true
}

# 지정 포트의 sglang 서버를 종료. 우아하게(SIGTERM)→대기→강제(SIGKILL).
# 사용: sglang_stop [port]   (기본 $PORT)
sglang_stop() {
  local port="${1:-$PORT}" pids i
  pids="$(_sglang_pids "$port")"
  if [ -z "$pids" ]; then
    echo "  기존 sglang 없음 (port=$port)"
    return 0
  fi
  echo "  기존 sglang 종료 (port=$port, PID: $(echo $pids | tr '\n' ' ')) → SIGTERM"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  # 우아한 종료 대기 (최대 30s) — sglang 이 워커들을 정리함
  for i in $(seq 1 30); do
    [ -z "$(_sglang_pids "$port")" ] && break
    sleep 1
  done
  # 잔여(메인+워커) 강제 종료
  local leftover; leftover="$(_sglang_all_pids "$port")"
  if [ -n "$leftover" ]; then
    echo "  잔여 프로세스 SIGKILL: $(echo $leftover | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -KILL $leftover 2>/dev/null || true
    sleep 2
  fi
  echo "  ✅ 종료 완료 (port=$port)"
}

# 서버가 health 응답할 때까지 대기 (백그라운드 기동용)
# 사용: sglang_wait_ready <host> <port> <timeout_sec> <pid>
sglang_wait_ready() {
  local host="$1" port="$2" timeout="$3" pid="$4" i
  for i in $(seq 1 "$timeout"); do
    if curl -fsS --max-time 3 "http://${host}:${port}/health" >/dev/null 2>&1; then
      echo "  ✅ health OK (${i}s)"; return 0
    fi
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "  ❌ 프로세스가 종료됨 (${i}s) — 로그 확인 필요"; return 1
    fi
    sleep 1
  done
  echo "  ⏱️ ${timeout}s 내 미준비"; return 1
}
