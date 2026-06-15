#!/usr/bin/env bash
# ============================================================================
# SGLang 서버만 종료 (지정 포트)
#
# 사용법:  bash scripts/sglang/stop_sglang.sh
#   PORT=30001 bash scripts/sglang/stop_sglang.sh   # 다른 포트
# 종료 로직은 lib/server_ctl.sh 의 sglang_stop (SIGTERM→대기→SIGKILL).
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$SCRIPT_DIR/../env.sh"

echo "--- SGLang 종료 (port=$PORT) ---"
sglang_stop "$PORT"

# 백그라운드 기동 시 남긴 pid 파일 정리 (있으면)
rm -f "$LOG_DIR/sglang_${MODEL_KEY}.pid" 2>/dev/null || true
