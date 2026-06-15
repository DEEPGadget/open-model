#!/usr/bin/env bash
# ============================================================================
# SGLang 서버 (재)기동 스크립트
#
#   동작: 기존 서버(같은 포트) 종료  →  새 서버 기동
#   기본은 foreground (직접 python 호출과 동일하게 출력이 터미널에 보임, Ctrl-C 로 중단).
#   --bg 주면 백그라운드(nohup+로그파일)로 띄우고 health 대기.
#
#   모델 정체성 → registry/models.sh, HW 규칙 → registry/compat_rules.sh (env.sh 가 로드)
#
# 사용법:
#   bash scripts/sglang/restart_server.sh <모델키> [--bg] [--ctx N]
#     <모델키>: deepseek | deepseek-nvfp4 | qwen32 | qwen27
#   예)
#     bash scripts/sglang/restart_server.sh qwen27            # foreground
#     bash scripts/sglang/restart_server.sh qwen32 --bg       # 백그라운드
#     TP_SIZE=4 bash scripts/sglang/restart_server.sh deepseek
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 인자 파싱 --------------------------------------------------------------
MODEL_KEY_ARG=""; CTX_OVERRIDE=""; BACKGROUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help|list)
      echo "사용: restart_server.sh <deepseek|deepseek-nvfp4|qwen32|qwen27> [--bg] [--ctx N]"; exit 0 ;;
    --bg|--background) BACKGROUND=1; shift ;;
    --ctx)   CTX_OVERRIDE="${2:-}"; shift 2 ;;
    --ctx=*) CTX_OVERRIDE="${1#*=}"; shift ;;
    *)       MODEL_KEY_ARG="$1"; shift ;;
  esac
done
[ -n "$MODEL_KEY_ARG" ] && export MODEL_KEY="$MODEL_KEY_ARG"

# 셸에 남은 stale 값 제거 → env.sh 가 키 기반/auto 로 새로 계산
unset CONTEXT_LEN
# shellcheck source=../env.sh
source "$SCRIPT_DIR/../env.sh"
[ -n "$CTX_OVERRIDE" ] && CONTEXT_LEN="$CTX_OVERRIDE"

# --- conda 활성화 + sglang/모델 점검 ----------------------------------------
activate_conda
if ! python -c "import sglang" 2>/dev/null; then
  echo "❌ '$CONDA_ENV' env 에 sglang 미설치 → bash $SCRIPTS_DIR/sglang/setup_sglang.sh" >&2; exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "❌ 모델 없음: $MODEL_DIR/config.json → bash $SCRIPTS_DIR/download_model.sh <repo> $MODEL_NAME" >&2; exit 1
fi

# --- 호환성 규칙 적용 (모델 traits × HW → 인자/ENV) -------------------------
compat_resolve
if [ -n "${COMPAT_ABORT:-}" ]; then
  echo "❌ 호환 불가: $COMPAT_ABORT" >&2; exit 1
fi

# --- 기동 인자 조립 ---------------------------------------------------------
LAUNCH_ARGS=(
  --model-path "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --tp "$TP_SIZE"
  "${QUANT_ARGS[@]}"
  "${MODEL_ARGS[@]}"
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
  --mem-fraction-static "$MEM_FRAC"
  "${EXTRA_ARGS[@]}"
)

echo "============================================================"
echo " SGLang (재)기동"
echo "   env / sglang : $CONDA_ENV / $(python -c 'import sglang;print(sglang.__version__)' 2>/dev/null || echo '?')"
echo "   model        : $MODEL_KEY ($MODEL_ARCH/$MODEL_QUANT) → served '$SERVED_NAME'"
echo "   hardware     : $(hw_summary)"
echo "   endpoint     : ${HOST}:${PORT}  | TP=$TP_SIZE | ctx=$CONTEXT_LEN | mem=$MEM_FRAC"
echo "   mode         : $([ "$BACKGROUND" = 1 ] && echo background || echo foreground)"
[ -n "${SGLANG_ENABLE_JIT_DEEPGEMM:-}" ] && echo "   fp8 env      : DeepGEMM=$SGLANG_ENABLE_JIT_DEEPGEMM cutlass_block_fp8=${SGLANG_SUPPORT_CUTLASS_BLOCK_FP8:-}"
echo "   launch args  : ${MODEL_ARGS[*]:-} ${QUANT_ARGS[*]:-} ${EXTRA_ARGS[*]:-}"
echo "------------------------------------------------------------"

# --- 1) 기존 서버 종료 ------------------------------------------------------
echo "[1/2] 기존 서버 정리"
sglang_stop "$PORT"

# --- 2) 새 서버 기동 --------------------------------------------------------
mkdir -p "$LOG_DIR"
echo "[2/2] 기동"

if [ "$BACKGROUND" = "1" ]; then
  LOG_FILE="$LOG_DIR/sglang_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log"
  nohup python -X faulthandler -u -m sglang.launch_server "${LAUNCH_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 < /dev/null &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$LOG_DIR/sglang_${MODEL_KEY}.pid"
  echo "  PID=$SERVER_PID  로그: $LOG_FILE"
  echo "  health 대기中 (최대 ${SGLANG_WAIT:-900}s)..."
  if sglang_wait_ready "127.0.0.1" "$PORT" "${SGLANG_WAIT:-900}" "$SERVER_PID"; then
    echo "→ 기동 완료. 헬스체크: MODEL_KEY=$MODEL_KEY bash $SCRIPTS_DIR/healthcheck.sh"
  else
    echo "→ ❌ 기동 실패. 로그 확인: tail -50 $LOG_FILE" >&2
    tail -30 "$LOG_FILE" 2>/dev/null >&2
    exit 1
  fi
else
  # foreground: bash 를 python 으로 대체 (Ctrl-C 가 서버로 직접 전달, 출력 그대로 보임)
  echo "  foreground 실행 (중단: Ctrl-C)"
  echo "============================================================"
  exec python -X faulthandler -u -m sglang.launch_server "${LAUNCH_ARGS[@]}"
fi
