#!/usr/bin/env bash
# ============================================================================
# SGLang 서버 기동 스크립트 (오케스트레이션만 담당)
#
#   모델 정체성   → registry/models.sh   (model_lookup)
#   HW 감지       → lib/detect_hw.sh     (hw_sm_family 등)
#   런치 인자 산출 → registry/compat_rules.sh (compat_resolve)
#   이 스크립트는 위를 엮어 "기존 종료 → 기동" 만 수행.
#
# 사용법:
#   bash scripts/sglang/restart_server.sh <모델키> [--ctx N]
#     <모델키>: deepseek | deepseek-nvfp4 | qwen32 | qwen27   (생략 시 env 기본=deepseek)
#   예)
#     bash scripts/sglang/restart_server.sh qwen27
#     TP_SIZE=4 bash scripts/sglang/restart_server.sh deepseek    # H200 4장
#     bash scripts/sglang/restart_server.sh qwen32 --ctx 32768
# 기타 파라미터(PORT/TP_SIZE/MEM_FRAC/DP_ATTENTION/DISABLE_CUDA_GRAPH 등): env.sh / compat_rules.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 인자 파싱: 모델키(위치인자) + 선택 플래그 -------------------------------
#   모델키/컨텍스트는 인자로 받아 최우선 적용 → 셸에 남은 export 값 오염 차단
MODEL_KEY_ARG=""
CTX_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help|list)
      echo "사용: restart_server.sh <deepseek|deepseek-nvfp4|qwen32|qwen27> [--ctx N]"; exit 0 ;;
    --ctx)   CTX_OVERRIDE="${2:-}"; shift 2 ;;
    --ctx=*) CTX_OVERRIDE="${1#*=}"; shift ;;
    *)       MODEL_KEY_ARG="$1"; shift ;;
  esac
done
[ -n "$MODEL_KEY_ARG" ] && export MODEL_KEY="$MODEL_KEY_ARG"

# 셸에 남은 stale 값 제거 → env.sh 가 기본값(키 기반/auto) 새로 계산
unset CONTEXT_LEN

source "$SCRIPT_DIR/../env.sh"

# 의도적 컨텍스트 override 는 --ctx 로만 (env 누수와 구분)
[ -n "$CTX_OVERRIDE" ] && CONTEXT_LEN="$CTX_OVERRIDE"

# --- 사전 점검 (1): conda + sglang ------------------------------------------
echo "============================================================"
echo " SGLang 기동 준비"
echo "   conda env  : $CONDA_ENV"
echo "   model key  : $MODEL_KEY ($MODEL_ARCH/$MODEL_QUANT)"
echo "   model_dir  : $MODEL_DIR"
echo "   hardware   : $(hw_summary)"
echo "   TP_SIZE    : $TP_SIZE"
echo "   served as  : $SERVED_NAME  @ ${HOST}:${PORT}"
echo "   ctx / mem  : $CONTEXT_LEN  / frac=$MEM_FRAC"
echo "============================================================"

activate_conda

if ! python -c "import sglang" 2>/dev/null; then
  echo "❌ '$CONDA_ENV' env에 sglang 미설치." >&2
  echo "   먼저 설치하세요:  bash $SCRIPTS_DIR/sglang/setup_sglang.sh" >&2
  exit 1
fi
echo "   sglang     : $(python -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo '?')"

# --- 사전 점검 (2): 모델 가중치 ---------------------------------------------
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "❌ 모델이 없습니다: $MODEL_DIR/config.json" >&2
  echo "   먼저 다운로드하세요:  bash $SCRIPTS_DIR/download_model.sh <hf-repo-id> $MODEL_NAME" >&2
  exit 1
fi

# --- 호환성 규칙 적용 (모델 traits × HW → 인자/ENV) -------------------------
#   conda 활성화 이후 호출 (sglang_has_flag 가 설치본을 조회)
compat_resolve
if [ -n "${COMPAT_ABORT:-}" ]; then
  echo "❌ 호환 불가: $COMPAT_ABORT" >&2
  exit 1
fi

# --- 기존 서버 종료 (재시작) ------------------------------------------------
echo "--- 기존 sglang 프로세스 정리 ---"
pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null && sleep 3 || echo "(기존 프로세스 없음)"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sglang_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log"

# --- 기동 -------------------------------------------------------------------
echo "--- SGLang 기동 (로그: $LOG_FILE) ---"
echo "    QUANT_ARGS=${QUANT_ARGS[*]:-(none)}"
echo "    MODEL_ARGS=${MODEL_ARGS[*]:-(none)}"
echo "    EXTRA_ARGS=${EXTRA_ARGS[*]:-(none)}"
[ -n "${SGLANG_ENABLE_JIT_DEEPGEMM:-}" ] && echo "    ENV DeepGEMM=$SGLANG_ENABLE_JIT_DEEPGEMM cutlass_block_fp8=${SGLANG_SUPPORT_CUTLASS_BLOCK_FP8:-}"
set -x
nohup python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --tp "$TP_SIZE" \
  "${QUANT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --trust-remote-code \
  --host "$HOST" \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRAC" \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &
set +x

SERVER_PID=$!
echo "$SERVER_PID" > "$LOG_DIR/sglang_${MODEL_KEY}.pid"
echo "→ PID $SERVER_PID 로 백그라운드 기동. 가중치 로딩에 수 분 소요."
echo "→ 진행 확인:  tail -f $LOG_FILE"
echo "→ 헬스체크 :  MODEL_KEY=$MODEL_KEY bash $SCRIPTS_DIR/healthcheck.sh"
