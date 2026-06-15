#!/usr/bin/env bash
# ============================================================================
# SGLang 서버 기동 스크립트 (모델은 첫 인자로 선택)
#
# 타깃    : H200 NVL 4장 (TP4)  ← 실제 운영
# 디버깅  : Pro 6000 BW SE 8장 (TP8) ← 현재 세션
#   → TP 수는 GPU 개수에서 자동 추정(env.sh)하거나 TP_SIZE 로 강제 지정
#
# 사용법:
#   bash scripts/sglang/restart_server.sh <모델키>
#     <모델키>: deepseek | deepseek-nvfp4 | qwen32 | qwen27   (생략 시 env 기본=deepseek)
#   예)
#     bash scripts/sglang/restart_server.sh qwen27           # Qwen3.6-27B
#     bash scripts/sglang/restart_server.sh qwen32           # Qwen3-32B-FP8
#     TP_SIZE=4 bash scripts/sglang/restart_server.sh deepseek   # H200 4장
# 기타 파라미터(PORT/TP_SIZE/MEM_FRAC 등): scripts/env.sh 참조
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 인자 파싱: 모델키(위치인자) + 선택 플래그
#   사용: restart_server.sh <모델키> [--ctx N]
#   - 모델키/컨텍스트는 인자로 받아 최우선 적용 → 셸에 남은 export 값으로 인한 오염 차단
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

# 셸에 남아있을 수 있는 값들을 비워 env.sh 가 기본값(키 기반/auto)을 새로 계산하게 함.
#   (이전에 source env.sh 한 인터랙티브 셸의 stale export 가 강제로 이기는 문제 차단)
unset CONTEXT_LEN

source "$SCRIPT_DIR/../env.sh"

# 의도적 컨텍스트 override 는 --ctx 로만 (env 누수와 구분)
[ -n "$CTX_OVERRIDE" ] && CONTEXT_LEN="$CTX_OVERRIDE"

# ---------------------------------------------------------------------------
# 모델별 런치 인자 (정체성=MODEL_NAME/SERVED_NAME 은 env.sh 레지스트리가 결정)
#   QUANT_ARGS : 양자화 인자 / MODEL_ARGS : 파서 등 모델별 인자
# ---------------------------------------------------------------------------
QUANT_ARGS=()
MODEL_ARGS=()
case "$MODEL_KEY" in
  deepseek|deepseek-v32-awq)
    QUANT_ARGS=(--quantization awq_marlin)   # Hopper(H200)에서 INT4 가속 커널
    MODEL_ARGS=(--tool-call-parser deepseekv32 --reasoning-parser deepseek-v3)
    ;;
  deepseek-nvfp4|deepseek-v32-nvfp4)
    # ⚠️ NVFP4는 Blackwell(B200/SM100) 전용.
    QUANT_ARGS=(--quantization modelopt_fp4)
    MODEL_ARGS=(--tool-call-parser deepseekv32 --reasoning-parser deepseek-v3)
    ;;
  qwen32|qwen3-32b-fp8)
    # Qwen3-32B 블록FP8 (GQA). FP8 가중치 자동 인식.
    # ⚠️ SM120 에선 DeepGEMM/flashinfer-trtllm FP8 커널 미지원 → 아래 SM120 블록에서
    #    DeepGEMM off + cutlass block FP8 자동 설정 (그래야 정상 동작).
    MODEL_ARGS=(--reasoning-parser qwen3)
    _IS_BLOCK_FP8=1
    ;;
  qwen27|qwen3.6-27b)
    # Qwen3.6-27B (Qwen3_5ForConditionalGeneration, bf16 → 양자화 인자 없음)
    # hybrid GDN(Gated DeltaNet/Mamba) 모델 → Blackwell(SM120)에선 attention 은 triton/trtllm_mha.
    #   ATTN_BACKEND   : attention 백엔드 override (기본 triton)
    #   MAMBA_BACKEND  : GDN/mamba 백엔드 (triton|flashinfer). ⚠️ SM120 + sglang 0.5.9 에선
    #                    triton 이 gibberish 출력 가능(알려진 버그) → flashinfer 로 시도해볼 것.
    MODEL_ARGS=(--reasoning-parser qwen3 \
                --attention-backend "${ATTN_BACKEND:-triton}" \
                --mamba-backend "${MAMBA_BACKEND:-triton}")
    [ -n "${MAMBA_SCHED:-}" ] && MODEL_ARGS+=(--mamba-scheduler-strategy "$MAMBA_SCHED")
    ;;
  *)
    echo "❌ 알 수 없는 MODEL_KEY: $MODEL_KEY" >&2
    echo "   사용 가능: deepseek | deepseek-nvfp4 | qwen32 | qwen27" >&2
    exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 사전 점검
# ---------------------------------------------------------------------------
echo "============================================================"
echo " SGLang 기동 준비"
echo "   conda env  : $CONDA_ENV"
echo "   model key  : $MODEL_KEY"
echo "   model_dir  : $MODEL_DIR"
echo "   TP_SIZE    : $TP_SIZE"
echo "   served as  : $SERVED_NAME  @ ${HOST}:${PORT}"
echo "   ctx / mem  : $CONTEXT_LEN  / frac=$MEM_FRAC"
echo "============================================================"

activate_conda

# SGLang 설치 확인
if ! python -c "import sglang" 2>/dev/null; then
  echo "❌ '$CONDA_ENV' env에 sglang 미설치." >&2
  echo "   먼저 설치하세요:  bash $SCRIPTS_DIR/sglang/setup_sglang.sh" >&2
  exit 1
fi
echo "   sglang     : $(python -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo '?')"

# 모델 경로 확인
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "❌ 모델이 없습니다: $MODEL_DIR/config.json" >&2
  echo "   먼저 다운로드하세요:  bash $SCRIPTS_DIR/download_model.sh <hf-repo-id> $MODEL_NAME" >&2
  exit 1
fi

# NVFP4 + 비-Blackwell 조합 가드
if [ "$MODEL_KEY" = "deepseek-nvfp4" ] || [ "$MODEL_KEY" = "deepseek-v32-nvfp4" ]; then
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "blackwell"; then
    echo "⚠️  NVFP4 프로파일인데 Blackwell GPU가 아닙니다. H200(Hopper)에서는 awq 프로파일을 쓰세요." >&2
    echo "    계속하려면 5초 내 Ctrl-C로 취소하지 않으면 진행합니다..." >&2
    sleep 5
  fi
fi

# ---------------------------------------------------------------------------
# 기존 서버 종료 (재시작)
# ---------------------------------------------------------------------------
echo "--- 기존 sglang 프로세스 정리 ---"
pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null && sleep 3 || echo "(기존 프로세스 없음)"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sglang_${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# 기동
#   - --tp $TP_SIZE          : 텐서 병렬 (H200=4, Pro6000=8)
#   - QUANT_ARGS / MODEL_ARGS : 프로파일별 양자화·파서 인자 (위 case 참조)
#   - --enable-dp-attention  : DeepSeek MLA 전용 최적화. deepseek 프로파일에서만 기본 on
#                              (DP_ATTENTION=0 으로 끔)
#   - --disable-cuda-graph   : SM120(Pro 6000) 호환 이슈 회피. DISABLE_CUDA_GRAPH=1 로 on
#   - --trust-remote-code    : 커스텀 아키텍처
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
# context-length: auto 면 sglang 자동결정(모델별 최대), 숫자면 강제.
[ "${CONTEXT_LEN}" != "auto" ] && EXTRA_ARGS+=(--context-length "$CONTEXT_LEN")
# dp-attention 은 deepseek(MLA) 프로파일에서만 의미. 기본값을 프로파일로 결정.
case "$MODEL_KEY" in deepseek*) _DP_DEFAULT=1 ;; *) _DP_DEFAULT=0 ;; esac
[ "${DP_ATTENTION:-$_DP_DEFAULT}" = "1" ] && EXTRA_ARGS+=(--enable-dp-attention)
# SM120 은 cuda-graph 이슈가 보고됨 → 자동 감지해 기본 on, override 가능
_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
case "$_CC" in 12.*) _CG_DEFAULT=1 ;; *) _CG_DEFAULT=0 ;; esac
[ "${DISABLE_CUDA_GRAPH:-$_CG_DEFAULT}" = "1" ] && EXTRA_ARGS+=(--disable-cuda-graph)

# SM120 + 블록FP8 모델: DeepGEMM('Unknown recipe') / flashinfer-trtllm('capability 120 미지원')
# 둘 다 SM120 미지원 → DeepGEMM 끄고 cutlass block FP8 사용해야 정상 동작(실측 검증됨).
if [ "${_IS_BLOCK_FP8:-0}" = "1" ] && case "$_CC" in 12.*) true ;; *) false ;; esac; then
  export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
  export SGLANG_SUPPORT_CUTLASS_BLOCK_FP8="${SGLANG_SUPPORT_CUTLASS_BLOCK_FP8:-1}"
  echo "    [SM120 FP8] DeepGEMM=$SGLANG_ENABLE_JIT_DEEPGEMM cutlass_block_fp8=$SGLANG_SUPPORT_CUTLASS_BLOCK_FP8"
fi

echo "--- SGLang 기동 (로그: $LOG_FILE) ---"
echo "    compute_cap=$_CC  EXTRA_ARGS=${EXTRA_ARGS[*]}  MODEL_ARGS=${MODEL_ARGS[*]}"
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
echo "→ 헬스체크 :  bash $SCRIPTS_DIR/healthcheck.sh"
