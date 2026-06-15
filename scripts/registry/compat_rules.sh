#!/usr/bin/env bash
# ============================================================================
# 호환성 규칙 엔진 — (모델 traits × HW SM패밀리) → sglang 런치 인자/ENV 산출
#
#   지금까지 실측으로 얻은 "지식"을 여기 한 곳에 박는다. 새 GPU/이슈는 여기만 수정.
#   compat_resolve 호출 시 아래 전역을 채움:
#     QUANT_ARGS[]  : 양자화 인자
#     MODEL_ARGS[]  : 파서/어텐션 백엔드 등 모델별 인자
#     EXTRA_ARGS[]  : cuda-graph/context-length/dp-attention 등
#     (필요시 SGLANG_* ENV export)
#     COMPAT_ABORT  : 비어있지 않으면 "이 조합은 불가" → 호출측이 중단
#
#   입력(전역): MODEL_ARCH MODEL_QUANT REASONING_PARSER TOOL_PARSER
#               CONTEXT_LEN  (env.sh)  + 선택 override:
#               DP_ATTENTION / DISABLE_CUDA_GRAPH / ATTN_BACKEND /
#               LINEAR_ATTN_BACKEND / MAMBA_BACKEND
#   HW: hw_sm_family() (lib/detect_hw.sh)
# ============================================================================

# 설치된 sglang 의 server_args 에 특정 플래그가 있는지 (import 없이 빠르게 확인)
sglang_has_flag() {
  local flag="$1" dir
  dir="$(python -c "import importlib.util as u; s=u.find_spec('sglang'); print(s.submodule_search_locations[0])" 2>/dev/null)" || return 1
  grep -rqs -- "$flag" "$dir/srt/server_args.py" 2>/dev/null
}

compat_resolve() {
  local sm; sm="$(hw_sm_family)"
  QUANT_ARGS=(); MODEL_ARGS=(); EXTRA_ARGS=(); COMPAT_ABORT=""

  # --- 파서 (레지스트리에서) ---
  [ -n "${REASONING_PARSER:-}" ] && MODEL_ARGS+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "${TOOL_PARSER:-}" ]      && MODEL_ARGS+=(--tool-call-parser "$TOOL_PARSER")

  # --- 양자화 규칙 ---
  case "$MODEL_QUANT" in
    awq)
      QUANT_ARGS=(--quantization awq_marlin) ;;            # Hopper INT4 Marlin
    nvfp4)
      QUANT_ARGS=(--quantization modelopt_fp4)
      case "$sm" in sm100|sm120) : ;; *) COMPAT_ABORT="NVFP4 는 Blackwell(SM100/SM120) 전용 — 현재 $sm";; esac ;;
    blockfp8)
      # 가중치 자동 인식(--quantization 불필요). SM120 은 DeepGEMM/flashinfer-trtllm
      # FP8 커널 미지원 → DeepGEMM 끄고 cutlass block FP8 사용해야 정상(실측).
      if [ "$sm" = "sm120" ]; then
        export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
        export SGLANG_SUPPORT_CUTLASS_BLOCK_FP8="${SGLANG_SUPPORT_CUTLASS_BLOCK_FP8:-1}"
      fi ;;
    fp8|bf16|"") : ;;                                      # 특수 처리 없음
    *) : ;;
  esac

  # --- 어텐션 계열 규칙 ---
  case "$MODEL_ARCH" in
    mla)
      # DeepSeek MLA/DSA 커널은 SM90/SM100 전용
      case "$sm" in
        sm90|sm100) : ;;
        *) COMPAT_ABORT="MLA/DSA(DeepSeek) 커널은 SM90/SM100 전용 — $sm 미지원 (H200/B200 에서 사용)";;
      esac
      # MLA 효율: dp-attention 기본 on (DP_ATTENTION=0 으로 끔)
      [ "${DP_ATTENTION:-1}" = "1" ] && EXTRA_ARGS+=(--enable-dp-attention) ;;
    gdn)
      # hybrid GDN/Mamba: SM120 은 triton 어텐션 필요. linear-attn-decode 플래그가
      # 있으면(신버전) 그걸 쓰고, 없으면(구버전) mamba-backend 로 폴백.
      MODEL_ARGS+=(--attention-backend "${ATTN_BACKEND:-triton}")
      if sglang_has_flag "linear-attn-decode-backend"; then
        MODEL_ARGS+=(--linear-attn-decode-backend "${LINEAR_ATTN_BACKEND:-triton}")
      else
        MODEL_ARGS+=(--mamba-backend "${MAMBA_BACKEND:-triton}")
      fi ;;
    gqa|"") : ;;
    *) : ;;
  esac

  # --- 범용: SM120 cuda-graph 이슈 회피 ---
  case "$sm" in
    sm120) [ "${DISABLE_CUDA_GRAPH:-1}" = "1" ] && EXTRA_ARGS+=(--disable-cuda-graph) ;;
    *)     [ "${DISABLE_CUDA_GRAPH:-0}" = "1" ] && EXTRA_ARGS+=(--disable-cuda-graph) ;;
  esac

  # --- context-length: auto 면 sglang 자동결정, 숫자면 강제 ---
  [ "${CONTEXT_LEN:-auto}" != "auto" ] && EXTRA_ARGS+=(--context-length "$CONTEXT_LEN")
}
