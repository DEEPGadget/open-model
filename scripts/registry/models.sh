#!/usr/bin/env bash
# ============================================================================
# 모델 레지스트리 — 모델의 "정체성 + 특성(traits)"만 선언
#   여기엔 HW별 플래그를 두지 않는다. 플래그는 registry/compat_rules.sh 가
#   (모델 traits × HW) 조합으로 산출한다. (관심사 분리)
#
# model_lookup <키>  → 아래 전역변수 설정 후 return 0 (없으면 return 1):
#   MODEL_NAME        : MODELS_DIR 하위 폴더명
#   SERVED_NAME       : API 에 노출될 모델 이름
#   MODEL_ARCH        : 어텐션 계열  mla | gqa | gdn
#   MODEL_QUANT       : 양자화 포맷  awq | nvfp4 | blockfp8 | fp8 | bf16
#   REASONING_PARSER  : sglang --reasoning-parser (없으면 "")
#   TOOL_PARSER       : sglang --tool-call-parser (없으면 "")
#
# 새 모델 추가 = 아래 case 에 한 블록 추가 (데이터 한 줄). 플래그 로직은 안 건드림.
# ============================================================================

model_lookup() {
  local key="$1"
  MODEL_NAME=""; SERVED_NAME=""; MODEL_ARCH=""; MODEL_QUANT=""
  REASONING_PARSER=""; TOOL_PARSER=""
  case "$key" in
    deepseek|deepseek-v32-awq)
      MODEL_NAME="DeepSeek-V3.2-AWQ";   SERVED_NAME="deepseek-v3.2"
      MODEL_ARCH="mla";  MODEL_QUANT="awq"
      REASONING_PARSER="deepseek-v3"; TOOL_PARSER="deepseekv32" ;;
    deepseek-nvfp4|deepseek-v32-nvfp4)
      MODEL_NAME="DeepSeek-V3.2-NVFP4"; SERVED_NAME="deepseek-v3.2"
      MODEL_ARCH="mla";  MODEL_QUANT="nvfp4"
      REASONING_PARSER="deepseek-v3"; TOOL_PARSER="deepseekv32" ;;
    qwen32|qwen3-32b-fp8)
      MODEL_NAME="Qwen3-32B-FP8";       SERVED_NAME="qwen3-32b"
      MODEL_ARCH="gqa";  MODEL_QUANT="blockfp8"
      REASONING_PARSER="qwen3" ;;
    qwen27|qwen3.6-27b)
      MODEL_NAME="Qwen3.6-27B";         SERVED_NAME="qwen3.6-27b"
      MODEL_ARCH="gdn";  MODEL_QUANT="bf16"
      REASONING_PARSER="qwen3" ;;
    *)
      return 1 ;;
  esac
  return 0
}

# 사용 가능한 키 목록 (에러 메시지/도움말용)
model_keys() { echo "deepseek | deepseek-nvfp4 | qwen32 | qwen27"; }
