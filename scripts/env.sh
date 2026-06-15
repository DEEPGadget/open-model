#!/usr/bin/env bash
# ============================================================================
# 공통 환경변수 (모든 스크립트가 source 해서 사용)
#
# - 이 파일은 절대경로를 박지 않고, 자기 위치(scripts/)를 기준으로 PROJECT_ROOT 계산
# - 나중에 바꿀 만한 값은 전부 여기 모음. 각 값은 이미 export 된 값이 있으면 그대로 둠
#   (= 호출 시  ENV_NAME=other bash xxx.sh  로 일회성 override 가능)
#
# 사용법(각 스크립트 상단):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/.../env.sh"     # env.sh 까지의 상대경로
# ============================================================================

# --- 경로 (이 env.sh 가 scripts/ 안에 있다는 전제로 루트 계산) -----------------
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = <root>/scripts
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$_ENV_SH_DIR/.." && pwd)}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$_ENV_SH_DIR}"
MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
LITELLM_CONFIG="${LITELLM_CONFIG:-$PROJECT_ROOT/litellm/litellm_config.yaml}"

# --- conda ------------------------------------------------------------------
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-serving}"

# --- 모듈 로드: HW 감지 / 모델 레지스트리 / 호환성 규칙 (함수 정의만) ----------
# shellcheck source=lib/detect_hw.sh
source "$_ENV_SH_DIR/lib/detect_hw.sh"
# shellcheck source=registry/models.sh
source "$_ENV_SH_DIR/registry/models.sh"
# shellcheck source=registry/compat_rules.sh
source "$_ENV_SH_DIR/registry/compat_rules.sh"

# --- 모델 선택 (레지스트리 조회) --------------------------------------------
# MODEL_KEY 로 선택. restart_server.sh 는 첫 인자로도 받음 (인자가 최우선).
#   model_lookup 이 MODEL_NAME/SERVED_NAME/MODEL_ARCH/MODEL_QUANT/파서를 "직접" 설정
#   → 셸에 남은 MODEL_NAME 등으로 오염되지 않음. 모델 추가는 registry/models.sh 에서.
# PROFILE 은 하위호환 별칭.
MODEL_KEY="${MODEL_KEY:-${PROFILE:-deepseek}}"
if ! model_lookup "$MODEL_KEY"; then
  echo "❌ env.sh: 알 수 없는 MODEL_KEY '$MODEL_KEY' (사용 가능: $(model_keys))" >&2
  return 1 2>/dev/null || exit 1
fi
PROFILE="$MODEL_KEY"                              # 하위호환 별칭
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"              # 폴더 경로 (키로부터 산출)

# --- SGLang / 의존성 버전 (채널 선택) ---------------------------------------
# SGLANG_CHANNEL 로 버전 세트 선택. 각 핀은 해당 sglang 버전 pyproject 기준.
#   stable = 0.5.9  : DeepSeek-V3.2(H200) 검증 완료. cu129. flashinfer 수동 설치(--no-deps).
#   next   = 0.5.13 : SM120 GDN/FP8 수정 + linear-attn-decode 플래그. cu130(드라이버 CUDA13).
#                     flashinfer[cu13] 정확핀 → resolver 에 맡김(수동설치 off).
# 개별 변수(SGLANG_VER 등)를 직접 주면 그게 우선.
SGLANG_CHANNEL="${SGLANG_CHANNEL:-stable}"
case "$SGLANG_CHANNEL" in
  stable)
    SGLANG_VER="${SGLANG_VER:-0.5.9}";  TORCH_VER="${TORCH_VER:-2.9.1}"
    FLASHINFER_VER="${FLASHINFER_VER:-0.6.3}";  SGL_KERNEL_VER="${SGL_KERNEL_VER:-0.3.21}"
    TORCH_CUDA="${TORCH_CUDA:-cu129}";  SGLANG_MANUAL_FLASHINFER="${SGLANG_MANUAL_FLASHINFER:-1}" ;;
  next)
    SGLANG_VER="${SGLANG_VER:-0.5.13}"; TORCH_VER="${TORCH_VER:-2.11.0}"
    FLASHINFER_VER="${FLASHINFER_VER:-0.6.12}"; SGL_KERNEL_VER="${SGL_KERNEL_VER:-}"
    TORCH_CUDA="${TORCH_CUDA:-cu130}";  SGLANG_MANUAL_FLASHINFER="${SGLANG_MANUAL_FLASHINFER:-0}" ;;
  *)
    echo "❌ env.sh: 알 수 없는 SGLANG_CHANNEL '$SGLANG_CHANNEL' (stable|next)" >&2
    return 1 2>/dev/null || exit 1 ;;
esac

# --- SGLang 서버 ------------------------------------------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.90}"
# CONTEXT_LEN=auto → sglang 이 모델 config 에서 자동 결정(모델별 최대값 사용).
# 숫자로 주면 그 값으로 강제(모델 최대보다 크면 sglang 이 거부).
CONTEXT_LEN="${CONTEXT_LEN:-auto}"
# TP_SIZE 미지정 시 가시 GPU 개수로 자동 설정 (H200 4장→4 / Pro6000 8장→8)
if [ -z "${TP_SIZE:-}" ]; then
  TP_SIZE="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
fi

# --- LiteLLM proxy ----------------------------------------------------------
LL_HOST="${LL_HOST:-0.0.0.0}"
LL_PORT="${LL_PORT:-4000}"
# 프록시 마스터 키: config.yaml 은 os.environ/LITELLM_MASTER_KEY 로 읽음.
# 키 값은 여기(커밋 안 됨)에서 주입. 운영 시 실제 키로 export 해서 override.
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-test}"

# --- conda 활성화 헬퍼 (각 스크립트에서 호출) --------------------------------
activate_conda() {
  # shellcheck disable=SC1091
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

export PROJECT_ROOT SCRIPTS_DIR MODELS_DIR LOG_DIR LITELLM_CONFIG
export CONDA_HOME CONDA_ENV MODEL_KEY PROFILE MODEL_NAME MODEL_DIR SERVED_NAME
export MODEL_ARCH MODEL_QUANT REASONING_PARSER TOOL_PARSER
export SGLANG_CHANNEL SGLANG_VER TORCH_VER FLASHINFER_VER SGL_KERNEL_VER TORCH_CUDA SGLANG_MANUAL_FLASHINFER
export HOST PORT MEM_FRAC CONTEXT_LEN TP_SIZE LL_HOST LL_PORT LITELLM_MASTER_KEY
