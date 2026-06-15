#!/usr/bin/env bash
# HuggingFace 모델 다운로드 → $MODELS_DIR/<name> (임시파일은 $MODELS_DIR/tmp)
#
# 사용법:
#   bash scripts/download_model.sh <model-id-또는-URL> [저장폴더명]
# 예:
#   bash scripts/download_model.sh QuantTrio/DeepSeek-V3.2-AWQ DeepSeek-V3.2-AWQ
#   bash scripts/download_model.sh https://huggingface.co/Qwen/Qwen3.6-27B
#   bash scripts/download_model.sh Qwen/Qwen3.6-27B my-qwen27   # 폴더명 지정
# 파라미터: MODELS_DIR / CONDA_ENV  (scripts/env.sh 참조)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# 인자 없으면 사용법(헤더 주석) 출력
[ $# -ge 1 ] || { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

activate_conda

# 1) 입력 정규화: URL이면 org/repo 만 추출 (/tree/..., 끝 슬래시 제거)
RAW="$1"
MODEL_ID="$(printf '%s' "$RAW" | sed -E 's#^https?://(www\.)?huggingface\.co/##; s#/tree/.*$##; s#/blob/.*$##; s#/+$##')"

# 2) 저장 폴더명: 2번째 인자 우선, 없으면 repo 이름(마지막 경로)
NAME="${2:-$(basename "$MODEL_ID")}"
DL_DIR="$MODELS_DIR/$NAME"

mkdir -p "$MODELS_DIR/tmp"
export TMPDIR="$MODELS_DIR/tmp"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "============================================================"
echo " HF 모델 다운로드"
echo "   입력      : $RAW"
echo "   model_id  : $MODEL_ID"
echo "   저장 위치 : $DL_DIR"
echo "   디스크 여유: $(df -h "$MODELS_DIR" | awk 'NR==2{print $4" / "$2}')"
echo "============================================================"

hf download "$MODEL_ID" --local-dir "$DL_DIR"

echo "=== 완료. 검증 ==="
du -sh "$DL_DIR"
echo "safetensors  : $(find "$DL_DIR" -name '*.safetensors' 2>/dev/null | wc -l) 개"
echo "incomplete   : $(find "$DL_DIR" -name '*.incomplete' 2>/dev/null | wc -l) 개 (0이어야 정상)"
ls "$DL_DIR"/config.json >/dev/null 2>&1 && echo "config.json  : OK" || echo "⚠️ config.json 없음 (GGUF 등 비-transformers 포맷일 수 있음)"
echo "→ 사용: SGLang 기동  bash $SCRIPTS_DIR/sglang/restart_server.sh"
