#!/usr/bin/env bash
# ============================================================================
# 하드웨어 감지 — GPU SM아키텍처 / 개수 / VRAM
#   - 모든 함수는 "호출 시점"에 nvidia-smi 조회 (source 시점엔 아무 동작 없음)
#   - FORCE_SM 로 SM 패밀리 강제 가능 (테스트/감지실패 대비)
# ============================================================================

# 첫 GPU compute capability (예: "12.0")
hw_compute_cap() { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '; }

# SM 패밀리 분류:
#   sm90  = Hopper (H100/H200)
#   sm100 = Datacenter Blackwell (B200/GB200)
#   sm120 = Workstation Blackwell (RTX Pro 6000 / RTX 5090)
#   unknown / other
hw_sm_family() {
  if [ -n "${FORCE_SM:-}" ]; then echo "$FORCE_SM"; return 0; fi
  case "$(hw_compute_cap)" in
    9.*)  echo "sm90" ;;
    10.*) echo "sm100" ;;
    12.*) echo "sm120" ;;
    "")   echo "unknown" ;;
    *)    echo "other" ;;
  esac
}

# 가시 GPU 개수
hw_gpu_count() {
  local n; n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  echo "${n//[!0-9]/}"
}

# 첫 GPU VRAM (MiB, 숫자만)
hw_vram_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

# 첫 GPU 이름
hw_gpu_name() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1; }

# 한 줄 요약 (로그/디버그용)
hw_summary() {
  echo "GPU=$(hw_gpu_name) | SM=$(hw_sm_family)($(hw_compute_cap)) | count=$(hw_gpu_count) | vram=$(hw_vram_mib)MiB"
}
