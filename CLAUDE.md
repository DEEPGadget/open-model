# CLAUDE.md — open-model 작업 가이드

이 레포에서 Claude 세션이 알아야 할 핵심 맥락/결정/규칙. 어느 GPU 노드(H200 / RTX Pro 6000 / RTX 5090)에서
세션을 시작하든 동일하게 적용된다. 사용자 가이드는 [README.md](README.md), 상세 분석은 [doc/](doc/) 참고.

## 프로젝트
SGLang 으로 오픈소스 LLM 을 서빙하고 LiteLLM proxy 로 OpenAI/Anthropic API 를 노출하는 배포 스크립트 모음.
핵심 컨셉: (GPU SM아키텍처 × 모델 어텐션계열 × 양자화) 조합마다 필요한 커널/플래그가 다른데,
그 지식을 **규칙 엔진**(`scripts/registry/compat_rules.sh`)에 코드화 → 모델 키 하나로 알맞게 기동.

## conda env (2개로 분리 — 절대 합치지 말 것)
sglang 과 litellm 은 `openai` 패키지 버전 핀이 충돌하므로 **반드시 별도 env**.
- **`sglang`** = SGLang 서빙 (`CONDA_ENV` 기본값)
- **`litellm`** = litellm[proxy] (`LITELLM_ENV` 기본값)
- 설치: `bash scripts/setup.sh` (또는 `setup.sh sglang` / `setup.sh litellm`, 클린은 `RECREATE=1`)
- ⚠️ litellm 을 sglang env 에 깔면 sglang 의 `openai==2.6.1` 핀이 깨진다(겪었던 실수).

## 버전 (SGLang 0.5.13 단일)
- `sglang[all]==0.5.13` / `torch==2.11.0` (**cu130** 휠) / `flashinfer[cu13]==0.6.12`
- 드라이버 CUDA 13.x 기준. 핀은 `scripts/env.sh` 에 집중. 0.5.13 은 DeepSeek MLA·DSA(H200) 와
  Qwen GDN/hybrid(SM120) 를 모두 지원 (후자는 `--linear-attn-decode-backend` 가 0.5.10+ 에만 있음).
- 과거 stable(0.5.9)/next(0.5.13) 채널 분리가 있었으나 **0.5.13 단일로 통일됨** (잔재 두지 말 것).
- 버전 올릴 땐 4종(sglang/torch/flashinfer/cuda)을 그 버전 pyproject 기준으로 함께 맞춘다.

## 모델 키 (registry/models.sh)
| 키 | 모델 | 어텐션/양자화 | 비고 |
|----|------|---------------|------|
| `deepseek` | DeepSeek-V3.2-AWQ | MLA·DSA / AWQ | **H200/SM90·B200/SM100 전용** (MLA 가 SM120 미지원) |
| `deepseek-nvfp4` | DeepSeek-V3.2-NVFP4 | MLA·DSA / NVFP4 | B200/SM100 |
| `qwen32` | Qwen3-32B-FP8 | GQA / block FP8 | SM120 OK |
| `qwen27` | Qwen3.6-27B | hybrid GDN / bf16 | SM120 OK |

모델 추가 = `scripts/registry/models.sh` 에 traits 한 블록(폴더명/서빙명/arch/quant/parser). 플래그 로직은 안 건드림.

## 하드웨어 호환성 규칙 (compat_rules.sh — 실측으로 얻은 지식)
- SM120 전반 → `--disable-cuda-graph`
- SM120 × block FP8(qwen32) → `SGLANG_ENABLE_JIT_DEEPGEMM=0` + `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8=1`
  (DeepGEMM/flashinfer-trtllm FP8 커널이 SM120 미지원 → cutlass 우회해야 정상)
- hybrid GDN(qwen27) → `--attention-backend triton --linear-attn-decode-backend triton`
  (안 붙이면 gibberish 출력. 0.5.9 에선 이 플래그가 없어 불가였음)
- MLA·DSA(deepseek) × SM120 → 기동 거부(SM90/SM100 에서 쓰라 안내)
- NVFP4 × 비-Blackwell → 거부
- 전부 `restart_server.sh` 가 GPU 감지(`lib/detect_hw.sh`) 후 자동 적용. `FORCE_SM=` 로 강제 가능.

## 자주 쓰는 명령
```bash
bash scripts/setup.sh                                  # env 설치(sglang+litellm)
bash scripts/download_model.sh <HF-repo-id> <폴더명>     # 가중치 (오래걸림)
bash scripts/sglang/restart_server.sh <키> --bg         # 서버 기동(백그라운드, health 대기)
bash scripts/litellm/start_litellm.sh                  # proxy 기동
MODEL_KEY=<키> bash scripts/healthcheck.sh             # 검증
bash scripts/status.sh / stop_all.sh
```
- 포트: SGLang `:30000`, LiteLLM `:4000`. proxy 모델명: `<served>-oai|-anth`, Qwen 은 `-*-fast`(thinking off)도 있음.
- `restart_server.sh` 는 foreground 기본(exec, Ctrl-C 중단), `--bg` 로 백그라운드. 기존 서버 자동 종료 후 기동.

## 작업 규칙 / 주의
- **오래 걸리는 작업(모델 다운로드, 대형 설치, 긴 벤치)은 직접 실행하지 말고 사용자에게 커맨드를 넘긴다.**
  짧은 검증(문법체크, status, curl, 이미 받은 파일 확인)은 직접 해도 됨.
- 서버 기동/HTTP 호출은 curl(가벼움)로 검증 가능. (에이전트 샌드박스에선 sglang 서버 기동이 exit 144 로 막힐 수 있으니,
  그 경우 사용자 터미널 기동 + curl 검증으로 분업)
- `CONTEXT_LEN` 등 env 누수 주의: `restart_server.sh` 가 인자/`unset` 으로 격리. 모델 선택은 위치인자가 최우선.
- VRAM/컨텍스트 조절 옵션(특히 RTX 5090 32GB)은 [doc/sequence-and-vram.md](doc/sequence-and-vram.md) 참고.

## 현재 상태 / 남은 일
- SM120(Pro 6000 ×8)에서 qwen32(block FP8)·qwen27(GDN) 정상, LiteLLM proxy(OpenAI/Anthropic/스트리밍/인증) 검증 완료.
- **미검증**: H200 에서 DeepSeek-V3.2(MLA·DSA) — 실제 H200 에서 `restart_server.sh deepseek` 로 확인 필요(SM90 지원이라 정상 예상).
- 개선 후보: 저VRAM 노브(`--kv-cache-dtype fp8`, `--max-total-tokens`, `--max-running-requests` 등) env 노출
  + VRAM 자동감지 보수 기본값 (doc 6장 참고).
