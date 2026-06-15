# open-model

여러 GPU 환경(H200 / RTX Pro 6000 / RTX 5090)에서 **SGLang** 으로 오픈소스 LLM 을 서빙하고,
**LiteLLM** proxy 로 OpenAI / Anthropic 호환 API 를 한 엔드포인트에서 노출하는 배포 스크립트 모음.

하드웨어(SM 아키텍처)·모델(어텐션 계열)·양자화 조합마다 필요한 커널/플래그가 다른데,
그 "지식" 을 규칙 엔진으로 코드화해서 **모델 키 하나만 주면 알아서 맞는 인자로 기동**한다.

---

## 🚀 Quick Start

전제: `conda` (miniconda) 설치됨, NVIDIA 드라이버 + GPU 인식됨.

```bash
cd ~/open-model

# 1) 환경 설치 (sglang 서빙 env + litellm proxy env, 각각 분리 생성)
bash scripts/setup.sh

# 2) 모델 가중치 다운로드 (예: Qwen3.6-27B)  — 시간 소요
bash scripts/download_model.sh Qwen/Qwen3.6-27B Qwen3.6-27B

# 3) 모델 서버 기동 (백그라운드, health 까지 대기)
bash scripts/sglang/restart_server.sh qwen27 --bg

# 4) LiteLLM proxy 기동 (OpenAI/Anthropic API 노출)
export LITELLM_MASTER_KEY=sk-local-test          # 접근 키 (원하는 값)
bash scripts/litellm/start_litellm.sh

# 5) 호출 테스트 (OpenAI 호환)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-oai","messages":[{"role":"user","content":"안녕!"}]}'
```

종료: `bash scripts/stop_all.sh` · 상태: `bash scripts/status.sh`

> 다른 모델은 3) 의 `qwen27` 자리에 다른 **모델 키**(아래 표)를 넣으면 된다.
> SM 아키텍처/양자화에 맞는 플래그(예: `--disable-cuda-graph`, FP8 커널 우회)는 자동 적용된다.

---

## 모델 키

| 키 | 모델 | 어텐션/양자화 | 비고 |
|----|------|---------------|------|
| `deepseek` | DeepSeek-V3.2-AWQ | MLA·DSA / AWQ INT4 | **H200/SM90·B200/SM100 전용** (MLA 커널이 SM120 미지원) |
| `deepseek-nvfp4` | DeepSeek-V3.2-NVFP4 | MLA·DSA / NVFP4 | B200/SM100 |
| `qwen32` | Qwen3-32B-FP8 | GQA / block FP8 | SM120 OK (FP8 커널 자동 우회) |
| `qwen27` | Qwen3.6-27B | hybrid GDN / bf16 | SM120 OK (GDN triton 백엔드 자동) |

모델 추가는 [scripts/registry/models.sh](scripts/registry/models.sh) 에 한 블록 추가하면 된다(데이터만, 플래그 로직은 안 건드림).

---

## LiteLLM 노출 모델명

한 proxy(`:4000`) 에서 아래 이름으로 선택. 백엔드는 현재 떠 있는 SGLang 서버(`:30000`) 하나라,
**해당 모델을 `restart_server.sh` 로 띄워둔 경우에만** 그 이름이 동작한다.

- `*-oai` : OpenAI 호환 (`POST /v1/chat/completions`)
- `*-anth` : Anthropic 호환 (`POST /v1/messages`)
- `*-fast` : **thinking 비활성** (Qwen 계열, reasoning 단계 생략 → 빠른 응답)

```
deepseek-v3.2-oai / -anth
qwen3-32b-oai / -anth / -oai-fast / -anth-fast
qwen3.6-27b-oai / -anth / -oai-fast / -anth-fast
```

예) 빠른 응답:
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-oai-fast","messages":[{"role":"user","content":"15+27?"}]}'
```

---

## 구조

```
open-model/
├── scripts/
│   ├── setup.sh                 # sglang + litellm env 일괄/개별 셋업
│   ├── env.sh                   # 공통 설정(경로/conda/포트/버전핀) + 모듈 로드
│   ├── download_model.sh        # HF 가중치 다운로드 → models/<name>
│   ├── start_all.sh / stop_all.sh / status.sh / healthcheck.sh
│   ├── lib/
│   │   ├── detect_hw.sh         # GPU SM패밀리(sm90/sm100/sm120)·VRAM·개수 감지
│   │   └── server_ctl.sh        # sglang 종료(SIGTERM→KILL)·health 대기 헬퍼
│   ├── registry/
│   │   ├── models.sh            # 모델 정체성/특성 선언 (model_lookup)
│   │   └── compat_rules.sh      # (모델 × HW) → 런치 인자/ENV 산출 (compat_resolve)
│   ├── sglang/
│   │   ├── setup_sglang.sh      # sglang env(0.5.13) 설치
│   │   ├── restart_server.sh    # 모델 (재)기동 — 기존 종료 후 기동
│   │   └── stop_sglang.sh
│   └── litellm/
│       ├── setup_litellm.sh     # litellm 전용 env 설치
│       ├── start_litellm.sh / stop_litellm.sh
├── litellm/litellm_config.yaml  # proxy 모델 라우팅/인증
├── models/                      # 가중치 (gitignore)
└── logs/                        # 실행 로그 (gitignore)
```

### conda env (2개로 분리)
sglang 과 litellm 은 `openai` 패키지 버전 핀이 충돌해서 **각각 별도 env** 에 둔다.

| env | 용도 | 변수 |
|-----|------|------|
| `sglang` | 모델 서빙 (SGLang 0.5.13) | `CONDA_ENV` |
| `litellm` | proxy (litellm[proxy]) | `LITELLM_ENV` |

### 버전 (SGLang 0.5.13 단일)
`sglang[all]==0.5.13` / `torch==2.11.0 (cu130)` / `flashinfer[cu13]==0.6.12`.
드라이버 CUDA 13.x 기준. 핀은 [scripts/env.sh](scripts/env.sh) 에 집중.

---

## 하드웨어 호환성 규칙 (자동 적용)

`restart_server.sh` 가 GPU 를 감지해서 (모델 × HW) 조합에 맞는 인자를 자동으로 붙인다.
이건 실제 구동하며 겪은 문제들을 [scripts/registry/compat_rules.sh](scripts/registry/compat_rules.sh) 에 코드화한 것:

| 조건 | 자동 처리 |
|------|-----------|
| SM120 (Pro6000/5090) 전반 | `--disable-cuda-graph` |
| SM120 × block FP8 (qwen32) | `SGLANG_ENABLE_JIT_DEEPGEMM=0` + `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8=1` (DeepGEMM/flashinfer-trtllm FP8 가 SM120 미지원) |
| hybrid GDN (qwen27) | `--attention-backend triton --linear-attn-decode-backend triton` (없으면 gibberish) |
| MLA·DSA (deepseek) × SM120 | **기동 거부** + 안내 (SM90/SM100 에서 쓰라고) |
| NVFP4 × 비-Blackwell | 기동 거부 |
| dp-attention | deepseek(MLA) 에서만 기본 on |

override 예: `DISABLE_CUDA_GRAPH=0`, `ATTN_BACKEND=trtllm_mha`, `FORCE_SM=sm90` 등.

---

## 자주 쓰는 명령

```bash
# 설치
bash scripts/setup.sh                          # 둘 다
bash scripts/setup.sh sglang                   # sglang env 만
RECREATE=1 bash scripts/setup.sh sglang        # 클린 재설치

# 다운로드
bash scripts/download_model.sh <HF-repo-id> <저장폴더명>

# 서버 기동/종료
bash scripts/sglang/restart_server.sh <키>            # foreground (Ctrl-C 중단)
bash scripts/sglang/restart_server.sh <키> --bg       # 백그라운드 (health 대기)
bash scripts/sglang/restart_server.sh <키> --ctx 32768  # 컨텍스트 길이 지정
bash scripts/sglang/stop_sglang.sh

# proxy
bash scripts/litellm/start_litellm.sh
bash scripts/litellm/stop_litellm.sh

# 전체 / 상태 / 검증
bash scripts/start_all.sh                       # sglang(기본 모델) + litellm
bash scripts/stop_all.sh
bash scripts/status.sh
MODEL_KEY=<키> bash scripts/healthcheck.sh      # health + 스모크 테스트
```

### 주요 환경변수 (env.sh)
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CONDA_ENV` | `sglang` | sglang 서빙 env |
| `LITELLM_ENV` | `litellm` | proxy env |
| `PORT` | `30000` | SGLang 포트 |
| `LL_PORT` | `4000` | LiteLLM 포트 |
| `TP_SIZE` | GPU 개수 자동 | 텐서 병렬 수 |
| `MEM_FRAC` | `0.90` | GPU 메모리 사용 비율 |
| `CONTEXT_LEN` | `auto` | 컨텍스트 길이(모델 최대 자동) |
| `LITELLM_MASTER_KEY` | `sk-local-test` | proxy 접근 키 |

---

## 문서
- [시퀀스/컨텍스트 길이와 VRAM 조절 옵션](doc/sequence-and-vram.md) — max sequence 의 4개 층위,
  KV 캐시 계산, 조절 가능한 모든 SGLang 노브, RTX 5090 32GB 맞춤 가이드.

## 참고 / 한계
- **H200 의 DeepSeek-V3.2(MLA·DSA)** 는 별도 H200 환경에서 검증 예정 (SM90 커널 지원 → 정상 예상).
- 모든 HF 모델을 커버하진 못한다. 새 모델/아키텍처는 구동하며 에러를 만나면 `models.sh`(정체성) +
  `compat_rules.sh`(규칙) 에 반영해 확장한다.
- 모델별로 SM 아키텍처/양자화 지원이 다르다 — 위 "모델 키" 표의 비고를 참고.
