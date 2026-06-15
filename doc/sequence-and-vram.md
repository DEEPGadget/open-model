# 시퀀스/컨텍스트 길이와 VRAM 조절 옵션

모델을 제한된 VRAM(특히 RTX 5090 32GB)에 올릴 때, **파라미터 크기(양자화) 외에** 조절할 수 있는
시퀀스/KV/메모리 관련 옵션을 정리한다. 기준: SGLang 0.5.13.

---

## 1. "max sequence" 의 4개 층위

흔히 "max sequence" 라고 하지만 실제로는 서로 다른 4개가 있다.

| 층위 | 무엇 | 어디서 정해짐 | 우리 모델 값 |
|---|---|---|---|
| **L1. 모델 학습 max** | 모델이 학습된 최대 위치(+RoPE/YaRN 확장) | `config.json` 의 `max_position_embeddings` (+`rope_scaling`) | qwen32 **40960** / qwen3.6 **262144** / deepseek **163840** (YaRN: 원본 4096 ×40) |
| **L2. 서버 context-length** | 1요청이 쓸 수 있는 최대 토큰(프롬프트+생성) | SGLang `--context-length` (auto 면 L1 에서 유도) | 기본 `auto` = L1 |
| **L3. KV 풀 용량** | 동시 처리되는 **모든 요청의 KV 토큰 총합** 상한 | SGLang `--max-total-tokens` (미지정 시 VRAM 에서 자동 산정) | 미노출(자동) |
| **L4. per-request 생성** | 이번 요청의 생성 토큰 수 | 요청 본문 `max_tokens` | 런타임 |

> 핵심: **L2(컨텍스트 상한)와 L3(KV 풀)이 VRAM 을 결정**한다. L1 은 "넘을 수 없는 천장" 일 뿐이며,
> 넘기려면 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 이 필요하다(권장 X).

---

## 2. VRAM 구성식

```
VRAM = 가중치(param×양자화)  +  KV 캐시  +  활성값(prefill)  +  오버헤드(cudagraph 등)
        └ 본 문서 범위 밖     └ L2/L3 로 조절   └ chunked-prefill   └ SM120 은 이미 off
```

---

## 3. KV 캐시 per-token 크기 (조절의 핵심)

표준 GQA 공식:

```
bytes/token = 2(K,V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

| 모델 | 계산 | per-token (fp16) | per-token (fp8 KV) |
|---|---|---|---|
| **Qwen3-32B** | 2·64·8·128·2 | **0.25 MB** | 0.125 MB |
| **Qwen3.6-27B** (hybrid GDN, full-attn 16층만 KV) | 2·16·4·256·2 | **0.125 MB** + GDN 상태(거의 상수) | 0.0625 MB |
| **DeepSeek-V3.2** (MLA, latent 압축) | ~61·(512+64)·2 | **~0.07 MB** | — |

컨텍스트별 KV(단일 요청 기준):

| 컨텍스트 | Qwen3-32B fp16 | Qwen3-32B fp8 | Qwen3.6-27B fp16 |
|---|---|---|---|
| 4K  | ~1.0 GB | ~0.5 GB | ~0.5 GB |
| 32K | ~8 GB   | ~4 GB   | ~4 GB   |
| 40K(최대) | ~10 GB | ~5 GB | ~5 GB |

**관찰**
1. fp8 KV 는 정확히 절반.
2. Qwen3.6 의 hybrid GDN 은 16층만 KV(나머지는 고정 크기 recurrent 상태)라 같은 길이에서 KV 가 훨씬 작다 → 장문/저VRAM 유리.
3. DeepSeek MLA 는 KV 가 매우 작지만 5090(SM120)에서 MLA 커널 미지원이라 논외.

---

## 4. 조절 가능한 모든 노브 (SGLang 0.5.13)

| 노브 | 무엇 | VRAM 효과 | 현재 노출? |
|---|---|---|---|
| `--mem-fraction-static` | 정적 풀(가중치+KV) GPU 비율 | 전체 풀 크기 | ✅ `MEM_FRAC` (0.90) |
| `--context-length` | L2 1요청 최대 토큰 | KV 상한 ↓ | ✅ `CONTEXT_LEN` (auto) |
| `--tp` | 가중치/KV 를 여러 GPU 로 분산 | 장당 부담 ↓ | ✅ `TP_SIZE` |
| `--max-total-tokens` | L3 KV 풀 총 토큰 | **KV 직접 캡** | ❌ 미노출 |
| `--max-running-requests` | 동시 요청 수 | 배치 KV ↓ | ❌ |
| `--kv-cache-dtype fp8_e4m3` | KV 를 FP8 로 | **KV 절반** | ❌ |
| `--chunked-prefill-size` | prefill 청크 토큰 | 활성값 peak ↓ | ❌ |
| `--max-prefill-tokens` | prefill 배치 상한 | 활성값 ↓ | ❌ |
| `--cpu-offload-gb` | 가중치 일부 CPU 로 | 가중치 VRAM ↓ (느려짐) | ❌ |
| `--page-size` | KV 페이징 단위 | 단편화 | ❌ |
| `--disable-cuda-graph` | 그래프 캡처 끔 | 캡처 메모리 ↓ | ✅ 자동(SM120) |

효과 크기 순(대략): **kv-cache fp8 ≈ context-length ↓ > max-total-tokens / max-running-requests > chunked-prefill > mem-fraction**.

---

## 5. RTX 5090 32GB 현실 점검

**냉정한 사실**: 현재 3개 모델은 단일 5090(32GB)에 **가중치만으로 안 들어간다**.

- Qwen3-32B-FP8: 가중치 ~32GB → KV 자리 0 → 불가
- Qwen3.6-27B bf16: ~54GB → 불가
- DeepSeek: 논외(크기 + MLA SM120 미지원)

따라서 5090 32GB 는 둘 중 하나다.

1. **더 작은/더 양자화된 모델** (예: 14B FP8 ~14GB, 7~8B 등) → 남은 공간을 아래 노브로 32GB 에 맞춤
2. **여러 5090 을 `--tp` 로 분산** (예: 32B FP8 를 2장 TP2)

**"param 외" 로 짜내는 예** (가중치 ~14GB 모델을 32GB 에, 남은 ~16GB 를 KV/활성에 배분):

```bash
CONTEXT_LEN=16384 \                  # L2: 컨텍스트 16K 제한 (KV 상한 ↓)
MEM_FRAC=0.92 \                      # 정적 풀 비율
... --kv-cache-dtype fp8_e4m3 \      # KV 절반 (효과 가장 큼)
    --max-total-tokens 32768 \       # KV 풀 총량 캡
    --max-running-requests 4 \       # 동시성 제한
    --chunked-prefill-size 2048      # prefill peak 억제
```

---

## 6. 현재 스크립트의 갭 + 개선 방향

지금 [scripts/env.sh](../scripts/env.sh) 로 조절 가능한 건 **`CONTEXT_LEN`, `MEM_FRAC`, `TP_SIZE`** 3개뿐이다.
저VRAM 시나리오에 가장 중요한 다음 노브가 아직 노출되지 않았다.

- `--kv-cache-dtype` (KV 절반)
- `--max-total-tokens` (KV 풀 캡)
- `--max-running-requests` (동시성)
- `--chunked-prefill-size` (활성값 peak)
- `--cpu-offload-gb` (가중치 일부 CPU)

**개선안**
1. env.sh 에 위 노브를 환경변수로 추가하고 `restart_server.sh` 의 `EXTRA_ARGS` 로 (설정된 경우만) 전달.
   예: `KV_CACHE_DTYPE / MAX_TOTAL_TOKENS / MAX_RUNNING_REQUESTS / CHUNKED_PREFILL_SIZE / CPU_OFFLOAD_GB`
2. `compat_rules.sh` 에 **"VRAM 작은 GPU 자동 감지(detect_hw 의 `hw_vram_mib`) → 보수적 기본값(kv fp8, 컨텍스트 축소)"** 규칙 추가.

---

## 부록: 모델별 KV 관련 파라미터 (config.json)

| 모델 | layers | kv_heads | head_dim | 특이 |
|---|---|---|---|---|
| Qwen3-32B | 64 | 8 | 128 | 표준 GQA |
| Qwen3.6-27B | 64 (full-attn 16) | 4 | 256 | hybrid GDN, `full_attention_interval=4` |
| DeepSeek-V3.2 | 61 | (MLA) | — | `kv_lora_rank=512`, YaRN |
