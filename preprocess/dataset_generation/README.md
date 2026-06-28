# Dataset Generation

Pipeline to build training pairs from `train` split retrieval results and generate
`match/not_match` reasoning + temporal grounding with a 30B VLM.

## Steps

1. Build top-1 pool (balanced match/not-match):

```bash
bash preprocess/dataset_generation/scripts/0_build_top1_pool.bash
```

2. (Optional) Start vLLM 30B server for `openai_api` backend:

```bash
bash preprocess/dataset_generation/scripts/1_vllm_30b.bash
```

3. Generate reasoning + temporal grounding:

```bash
bash preprocess/dataset_generation/scripts/2_generate_match_grounding.bash
```

Default local settings are aligned to ActivityNet extracted npy:
- `PREFER_NPY=1`
- `VIDEO_NPY_ROOT=data/activitynet/train/video_npy_with_meta`
- `VIDEO_META_JSONL=${VIDEO_NPY_ROOT}/meta.jsonl` (injects `raw_fps`/`frames_indices`)
- `VIDEO_FPS=1.0`
- `VIDEO_MAX_PIXELS=200704` (`256 * 28 * 28`)
- `LOCAL_IMAGE_PATCH_SIZE=14` (effective factor `28`)

`2_generate_match_grounding.bash` defaults to `BACKEND=local_vllm` (colocated
vLLM + local vision preprocessing). To use HTTP server mode instead:

```bash
BACKEND=openai_api bash preprocess/dataset_generation/scripts/2_generate_match_grounding.bash
```

To avoid repeated model loading with `BACKEND=local_vllm`, run multiple jobs in one process:

```bash
cat > /tmp/match_jobs.jsonl <<'JSONL'
{"pool_jsonl":"data/dataset_generation/top1_pool.train.jsonl","output_jsonl":"data/dataset_generation/top1_reasoning_grounding.train.jsonl"}
{"pool_jsonl":"data/dataset_generation/top1_pool.val.jsonl","output_jsonl":"data/dataset_generation/top1_reasoning_grounding.val.jsonl"}
JSONL
JOBS_JSONL=/tmp/match_jobs.jsonl bash preprocess/dataset_generation/scripts/2_generate_match_grounding.bash
```

## Default outputs

- Pool: `data/dataset_generation/top1_pool.train.jsonl`
- Generated: `data/dataset_generation/top1_reasoning_grounding.train.jsonl`
