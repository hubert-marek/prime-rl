# Agent Diff Bench on Merged Ministral-3

This folder contains the local pieces needed to run `agent-diff-bench` RL on
the merged `Ministral-3` checkpoint without committing secrets.

## Files

- `agent-diff.toml` — original environment/sampling scaffold
- `rl_ministral_merged.toml` — ready `uv run rl` config using an external inference server
- `secrets.env` — local-only environment variables for the environment, ignored by git

## Merge the adapter

Merge the LoRA adapter into a normal HF checkpoint directory:

```bash
uv run python merge_model.py
```

This writes:

```text
outputs/Ministral-3-14B-Agent-Diff-SFT-merged
```

The merged output is an HF-style directory. Do not force `config_format =
"mistral"` or `load_format = "mistral"` when serving it.

## Start inference

Serve the merged model first. This repo includes a local vLLM workaround for
the nested `ministral3` text tower used by the merged checkpoint.

Example config:

```toml
gpu_memory_utilization = 0.5
vllm_extra = { attention_backend = "FLASH_ATTN" }

[server]
port = 8010

[model]
name = "outputs/Ministral-3-14B-Agent-Diff-SFT-merged"
dtype = "bfloat16"
enforce_eager = true
max_model_len = 2048
```

Start it with:

```bash
uv run inference @ /path/to/infer.toml
```

Quick check:

```bash
curl http://127.0.0.1:8010/v1/models
```

## Run RL

The provided RL config assumes the inference server is already running at
`http://127.0.0.1:8010/v1` and uses one training GPU locally:

```bash
uv run rl @ configs/agent_diff_bench/rl_ministral_merged.toml
```

If you want to scale up later, increase:

- `seq_len`
- `orchestrator.batch_size`
- `orchestrator.rollouts_per_example`
- `orchestrator.sampling.max_tokens`

Start small first and verify a few steps end-to-end.
