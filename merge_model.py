from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

"""Merge the Agent Diff LoRA adapter into a normal HF Ministral-3 checkpoint.

The output directory is a Hugging Face-style model folder suitable for:
- direct Transformers loading
- `uv run inference` using the normal HF loading path

Do not force `config_format = "mistral"` or `load_format = "mistral"` when
serving the merged output.
"""

model_id = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
adapter_repo = "hubertmarek/Ministral-3-14B-Agent-Diff-SFT-LoRA"
output_dir = Path("outputs/Ministral-3-14B-Agent-Diff-SFT-merged")

base = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

model = PeftModel.from_pretrained(base, adapter_repo)
merged = model.merge_and_unload()

merged.save_pretrained(
    output_dir,
    safe_serialization=True,
    max_shard_size="5GB",
)
processor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Merged model written to {output_dir}")
