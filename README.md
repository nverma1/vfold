# VFold

Code for **VFold: Symmetry-Aware Cross-Layer Value Cache Compression**
(Neha Verma, Sungwon Kim, Kenton Murray, Kevin Duh).

VFold compresses the KV cache of a pretrained model by sharing one value cache
across adjacent layers. It works in two steps:

1. **Offline: fit and fold the alignment maps.** For each group of consecutive layers, we match
   KV heads with the Hungarian algorithm and fit a CCA map per matched head.
   The map goes into `W_V` and its inverse into `W_O`. The aligned model
   computes exactly the same function as the original.
2. **At inference: share the value cache.** Each layer attends with its own,
   unchanged values. Only after attention are the values averaged into one
   shared cache. The first 4 tokens (sinks) and the last 128 tokens stay
   unmerged; a token is merged when it leaves that window.

With pairs of layers this halves the value cache (25% of the whole KV cache).
Groups of more than two layers are also supported.

This repo is being released in stages. It currently holds the core method.
Evaluation code, baselines and the composition experiments will follow.

## Install

```bash
git clone git@github.com:nverma1/vfold.git
cd vfold
pip install -e .
```

The code imports as `vfold`. It needs a recent `transformers` with the
`Cache` / `DynamicLayer` API.

## Step 1: build an aligned checkpoint

```bash
python -m vfold.attn_head_align.align_heads \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --save-dir checkpoints
```

This runs WikiText-2 samples through the model, fits the
maps for every adjacent layer pair, folds them into the weights, and saves the
model to `checkpoints/vfold_meta-llama-Llama-3.1-8B-Instruct_nsamples_128`.
These are the paper settings. The tokenizer is not saved; load it from the
original model.

Useful options:

- `--merge-layer-groups 0:1:2:3,4:5:6:7,...` fits groups instead of pairs.
  Each group's first layer must be its lowest. Pairs use the first layer as
  the reference; larger groups use the middle layer.
- `--calib-source gsm8k` calibrates on GSM8K instead of WikiText-2.
- `--nsamples`, `--calib-seed` set the calibration size and seed.

The layer groups are saved next to the checkpoint in
`merge_layer_groups.json`. The runtime cache must use the same groups: a
mismatch does not raise an error, it just gives wrong outputs.

## Step 2: run with the shared value cache

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from vfold.attn_head_align.general_lib import load_merge_groups
from vfold.attn_head_align.shared_adjacent_cache import (
    patch_model_to_use_shared_adjacent_cache,
)

ckpt = "checkpoints/vfold_meta-llama-Llama-3.1-8B-Instruct_nsamples_128"
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

patch_model_to_use_shared_adjacent_cache(
    model,
    share_values_only=True,
    sink_tokens=4,
    critical_tokens=128,
    merge_layer_groups=load_merge_groups(ckpt),  # None -> all adjacent pairs
)

inputs = tokenizer("...", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:]))
```

Supported model types: Llama, Qwen2/Qwen3 and Mistral, including
Mistral-Small-3.1-24B (load it with `Mistral3ForConditionalGeneration`).

## What's here

| File | What it does |
|---|---|
| `attn_head_align/align_heads.py` | Step 1 entry point: calibrate, fit, fold, save. |
| `attn_head_align/align_heads_lib.py` | CCA maps, Hungarian head matching, folding into the weights. |
| `attn_head_align/activation_dicts.py` | Hooks that collect each layer's values during calibration. |
| `attn_head_align/general_lib.py` | Model loading and the merge-group layout. |
| `attn_head_align/shared_adjacent_cache.py` | The shared value cache used at inference. |
| `data_utils.py` | Calibration data. |
| `model_utils.py` | Finds the decoder layers on plain and multimodal models. |

## Citation

```bibtex
@article{verma2026vfold,
  title  = {VFold: Symmetry-Aware Cross-Layer Value Cache Compression},
  author = {Verma, Neha and Kim, Sungwon and Murray, Kenton and Duh, Kevin},
  year   = {2026}
}
```
