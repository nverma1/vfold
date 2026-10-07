"""
Main runner script to apply invariant transformations and write the model back down to disk.
"""

from collections import defaultdict
import torch

from vfold.attn_head_align.activation_dicts import (
    add_hooks_to_model,
    cat_dicts,
    clear_activations_dict,
)
from vfold.attn_head_align.align_heads_lib import do_alignment, apply_v_alignment
from vfold.attn_head_align.general_lib import (
    load_model_and_tokenizer,
    format_merge_groups,
    parse_merge_groups,
    resolve_groups,
    save_merge_groups,
)
from vfold.data_utils import CALIB_SOURCES, get_calib_dataset, prepare_dataloader
from vfold.model_utils import get_text_config


def main(args):
     
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, 
        num_gpus=torch.cuda.device_count()
    )

    # Instantiate activations dictionaries
    total_activations_dict = defaultdict(dict)
    activations_dict = defaultdict(dict)

    # add hooks to the model to get the activations
    add_hooks_to_model(model, activations_dict)

    # Get dataset and dataloader for calibratoin
    dataset = get_calib_dataset(args.calib_source, split="train")
    dataloader = prepare_dataloader(
        dataset, tokenizer, batch_size=1, nsamples=args.nsamples, seed=args.calib_seed
    )

    # print calibration info, throw error if not enough samples
    n_packed = len(dataloader.sampler)
    print(f"Calibration: {args.calib_source} train, seed {args.calib_seed}, {n_packed} packed samples")
    if n_packed < args.nsamples:
        raise ValueError(
            f"{args.calib_source} had only {n_packed} packed samples; asked for {args.nsamples}"
        )

    # Use inference_mode & populate total_activations_dict
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            # Handle device placement based on whether device_map is used
            first_param_device = next(model.parameters()).device
            inputs = batch["input_ids"].to(first_param_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(first_param_device, non_blocking=True)

            # Forward pass
            _ = model(inputs, attention_mask=attention_mask)

            # Delete input and mask for memory efficiency
            del inputs, attention_mask

            # Process and accumulate activations. Then, clear the activations dictionary.
            total_activations_dict = cat_dicts(total_activations_dict, activations_dict)
            clear_activations_dict(activations_dict)

            # Clear GPU cache on all GPUs the model is actually using.
            for device in {p.device for p in model.parameters() if p.device.type == 'cuda'}:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # parse merge groups, put list of groups in merge_layer_groups
    n_layers = get_text_config(model).num_hidden_layers
    merge_layer_groups = (
        parse_merge_groups(args.merge_layer_groups) if args.merge_layer_groups else None
    )
    merge_layer_groups = resolve_groups(n_layers, merge_layer_groups, None)
    group_sizes = sorted({len(g) for g in merge_layer_groups})
    print(
        f"alignment groups: {len(merge_layer_groups)} groups, sizes={group_sizes}, "
        f"-> {format_merge_groups(merge_layer_groups)}"
    )

    # Get alignment matrices from activations
    alignment = do_alignment(
        total_activations_dict,
        num_kv_heads=get_text_config(model).num_key_value_heads,
        head_dim=get_text_config(model).head_dim,
        verbose=args.verbose,
        merge_layer_groups=merge_layer_groups,
    )

    # Fold the value alignment maps into the model weights
    model = apply_v_alignment(model, alignment, verbose=args.verbose)

    # Save the aligned model, record the layout this checkpoint was aligned for
    safe_name = args.model_name.replace("/", "-")
    save_path = f"{args.save_dir}/vfold_{safe_name}_nsamples_{args.nsamples}"
    if args.calib_source != "wikitext":
        save_path = f"{save_path}_calib-{args.calib_source}"
    if args.calib_seed != 42:
        save_path = f"{save_path}_seed{args.calib_seed}"
    if max(len(g) for g in merge_layer_groups) > 2:
        sizes = "-".join(str(n) for n in sorted({len(g) for g in merge_layer_groups}))
        save_path = f"{save_path}_groups{sizes}"
    model.save_pretrained(save_path)

    # Record a json file with the merge group layout and the reference layers
    reference_layers = {layer_idx: alignment[layer_idx][2] for layer_idx in alignment}
    save_merge_groups(save_path, merge_layer_groups, reference_layers)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples (2048 tokens each)")
    parser.add_argument("--calib-source", type=str, default="wikitext", choices=CALIB_SOURCES,
                        help="Calibration corpus (train split), packed to 2048 tokens")
    parser.add_argument("--calib-seed", type=int, default=42, help="Seed for sampling calibration sequences")
    parser.add_argument(
        "--merge-layer-groups", type=str, default=None,
        help="Colon-separated layer groups sharing one cache, comma-separated between "
             "groups, e.g. '0:1:2:3,4:5:6:7'. The first index of each group owns the "
             "buffer and must be the group's lowest layer. Omit for all adjacent pairs.",
    )
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name") # or "Qwen/Qwen3-8B", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save the aligned models")

    args = parser.parse_args()
    main(args)

