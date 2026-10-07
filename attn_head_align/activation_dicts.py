"""Hooks that collect the values (v_proj outputs) of each attention layer, plus
helpers to concatenate and clear the collected activations."""

import gc
import torch
from vfold.model_utils import get_layers


def add_hooks_to_model(model, activations_dict):
    """After each forward pass, activations_dict[layer]["v_proj_output"] hook gets that
    layer's values, move to the CPU so calibration does not fill the GPU."""
    def save_values(layer_idx):
        def hook(module, input, output):
            activations_dict[layer_idx]["v_proj_output"] = output.detach().cpu()
        return hook

    # apply the save values hook to each layer
    for layer_idx, layer in enumerate(get_layers(model)):
        if hasattr(layer, "self_attn"):
            layer.self_attn.v_proj.register_forward_hook(save_values(layer_idx))

def cat_dicts(full_activations_dict, activations_dict):
    """Append this batch's activations to the running totals, along the token axis."""
    for layer_idx in activations_dict:
        for key in activations_dict[layer_idx]:
            tensor = activations_dict[layer_idx][key].squeeze().detach().cpu() # Ensure tensor is on CPU
            # if not in dict yet, add it
            if key not in full_activations_dict[layer_idx]:
                full_activations_dict[layer_idx][key] = tensor
            else: # concatenate the new tensor to the existing tensor
                full_activations_dict[layer_idx][key] = torch.cat([
                    full_activations_dict[layer_idx][key], # existing tensor
                    tensor # new tensor
                ], dim=0) 
    return full_activations_dict

def clear_activations_dict(activations_dict):
    """Drop this batch's activations so their memory is freed before the next batch."""
    for layer_activations in activations_dict.values():
        layer_activations.clear()
    gc.collect()

