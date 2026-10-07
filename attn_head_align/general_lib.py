import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration

def load_model_and_tokenizer(model_name, num_gpus):
    """Load the model for calibration (not inference).
    Uses eager attention, as when the paper checkpoints were built; other attention
    kernels give slightly different activations and so slightly different maps.
    """
    model_kwargs = dict(
        torch_dtype="auto",
        device_map="auto" if num_gpus > 1 else "cuda",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    # Mistral-Small 3.x is a vision-language model that AutoModelForCausalLM cannot
    # load. Use get_layers (vfold.model_utils) to reach its decoder layers.
    if "Mistral-Small" in model_name:
        model = Mistral3ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Merge groups
#
# A group is a list of consecutive layers whose aligned values are averaged into
# one shared value cache, which every member reads (paper, Sec. 4 and App. D).
# Pairs are groups of size 2.
#
# Two different roles inside a group:
#   - the reference layer, whose frame the others are aligned into
#     (select_group_reference);
#   - the storage layer, group[0], which holds the shared cache in memory because
#     it runs first.
#
# Alignment and the runtime cache both use these helpers, so they always agree
# on the layout.
# ---------------------------------------------------------------------------

MERGE_GROUPS_FILENAME = "merge_layer_groups.json"


def parse_merge_groups(spec: str) -> list[list[int]]:
    """Parse "2:3:4:5,6:7:8:9" into [[2, 3, 4, 5], [6, 7, 8, 9]]."""
    groups = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            groups.append([int(x) for x in chunk.split(":")])
        except ValueError as e:
            raise ValueError(
                f"Bad merge group {chunk!r} in spec {spec!r}: expected colon-separated ints"
            ) from e
    if not groups:
        raise ValueError(f"No merge groups parsed from spec {spec!r}")
    return groups


def format_merge_groups(groups: list[list[int]]) -> str:
    """Inverse of parse_merge_groups, for logging and for round-tripping to disk."""
    return ",".join(":".join(str(i) for i in g) for g in groups)


def groups_from_pairs(pairs) -> list[list[int]]:
    """Turn (owner, member) pairs into 2-layer groups."""
    return [[int(source), int(consumer)] for source, consumer in pairs]


def default_adjacent_groups(num_layers: int) -> list[list[int]]:
    """The default layout: all adjacent pairs (0,1), (2,3), ..."""
    return [[i, i + 1] for i in range(0, num_layers - 1, 2)]


def validate_groups(groups: list[list[int]], num_layers: int) -> None:
    """Raise ValueError on a layout the cache cannot use.

    Each group needs at least 2 distinct layers, groups must not overlap, and the
    first layer of each group must be its lowest, because it stores the shared
    cache and so must run before the other members.
    """
    seen: dict[int, int] = {}
    for gi, g in enumerate(groups):
        if len(g) < 2:
            raise ValueError(f"Merge group {g} has size {len(g)}; groups must have >= 2 layers")
        if len(set(g)) != len(g):
            raise ValueError(f"Merge group {g} repeats a layer index")
        for layer in g:
            if not 0 <= layer < num_layers:
                raise ValueError(
                    f"Merge group {g} references layer {layer}, outside [0, {num_layers})"
                )
            if layer in seen:
                raise ValueError(
                    f"Layer {layer} appears in both group {groups[seen[layer]]} and {g}; "
                    "merge groups must be disjoint"
                )
            seen[layer] = gi
        if g[0] != min(g):
            raise ValueError(
                f"Merge group {g} must list its lowest layer first (got owner {g[0]}, "
                f"minimum {min(g)}): the owner holds the shared buffer and must run first"
            )


def resolve_groups(
    num_layers: int,
    merge_layer_groups: list[list[int]] | None = None,
    merge_layer_pairs=None,
) -> list[list[int]]:
    """Pick the group layout from (groups | pairs | default) and validate it."""
    if merge_layer_groups is not None and merge_layer_pairs is not None:
        raise ValueError("merge_layer_groups and merge_layer_pairs are mutually exclusive")
    if merge_layer_groups is not None:
        groups = [list(g) for g in merge_layer_groups]
    elif merge_layer_pairs is not None:
        groups = groups_from_pairs(merge_layer_pairs)
    else:
        groups = default_adjacent_groups(num_layers)
    validate_groups(groups, num_layers)
    return groups


def select_group_reference(group: list[int]) -> int:
    """The group member whose values the other members align into (paper, App. D).

    For a group of k consecutive layers starting at layer l, the reference is the
    middle layer l + k // 2. Pairs use the first layer, l.
    """
    return group[0] if len(group) == 2 else group[len(group) // 2]


def save_merge_groups(save_path: str, groups: list[list[int]], references: dict | None = None) -> None:
    """Save the group layout (and each layer's reference) next to the checkpoint.

    The runtime cache must use the layout the maps were fit for. A mismatch would
    not raise an error; it would just give wrong outputs. This file lets the eval
    scripts check.
    """
    info_dict = {
        "merge_layer_groups": [list(g) for g in groups],
        "spec": format_merge_groups(groups),
    }
    if references is not None:
        info_dict["reference_layers"] = {str(k): int(v) for k, v in references.items()}
    out = Path(save_path) / MERGE_GROUPS_FILENAME
    out.write_text(json.dumps(info_dict, indent=2))
    print(f"  Saved merge group layout to {out}")


def load_merge_groups(checkpoint_path: str) -> list[list[int]] | None:
    """Read the group layout a checkpoint was aligned for.

    Returns None if the file is missing. Checkpoints built before this file existed
    (including the paper's pairs checkpoints) use adjacent pairs.
    """
    path = Path(checkpoint_path) / MERGE_GROUPS_FILENAME
    if not path.exists():
        return None
    info_dict = json.loads(path.read_text())
    return [list(g) for g in info_dict["merge_layer_groups"]]