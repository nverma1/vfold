"""Helpers for reaching a model's decoder stack regardless of its wrapper class.

Multimodal checkpoints nest the text decoder one level deeper than plain causal
LMs. Mistral-Small-3.1-24B-Instruct-2503, for example, loads as
Mistral3ForConditionalGeneration, whose layers live at
`model.model.language_model.layers` -- `model.model.layers` raises
AttributeError. Anything that iterates decoder layers or pokes
`self_attn.config` must go through here so it works for both shapes.

Kept import-light on purpose (no torch/transformers at module scope) so the
compression patch modules can import it cheaply; `load_model` imports
transformers lazily.
"""


def get_text_model(model):
    """Return the module that owns the decoder `layers` ModuleList."""
    inner = getattr(model, "model", model)

    # Plain causal LM: LlamaForCausalLM.model.layers
    if hasattr(inner, "layers"):
        return inner

    # VLM wrapper: Mistral3ForConditionalGeneration.model.language_model.layers
    lm = getattr(inner, "language_model", None)
    if lm is not None:
        if hasattr(lm, "layers"):
            return lm
        # Some wrappers nest a full *ForCausalLM under .language_model
        nested = getattr(lm, "model", None)
        if nested is not None and hasattr(nested, "layers"):
            return nested

    raise AttributeError(
        f"Could not locate decoder layers on {type(model).__name__}. "
        f"Add its layout to vfold/model_utils.get_text_model."
    )


def get_layers(model):
    """Return the decoder layer ModuleList."""
    return get_text_model(model).layers


def get_num_layers(model):
    return len(get_layers(model))


def get_text_config(model):
    """Return the config governing the text decoder.

    On multimodal wrappers `model.config` is the composite config; the decoder
    (and therefore `self_attn.config`) reads from `config.text_config`.
    """
    config = model.config
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        return getter()
    return config


def load_model(model_path, tokenizer_path=None, **model_kwargs):
    """Load a checkpoint with the class its family requires.

    Mistral-Small 3.x is Mistral3ForConditionalGeneration (a Pixtral-style VLM);
    transformers registers it under AutoModelForImageTextToText, so
    AutoModelForCausalLM.from_pretrained raises "Unrecognized configuration
    class" for it with no fallback. Its decoder lives at
    model.model.language_model, reached via get_layers. Aligned checkpoints keep
    the base model id in their path, so a substring check finds them too.
    """
    paths = f"{tokenizer_path or ''} {model_path}".lower()
    if "mistral-small" in paths:
        from transformers import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration.from_pretrained(
            model_path, **model_kwargs
        )

    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
