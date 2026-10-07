"""
The VFold KV cache: layers in a merge group share one value cache

The first layer of each group (the owner) stores the shared cache. Each later
member attends with its own values for the newest tokens, then folds them into
the shared cache as a running average. The first few tokens (sinks) and the most
recent ones (the critical window) are kept per layer; a token leaving the
window is folded in then.

share_keys_only runs the same merge on keys instead. It is only used for the
keys-vs-values comparison in the paper.
"""

from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicLayer
from vfold.model_utils import get_layers, get_text_config, get_text_model
from .general_lib import resolve_groups


class SharedAdjacentLayer(DynamicLayer):
    """
    A non-cache owner member of a merge group. It shares one component (values, or keys
    for Table 1) with the cache owner, source_layer, and stores the other itself.

    share_keys / share_values: which component is shared. Exactly one must be True.
    """

    def __init__(
        self,
        source_layer: DynamicLayer, # this is the cache owner pointed to
        share_keys: bool = False,
        share_values: bool = True,
        sink_tokens: int = 0,
        critical_tokens: int = 0,
        group_index: int = 1, # 1 indexed for math purposes
    ):
        if share_keys == share_values:
            raise ValueError("share exactly one of keys or values")
        if group_index < 1:
            raise ValueError(f"group_index must be >= 1 (the owner is 0), got {group_index}")

        super().__init__()
        self.source_layer = source_layer
        self.share_keys = share_keys
        self.share_values = share_values
        self.sink_tokens = sink_tokens
        self.critical_tokens = critical_tokens
        self.group_index = group_index
        # Weight on the shared cache when this member folds itself in.
        self.source_weight = group_index / (group_index + 1)

        self._prefill_consumed = False
        self._sink_keys: torch.Tensor | None = None
        self._sink_values: torch.Tensor | None = None
        self._critical_keys: torch.Tensor | None = None
        self._critical_values: torch.Tensor | None = None
        # Absolute sequence positions of the entries in the critical window, so
        # eviction knows which slot of the shared buffer each one belongs to.
        self._critical_pos: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        if not self.share_keys:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        if not self.share_values:
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        if self.sink_tokens > 0 or self.critical_tokens > 0:
            self._sink_keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self._sink_values = torch.tensor([], dtype=self.dtype, device=self.device)
            self._critical_keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self._critical_values = torch.tensor([], dtype=self.dtype, device=self.device)
            self._critical_pos = torch.tensor([], dtype=torch.long, device=self.device)
        self.is_initialized = True
    
    def _splice_with_exclusions(
        self,
        keys_out: torch.Tensor,
        values_out: torch.Tensor,
        keys_are_own_storage: bool = False,
        values_owned: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace sink and critical positions with our own stored K/V.

        ``keys_are_own_storage`` -- ``keys_out`` is this layer's own K cache with
        nothing applied on top. ``_store_sink_critical`` sliced the sink/critical
        key buffers out of the same ``key_states`` that were concatenated into
        it, so the writes below would assign each element the value it already
        holds. Skip them, and skip the clone that only existed to hold them.

        ``values_owned`` -- ``values_out`` was freshly built by this call (the
        ``torch.cat`` in ``_attention_states_v_merge``) and aliases nothing, so it
        can be written in place. The first call returns the caller's own tensor
        instead, which must still be copied.

        Both default to False, so a caller that has not reasoned about aliasing
        keeps the original copy-first behaviour.
        """

        # if no sink or critical tokens, return the original keys and values
        if self.sink_tokens == 0 and self.critical_tokens == 0:
            return keys_out, values_out

        # get the number of sink and critical tokens
        seq_len = keys_out.shape[-2]
        n_sink = min(self.sink_tokens, seq_len)
        n_critical = min(self.critical_tokens, seq_len)
        if n_sink == 0 and n_critical == 0:
            return keys_out, values_out

        # replace the sink and critical positions with the stored K/V
        splice_keys = not keys_are_own_storage # this is true if the keys are not owned by this layer
        if splice_keys:
            keys_out = keys_out.clone()
        if not values_owned: # this is true if the values are not owned by this layer
            values_out = values_out.clone()

        # replace the sink and critical positions with the stored K/V
        if n_sink > 0 and self._sink_keys is not None and self._sink_keys.numel() > 0:
            n_sink_actual = min(n_sink, self._sink_keys.shape[-2])
            if splice_keys:
                keys_out[..., :n_sink_actual, :] = self._sink_keys[..., :n_sink_actual, :]
            values_out[..., :n_sink_actual, :] = self._sink_values[..., :n_sink_actual, :]
        if n_critical > 0 and self._critical_keys is not None and self._critical_keys.numel() > 0:
            n_crit_actual = min(n_critical, self._critical_keys.shape[-2])
            if splice_keys:
                keys_out[..., -n_crit_actual:, :] = self._critical_keys[..., -n_crit_actual:, :]
            values_out[..., -n_crit_actual:, :] = self._critical_values[..., -n_crit_actual:, :]
        return keys_out, values_out

    def _store_sink_critical(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Store sink and critical tokens from the incoming batch.

        Returns ``(positions, keys, values)`` for the entries pushed out of the
        critical window by this call, or None if nothing was evicted. The window
        is full the moment prefill ends, so during generation this returns one
        entry per step.
        """

        # get the number of new tokens created
        new_n = key_states.shape[-2]
        seq_len = start_pos + new_n

        # Sink: positions [0, sink_tokens). In batch: indices [0, min(new_n, sink_tokens - start_pos))
        n_sink_new = max(0, min(new_n, self.sink_tokens - start_pos))

        # Critical: positions [seq_len - critical_tokens, seq_len). In batch: last min(new_n, critical_tokens) tokens
        n_critical_new = min(new_n, self.critical_tokens) if self.critical_tokens > 0 else 0

        if n_sink_new > 0:
            # get the slice of the key and value states for the sink tokens
            sink_slice_k = key_states[..., :n_sink_new, :]
            sink_slice_v = value_states[..., :n_sink_new, :]
            if self._sink_keys.numel() == 0:
                self._sink_keys = sink_slice_k.clone()
                self._sink_values = sink_slice_v.clone()
            else:
                self._sink_keys = torch.cat([self._sink_keys, sink_slice_k], dim=-2)
                self._sink_values = torch.cat([self._sink_values, sink_slice_v], dim=-2)
            self._sink_keys = self._sink_keys[..., : self.sink_tokens, :]
            self._sink_values = self._sink_values[..., : self.sink_tokens, :]

        if n_critical_new > 0:
            # get the slice of the key and value states for the critical tokens
            crit_slice_k = key_states[..., -n_critical_new:, :]
            crit_slice_v = value_states[..., -n_critical_new:, :]
            # We take the *last* n_critical_new of the batch, so these end at
            # seq_len - 1. Tracking positions explicitly rather than deriving them
            # from the buffer length keeps eviction correct even if a gap ever
            # opens up (see the contiguity check below).
            crit_pos_new = torch.arange(
                seq_len - n_critical_new, seq_len, device=key_states.device
            )
            if self._critical_keys.numel() == 0:
                self._critical_keys = crit_slice_k.clone()
                self._critical_values = crit_slice_v.clone()
                self._critical_pos = crit_pos_new
            else:
                self._critical_keys = torch.cat([self._critical_keys, crit_slice_k], dim=-2)
                self._critical_values = torch.cat([self._critical_values, crit_slice_v], dim=-2)
                self._critical_pos = torch.cat([self._critical_pos, crit_pos_new])

            # _splice_with_exclusions writes this buffer onto the trailing
            # positions of the output, which is only correct while the window is
            # a contiguous suffix. A multi-token step shorter than critical_tokens
            # would skip a position and silently misplace every value after it.
            # Greedy decoding never does that; fail loudly if anything else does.
            n_held = self._critical_pos.shape[0]
            if n_held > 1 and int(self._critical_pos[-1] - self._critical_pos[0]) != n_held - 1:
                raise RuntimeError(
                    "critical window is not a contiguous run of positions "
                    f"({self._critical_pos[0].item()}..{self._critical_pos[-1].item()} "
                    f"holding {n_held}); this happens when a forward pass adds more "
                    "than one but fewer than critical_tokens positions"
                )

            n_evict = n_held - self.critical_tokens
            evicted = None
            if n_evict > 0:
                # Views into the pre-truncation buffers, which stay alive as long
                # as the caller holds them.
                evicted = (
                    self._critical_pos[:n_evict],
                    self._critical_keys[..., :n_evict, :],
                    self._critical_values[..., :n_evict, :],
                )
            self._critical_keys = self._critical_keys[..., -self.critical_tokens :, :]
            self._critical_values = self._critical_values[..., -self.critical_tokens :, :]
            self._critical_pos = self._critical_pos[-self.critical_tokens :]
            return evicted
        return None

    def _blend(self, source: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
        """Running average: shared cache weighted k/(k+1), this member 1/(k+1)."""
        a = self.source_weight
        if a == 0.5:
            return 0.5 * (source + own)
        return a * source + (1.0 - a) * own

    '''
    This function takes in the start position and the number of tokens to merge, 
    and returns a mask of the positions that should receive in-place V merge.
    The other positions should receive the raw own V.
    '''
    def _persist_merge_mask(
        self,
        start_pos: int,
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        # if no tokens to merge, return an empty mask
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        # if no sink or critical tokens, return a mask of all positions to merge
        if self.sink_tokens == 0 and self.critical_tokens == 0:
            return torch.ones(n, dtype=torch.bool, device=device)
        
        # get the positions of the tokens to look at
        seq_len = start_pos + n
        positions = torch.arange(start_pos, start_pos + n, device=device)
        
        # if sink, return a mask of the positions that are in the sink
        in_sink = positions < self.sink_tokens
        # if critical, return a mask of the positions that are in the critical
        in_critical = (
            positions >= seq_len - self.critical_tokens
            if self.critical_tokens > 0
            else torch.zeros(n, dtype=torch.bool, device=device)
        )
        # return a mask of the positions that are not in the sink or critical
        return ~(in_sink | in_critical)

    def _write_merger_in_place(
        self,
        source_states: torch.Tensor,
        own_states: torch.Tensor,
        n: int,
        start_pos: int,
    ) -> None:
        """Overwrite selected positions in the last ``n`` source slots with the linear merger.

        Component-agnostic: ``source_states``/``own_states`` are the leader's and
        follower's V (the deployed case) or K (the K-vs-V measurement case).
        """
        if n == 0:
            return
        tail = source_states[..., -n:, :]
        mask = self._persist_merge_mask(start_pos, n, tail.device)
        if not mask.any():
            return
        # Under device_map="auto" the model is split across GPUs, so the source
        # layer and its consumer can land on different devices. We write in
        # place into the source, so bring own_states to the source's device
        # rather than the other way round.
        if own_states.device != tail.device:
            own_states = own_states.to(tail.device)
        merged = self._blend(tail, own_states)
        if mask.all():
            tail.copy_(merged)
            return
        mask_expanded = mask.view(*([1] * (tail.dim() - 2)), n, 1)
        tail.copy_(torch.where(mask_expanded, merged, tail))

    def _merge_evicted_into_source(
        self,
        source_states: torch.Tensor,
        own_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Fold states leaving the critical window into the shared buffer.

        KV-agnostic, like ``_write_merger_in_place``: ``own_states`` are
        the follower's V (the deployed case) or K.

        Positions in the critical window were skipped by the in-place merge when
        they were written, so the buffer still holds the leader's raw state
        there. Blending on the way out is what stops those positions from
        silently degrading to naive sharing once they age out. Each position is
        evicted exactly once, so this cannot double-merge.
        """
        if source_states is None or positions.numel() == 0:
            return
        # A position can outrun the buffer if the leader has not yet been written
        # this step. Skip rather than index out of bounds.
        in_range = positions < source_states.shape[-2]
        if not bool(in_range.all()):
            positions = positions[in_range]
            own_states = own_states[..., in_range, :]
            if positions.numel() == 0:
                return

        # Under device_map="auto" the leader and its follower can sit on
        # different GPUs. We write into the leader, so move the follower's states.
        idx = positions.to(source_states.device)
        if own_states.device != source_states.device:
            own_states = own_states.to(source_states.device)

        dim = source_states.dim() - 2
        current = source_states.index_select(dim, idx)
        merged = self._blend(current, own_states)
        source_states.index_copy_(dim, idx, merged.to(source_states.dtype))

    def _attention_states_v_merge(
        self,
        source_states: torch.Tensor,
        own_states: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build follower attention states: raw own at new positions, merger at older ones.

        Component-agnostic, same as ``_write_merger_in_place``. Only one component is
        ever persisted per layer (see ``update``), so a single ``_prefill_consumed``
        flag is sufficient.
        """
        if not self._prefill_consumed:
            self._prefill_consumed = True
            return own_states
        return torch.cat(
            [source_states[..., :-n, :].to(device), own_states], dim=-2
        )

    def reset(self) -> None:
        """Clear follower-local state for a new sequence."""
        self._prefill_consumed = False
        if (
            not self.share_keys
            and self.is_initialized
            and hasattr(self, "keys")
            and self.keys.numel() > 0
        ):
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        # Symmetric case: under share_keys_only the follower stores its own values,
        # which would otherwise carry over into the next sequence.
        if (
            not self.share_values
            and self.is_initialized
            and hasattr(self, "values")
            and self.values.numel() > 0
        ):
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,  # [batch, heads, new tokens, head_dim]
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,  # unused, but callers pass it
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Position of the first new token.
        new_n = key_states.shape[-2]
        if self.share_keys:
            # start position is the length of the sequence minus the number of new tokens
            start_pos = (
                self.source_layer.get_seq_length() - new_n
                if self.source_layer.is_initialized and self.source_layer.keys.numel() > 0
                else 0
            )
        else:
            # start position is the length of the keys tensor
            start_pos = self.keys.shape[-2] if self.keys.numel() > 0 else 0

        # If there are sink or critical tokens, store them. 
        evicted = None
        if self.sink_tokens > 0 or self.critical_tokens > 0: 
            # evicted is a tuple of the positions, keys, and values that are stored uncompressed
            evicted = self._store_sink_critical(key_states, value_states, start_pos)

        # The owner has not written yet (it has nothing to share): attend with our own states.
        source_empty = not self.source_layer.is_initialized or (
            self.source_layer.keys if self.share_keys else self.source_layer.values
        ).numel() == 0
        device = key_states.device

        if self.share_values:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            if source_empty:
                return key_states, value_states
            shared, own, evicted_own = self.source_layer.values, value_states, 2
        else:
            if source_empty:
                return key_states, value_states
            self.values = torch.cat([self.values, value_states], dim=-2)
            shared, own, evicted_own = self.source_layer.keys, key_states, 1

        # Fold the new tokens into the shared cache (the owner already attended with it).
        self._write_merger_in_place(shared, own, new_n, start_pos)
        # Fold in what just left the critical window. Evicted positions are older
        # than the newest n, so this never overlaps the write above.
        if evicted is not None:
            self._merge_evicted_into_source(shared, evicted[evicted_own], evicted[0])
        # _attention_states_v_merge returns our own tensor on its first call and a
        # new torch.cat after, so read the flag before calling it.
        merged_is_new = self._prefill_consumed
        merged = self._attention_states_v_merge(shared, own, new_n, device)

        if self.share_values:
            return self._splice_with_exclusions(
                self.keys, merged, keys_are_own_storage=True, values_owned=merged_is_new,
            )
        return self._splice_with_exclusions(merged, self.values)

    def get_seq_length(self) -> int:
        if self.share_keys:
            return self.source_layer.get_seq_length()
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.source_layer.get_mask_sizes(cache_position)

    def get_max_cache_shape(self) -> int:
        return self.source_layer.get_max_cache_shape()


class SharedAdjacentCache(Cache):
    """
    A cache where the layers of each merge group share one value cache
    (share_values_only, VFold) or one key cache (share_keys_only, demo only).

    sink_tokens / critical_tokens: see SharedAdjacentLayer.

    merge_layer_groups: lists of consecutive layers, e.g. [[0,1,2,3], [4,5,6,7]].
        The first layer of each group owns the shared cache. It must be the lowest,
        because it runs first. merge_layer_pairs is the same thing as a list of
        (owner, member) pairs. With neither, all adjacent pairs (0,1), (2,3), ...
        are merged. Layers in no group keep their own cache.

    layer_classes: optional (owner_cls, member_cls) replacing the default layer
        classes, so another method can run on top of the merge. See
        composition/think_vmerge/ and composition/kivi_vmerge/.
    """

    def __init__(
        self,
        config,
        num_layers: int,
        share_values_only: bool = False,
        share_keys_only: bool = False,
        sink_tokens: int = 0,
        critical_tokens: int = 0,
        merge_layer_pairs: list[tuple[int, int]] | None = None,
        merge_layer_groups: list[list[int]] | None = None,
        layer_classes: tuple[type, type] | None = None,
    ):
        if share_values_only == share_keys_only:
            raise ValueError("set exactly one of share_values_only and share_keys_only")

        source_layer_cls = DynamicLayer
        follower_layer_cls = SharedAdjacentLayer
        if layer_classes is not None:
            source_layer_cls, follower_layer_cls = layer_classes

        groups = resolve_groups(num_layers, merge_layer_groups, merge_layer_pairs)

        # Map each member to its group's owner, and record its position in the group.
        copy_to_original: dict[int, int] = {}
        group_index: dict[int, int] = {}
        for group in groups:
            owner = group[0]
            for k, member in enumerate(group[1:], start=1):
                copy_to_original[member] = owner
                group_index[member] = k

        layers = []
        for i in range(num_layers):
            if i in copy_to_original:
                layers.append(
                    follower_layer_cls(
                        layers[copy_to_original[i]],
                        share_keys=share_keys_only,
                        share_values=share_values_only,
                        sink_tokens=sink_tokens,
                        critical_tokens=critical_tokens,
                        group_index=group_index[i],
                    )
                )
            else:
                layers.append(source_layer_cls())
        super().__init__(layers=layers)
        self.merge_layer_groups = groups

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range for cache with {len(self.layers)} layers"
            )
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

_STANDARD_MODEL_TYPES = {"llama", "qwen3", "mistral"}


def custom_cache_creator(
    model,
    post_init: Callable[[Cache], None] | None = None,
    **cache_kwargs,
) -> Callable[[], Cache]:
    """Return a function that builds a new SharedAdjacentCache for this model."""
    # get the decoder model config
    config = get_text_config(model)
    model_type = getattr(config, "model_type", None)
    if model_type in _STANDARD_MODEL_TYPES:
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = len(get_layers(model))

        def _make_cache():
            cache = SharedAdjacentCache(config, num_layers, **cache_kwargs)
            # A fresh cache is built per generation, so anything that has to be
            # wired to the model's modules has to happen here rather than once
            # at patch time.
            if post_init is not None:
                post_init(cache)
            return cache

        return _make_cache
    else:
        raise ValueError(
            f"Shared adjacent cache not supported for model type {model_type}"
        )


'''
Check if we should create a new cache (start of sequence).
'''
def _cache_position_check(cache_position, past_key_values) -> bool:
    if past_key_values is None:
        return True
    if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
        return True
    # This checks if 'cache_position' is a tensor that contains the value 0, 
    # which typically indicates the start of a new sequence (for example, the first position in a fresh batch for generation/prefill).
    # If so, it returns True to signal that the cache should be (re-)initialized.
    if cache_position is not None and isinstance(cache_position, torch.Tensor):
        return 0 in cache_position
    return False

def _patch_standard_model(model, post_init=None, **cache_kwargs) -> None:
    """Make the model build a fresh SharedAdjacentCache at the start of each sequence."""
    cache_creator = custom_cache_creator(model, post_init=post_init, **cache_kwargs)

    # patch the decoder, get the correct entity for Mistral3Model which is vlm
    inner_model = get_text_model(model)
    original_forward = inner_model.forward

    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        **kwargs,
    ):
        if use_cache is None:
            use_cache = inner_model.config.use_cache
        if use_cache and _cache_position_check(cache_position, past_key_values):
            past_key_values = cache_creator()

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

    # patch the decoder forward pass
    inner_model.forward = patched_forward


def patch_model_to_use_shared_adjacent_cache(
    model: AutoModelForCausalLM,
    share_values_only: bool = False,
    share_keys_only: bool = False,
    sink_tokens: int = 0,
    critical_tokens: int = 0,
    merge_layer_pairs: list[tuple[int, int]] | None = None,
    merge_layer_groups: list[list[int]] | None = None,
    layer_classes: tuple[type, type] | None = None,
    post_init: Callable[[Cache], None] | None = None,
) -> None:
    """
    Patch a model to use SharedAdjacentCache. The cache options are described
    there. 
    """
    mt = getattr(get_text_config(model), "model_type", None)
    if mt in _STANDARD_MODEL_TYPES:
        _patch_standard_model(
            model,
            post_init=post_init,
            share_values_only=share_values_only,
            share_keys_only=share_keys_only,
            sink_tokens=sink_tokens,
            critical_tokens=critical_tokens,
            merge_layer_pairs=merge_layer_pairs,
            merge_layer_groups=merge_layer_groups,
            layer_classes=layer_classes,
        )
    else:
        raise ValueError(
            f"Shared adjacent cache not implemented for model_type={mt}. "
            f"Supported types: {_STANDARD_MODEL_TYPES}"
        )
