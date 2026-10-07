import torch
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment

from vfold.model_utils import get_layers, get_text_config
from vfold.attn_head_align.general_lib import resolve_groups, select_group_reference


def do_alignment(
    activations_dict,  # {layer_idx: {"v_proj_output": [num_toks, kv_dim]}}
    num_kv_heads: int,
    head_dim: int,
    verbose: bool = False,
    merge_layer_groups: list[list[int]] | None = None,  # None -> adjacent pairs
) -> dict[int, tuple[torch.Tensor, torch.Tensor, int]]:
    """Fit value alignment maps for every merge group.
    Each group shares one value cache, so every member is mapped into the frame of
    one reference member. The reference keeps its weights and gets no entry.
    Returns {layer_idx: (v_map, head_perm, ref_idx)}.
    """
    n_layers = len(activations_dict)
    alignment_matrices = {}

    # get list of groups
    groups = resolve_groups(n_layers, merge_layer_groups, None)
    for group in groups:
        for member in group:
            if member not in activations_dict:
                raise ValueError(f"Missing activations for layer {member} (group {group})")

        # get the reference layer for this group, depends on how big group is
        ref_idx = select_group_reference(group)
        for layer_idx in group:
            if layer_idx == ref_idx:
                continue  # the reference doesnt need a map

            # otherwise, fit a map for this layer to the reference, using hierarchical matching for heads
            v_map, head_perm = match_values_hierarchical(
                activations_dict[ref_idx],
                activations_dict[layer_idx],
                num_kv_heads,
                head_dim,
                verbose=verbose,
            )
            print(f"got value map for layer {layer_idx} (reference {ref_idx})")
            alignment_matrices[layer_idx] = (v_map, head_perm, ref_idx)

    return alignment_matrices


def match_values_hierarchical(
    reference_activations: Dict[str, torch.Tensor],
    current_activations: Dict[str, torch.Tensor],
    num_kv_heads: int,
    head_dim: int,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match KV heads between two layers and fit a CCA map for each matched pair.

    Stage 1: for every head pair (i in the reference layer, j in this layer), fit a
             CCA map and record how well it aligns the two heads' values.
    Stage 2: pick the one-to-one head pairing with the lowest total residual,
             using the Hungarian algorithm.

    Returns V_rot, a [kv_dim, kv_dim] matrix with one head_dim x head_dim block per
    matched head pair, and head_perm, the head pairing as a permutation matrix.
    """

    # Reshape the activations to [num_toks, num_kv_heads * head_dim]
    v_ref = reference_activations["v_proj_output"].reshape(-1, num_kv_heads * head_dim).float() 
    v_current = current_activations["v_proj_output"].reshape(-1, num_kv_heads * head_dim).float() 

    # Drop tokens where either layer has non-finite activations (can occur with fp16 overflow)
    valid = torch.isfinite(v_ref).all(dim=1) & torch.isfinite(v_current).all(dim=1)
    if not valid.all():
        v_ref = v_ref[valid]
        v_current = v_current[valid]
    
    # make cost matrix across heads
    kv_dim = num_kv_heads * head_dim
    cost_matrix = torch.zeros(num_kv_heads, num_kv_heads)
    # for storing maps
    Rs: Dict[Tuple[int, int], torch.Tensor] = {}

    # then, go through each head pair, fit a map, write down the map and the cost
    for i in range(num_kv_heads):
        vr = v_ref[:, i * head_dim:(i + 1) * head_dim].float() #[num_toks, head_dim]
        for j in range(num_kv_heads):
            vc = v_current[:, j * head_dim:(j + 1) * head_dim].float()
            T, T_inv_T = match_algorithm_cca(vr, vc, verbose=verbose)
            Rs[(i, j)] = T_inv_T.float()
            cost_matrix[i, j] = (vr @ T.float() - vc).norm().item()

    # Stage 2: Hungarian head matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    head_perm = rotate_heads(row_ind, col_ind, head_dim, num_kv_heads)

    if verbose:
        print("Head alignment cost matrix (residual):")
        print(cost_matrix.numpy().round(2))
        for ref_h, cur_h in zip(row_ind, col_ind):
            print(f"  reference head {ref_h} → current head {cur_h}  "
                    f"(cost: {cost_matrix[ref_h, cur_h]:.4f})")
    # build the full map
    # by composing the permutation and the maps
    # loop through the hungarian assignment, and write down the map for each pair
    # this will be applied as V_rot @ W_v later. 
    V_rot = torch.zeros(kv_dim, kv_dim)
    for ref_h, cur_h in zip(row_ind, col_ind):
        V_rot[ref_h * head_dim:(ref_h + 1) * head_dim,
                cur_h * head_dim:(cur_h + 1) * head_dim] = Rs[(ref_h, cur_h)]

    return V_rot, head_perm


def match_algorithm_cca(
    data_1: torch.Tensor,  # [num_tokens, head_dim]
    data_2: torch.Tensor,  # [num_tokens, head_dim]
    reg: float = 1e-4,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CCA map between two heads' values.

    Args:
        data_1: [num_tokens, head_dim], reference activations.
        data_2: [num_tokens, head_dim], current activations.
        reg: Regularization for covariances.
        verbose: Print CCA details.

    Returns:
        T: [head_dim, head_dim] map from reference's frame to current's.
        T_inv_T: [head_dim, head_dim], T^{-T}, computed analytically.
    """
    orig_dtype = data_1.dtype
    S11, S22, S12 = cca_covariances(data_1, data_2)
    T, T_inv_T = match_algorithm_cca_from_covariance(S11, S22, S12, reg=reg, verbose=verbose)
    return T.to(orig_dtype), T_inv_T.to(orig_dtype)


def cca_covariances(
    data_1: torch.Tensor,  # [num_tokens, head_dim]
    data_2: torch.Tensor,  # [num_tokens, head_dim]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Covariances for CCA: Sigma_11, Sigma_22, Sigma_12, in float64.
    Regularization is added later, in match_algorithm_cca_from_covariance.
    """
    data_1 = data_1.double()
    data_2 = data_2.double()
    n = data_1.shape[0]
    return data_1.T @ data_1 / n, data_2.T @ data_2 / n, data_1.T @ data_2 / n


def match_algorithm_cca_from_covariance(
    Sigma_11: torch.Tensor,  # [head_dim, head_dim], unregularized
    Sigma_22: torch.Tensor,
    Sigma_12: torch.Tensor,
    reg: float = 1e-4,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CCA map from covariances (paper, Appendix B).
    Adds reg * I to Sigma_11 and Sigma_22, then returns T and T^-T. T^-T is
    computed in closed form, so T is never inverted here.
    """
    Sigma_11 = Sigma_11.double()
    Sigma_22 = Sigma_22.double()
    Sigma_12 = Sigma_12.double()
    d = Sigma_11.shape[0]
    eye = torch.eye(d, device=Sigma_11.device, dtype=Sigma_11.dtype)
    Sigma_11 = Sigma_11 + reg * eye
    Sigma_22 = Sigma_22 + reg * eye

    def _inv_sqrt(M: torch.Tensor) -> torch.Tensor:
        """Symmetric matrix inverse square root via eigendecomposition."""
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals = torch.nan_to_num(eigvals, nan=reg).clamp(min=reg)
        return eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T

    def _sqrt(M: torch.Tensor) -> torch.Tensor:
        """Symmetric matrix square root via eigendecomposition."""
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals = eigvals.clamp(min=reg)
        return eigvecs @ torch.diag(eigvals.pow(0.5)) @ eigvecs.T

    S11_inv_sqrt = _inv_sqrt(Sigma_11)
    S11_sqrt     = _sqrt(Sigma_11)
    S22_inv_sqrt = _inv_sqrt(Sigma_22)
    S22_sqrt     = _sqrt(Sigma_22)

    # Whitened cross-covariance  M = Σ_11^{-1/2} Σ_12 Σ_22^{-1/2}
    M = S11_inv_sqrt @ Sigma_12 @ S22_inv_sqrt  # [d, d]

    if verbose:
        print(f"  CCA M condition number: {torch.linalg.cond(M).item():.2e}")

    U, S_vals, Vh = torch.linalg.svd(M, full_matrices=False)

    if verbose:
        print(f"  CCA canonical correlations (top 5): {S_vals[:5].tolist()}")

    # T = Σ_11^{-1/2} U Vh Σ_22^{1/2}
    # T^{-T} = Σ_11^{1/2} U Vh Σ_22^{-1/2}  (analytical, avoids direct matrix inversion)
    T = S11_inv_sqrt @ U @ Vh @ S22_sqrt
    T_inv_T = S11_sqrt @ U @ Vh @ S22_inv_sqrt
    return T, T_inv_T


def rotate_heads(row_ind, col_ind, head_dim, num_kv_heads):
    """Permutation matrix that pairs head row_ind[i] with head col_ind[i]."""
    kv_dim = num_kv_heads * head_dim
    rot = torch.zeros((kv_dim, kv_dim))
    for i in range(num_kv_heads):
        row_start = row_ind[i] * head_dim
        row_end = row_start + head_dim
        col_start = col_ind[i] * head_dim
        col_end = col_start + head_dim
        rot[row_start:row_end, col_start:col_end] = torch.eye(head_dim)
    return rot


def expand_head_perm(block_matrix, num_kv_heads, head_dim, num_key_value_groups):
    """
    Expand a block matrix from KV-head space to full attention-head space,
    preserving off-diagonal (head-permutation) structure.

    For queries/o_proj (num_key_value_groups>1), each KV-head block (i, j)
    is tiled into G copies placed at positions (i*G+g, j*G+g) for g in 0..G-1.

    Args:
        block_matrix: [kv_dim, kv_dim] block matrix (e.g. V_rot or head_perm)
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head
        num_key_value_groups: Number of query heads per KV head (GQA groups)

    Returns:
        Expanded matrix of shape
        [num_kv_heads * num_key_value_groups * head_dim, ...]
    """
    if num_key_value_groups == 1:
        return block_matrix

    full_dim = num_kv_heads * num_key_value_groups * head_dim
    device = block_matrix.device
    dtype = block_matrix.dtype
    expanded = torch.zeros(full_dim, full_dim, device=device, dtype=dtype)

    for i in range(num_kv_heads):
        for j in range(num_kv_heads):
            candidate = block_matrix[i * head_dim:(i + 1) * head_dim, j * head_dim:(j + 1) * head_dim]
            if candidate.abs().max() > 1e-6:
                for g in range(num_key_value_groups):
                    r = (i * num_key_value_groups + g) * head_dim
                    c = (j * num_key_value_groups + g) * head_dim
                    expanded[r:r + head_dim, c:c + head_dim] = candidate

    return expanded




def apply_v_alignment(model, alignment, verbose=False):
    """Fold the value maps into the weights without changing the model's outputs.

    The map goes into v_proj and its inverse into o_proj. The head pairing also
    permutes the KV heads, so the same permutation is applied to k_proj and q_proj
    to keep each query group attending with its own values (paper, Appendix A).

    Args:
        model: a Llama/Mistral/Qwen model with grouped-query attention.
        alignment: {layer_idx: (v_map, head_perm, ref_idx)} from do_alignment.
        verbose: print extra info for debugging.
    """
    config = get_text_config(model)
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_key_value_groups = num_attention_heads // num_key_value_heads

    with torch.no_grad():
        for layer_idx in alignment:
            # Get the modules for the layer, and their weights
            v_proj_weight = get_layers(model)[layer_idx].self_attn.v_proj.weight
            k_proj_weight = get_layers(model)[layer_idx].self_attn.k_proj.weight
            q_proj_weight = get_layers(model)[layer_idx].self_attn.q_proj.weight
            o_proj_weight = get_layers(model)[layer_idx].self_attn.o_proj.weight

            v_proj = get_layers(model)[layer_idx].self_attn.v_proj
            o_proj = get_layers(model)[layer_idx].self_attn.o_proj
            k_proj = get_layers(model)[layer_idx].self_attn.k_proj
            q_proj = get_layers(model)[layer_idx].self_attn.q_proj

            device = v_proj_weight.device
            dtype = v_proj_weight.dtype

            V_rot = alignment[layer_idx][0].to(device=device)
            head_perm = alignment[layer_idx][1].to(device=device)

            # Permute the KV heads in k_proj and q_proj to match the head pairing.
            k_proj.weight.data = (head_perm.float() @ k_proj_weight.float()).to(dtype)
            q_head_perm = expand_head_perm(head_perm, num_key_value_heads, head_dim, num_key_value_groups)
            q_proj.weight.data = (q_head_perm.float() @ q_proj_weight.float()).to(dtype)

            # Apply the value map to v_proj.
            v_proj.weight.data = (V_rot.float() @ v_proj_weight.float()).to(dtype)
            if verbose:
                print(f"Layer {layer_idx}: v_proj.weight {v_proj_weight.shape}")

            # Undo the map in o_proj, expanded from KV heads to all query heads.
            R_eff = expand_head_perm(V_rot, num_key_value_heads, head_dim, num_key_value_groups)
            try:
                R_eff_inv = torch.linalg.inv(R_eff.float())
            except torch._C._LinAlgError:
                print(f"  Warning: R_eff singular for layer {layer_idx}, using pseudoinverse")
                R_eff_inv = torch.linalg.pinv(R_eff.float())
            o_proj.weight.data = (o_proj_weight.float() @ R_eff_inv).to(dtype).contiguous()

    return model
