import torch


def pairwise_distance(a, squared=False, eps=1e-16):
    """Computes the pairwise distance matrix with numerical stability."""
    operand = a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1)
    operandT = torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    pairwise_distances_squared = torch.add(operand, operandT) - 2 * (
        torch.mm(a, torch.t(a))
    )
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * eps
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(), device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def o_metric(x_matrix, y_matrix, directional=False):
    """Computes the point-proximity O metric from
    "Analysing patterns of spatial and niche overlap among species at multiple resolutions",
    by Marcel Cardillo and Dan Warren, published in Global Ecology and BioGeography, 2016.


    Given two matrices, an `x_matrix` of M x n, and a `y_matrix`
    of P x n, both in n-dimensional space of M and P points, respectively,
    this function computes the point-proximity O metric for the two datasets.

    """
    con_x = torch.cdist(x_matrix.unsqueeze(0), x_matrix.unsqueeze(0))[0]
    if not directional:
        con_y = torch.cdist(y_matrix.unsqueeze(0), y_matrix.unsqueeze(0))[0]
    hetero = torch.cdist(x_matrix.unsqueeze(0), y_matrix.unsqueeze(0))[0]

    con_x.fill_diagonal_(torch.max(con_x))
    if not directional:
        con_y.fill_diagonal_(torch.max(con_y))

    con_x = torch.min(con_x, dim=1)[0]
    if not directional:
        con_y = torch.min(con_y, dim=1)[0]

    hetero_x = torch.min(hetero, dim=1)[0]
    if not directional:
        hetero_y = torch.min(hetero.T, dim=1)[0]

    o_x = con_x / hetero_x
    if not directional:
        o_y = con_y / hetero_y

    p_x = torch.sum((o_x > 1).float()) / o_x.shape[0]
    if not directional:
        p_y = torch.sum((o_y > 1).float()) / o_y.shape[0]

    if directional:
        o_val = p_x
    else:
        o_val = (p_x + p_y) / 2
    return o_val
