__all__ = ["sample_entropy"]
import torch


def sample_entropy(
    x: torch.Tensor, m: int = 2, r: float = None, stride: int = 1, subsample: int = 1
):
    """Returns the (batched) sample entropy of the given time series.

    Sample entropy is a measure of complexity of sequences that can be related
    to predictability. Sample entropy (SE) is defined as the negative logarithm of
    the following ratio:
        SE(X,m,r) = -ln(C(X, m+1, r) / C(X, m, r))
    where C(X,m,r) is the number of partial vectors of length m in sequence X whose
    Chebyshev distance is less than r.
    Note, `0 <= SE >= -ln(2/[(T-m-1)(T-m)])`, where T is the sequence length

    Based on
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy.

    Args:
        x: (B,T) tensor of batched time-series
        m: embedding length
        r: distance threshold, if None then will be computed as `0.2std(x)`
        stride: step between embedding vectors
        subsample: reduce the number of possible vectors of length m

    Returns:
        se: (B,) tensor containing sample entropy for each sequence
    """
    x = torch.atleast_2d(x)
    B, T = x.shape
    if r is None:
        r = torch.std(x) * 0.2

    def _num_close(elen: int):
        unf = x.unfold(1, elen, stride)  # B,N,elen
        if subsample > 1:
            unf = unf[:, ::subsample, :]
        N = unf.shape[1]
        d = torch.cdist(unf, unf, p=float("inf"))  # B,N,N
        idx = torch.triu_indices(N, N, 1)  # take pairwise distances excl. diagonal
        C = (d[:, idx[0], idx[1]] < r).sum(-1)  # B
        return C

    A = _num_close(m + 1)
    B = _num_close(m)

    # Exceptional handling, return upper bound. No regularities found
    mask = torch.logical_or(A == 0, B == 0)
    A[mask] = 2.0
    B[mask] = (T - m - 1) * (T - m)

    return -torch.log(A / B)
