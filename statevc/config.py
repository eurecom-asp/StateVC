from dataclasses import dataclass


@dataclass(frozen=True)
class StateVCConfig:
    """Configuration matching the final ICASSP StateVC setup."""

    cluster_dim: int = 24
    k_max: int = 4
    n_min: int = 20
    m_min: float = 20.0
    smooth_window: int = 5
    strength: float = 1.0
    operator: str = "mean"
    routing: str = "soft"
    block_size: int = 2

    # Pair-specific diagonal GMM settings used in the final shared-GMM path.
    gmm_seed: int = 0
    gmm_n_init: int = 3
    gmm_max_iter: int = 100
    gmm_reg_covar: float = 1e-4
    gmm_init_params: str = "kmeans"

    # Numerical floor used by the operator ablations. The main Mean system does
    # not depend on this value.
    eps: float = 1e-4

    def validate(self) -> None:
        if self.cluster_dim < 1:
            raise ValueError("cluster_dim must be >= 1")
        if self.k_max < 1:
            raise ValueError("k_max must be >= 1")
        if self.n_min < 1:
            raise ValueError("n_min must be >= 1")
        if self.m_min < 0:
            raise ValueError("m_min must be >= 0")
        if self.smooth_window < 1 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if self.operator not in {"mean", "diagonal", "block", "none"}:
            raise ValueError("operator must be one of: mean, diagonal, block, none")
        if self.routing not in {"soft", "hard"}:
            raise ValueError("routing must be one of: soft, hard")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
