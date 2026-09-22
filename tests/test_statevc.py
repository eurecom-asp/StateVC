import numpy as np
import torch

from statevc.config import StateVCConfig
from statevc.core import convert_features
from statevc.gmm import fit_shared_gmm
from statevc.operators import gaussian_transport_map
from statevc.states import smooth_posteriors, source_variance_order


def test_smoothing_preserves_probability_rows():
    g = torch.tensor(
        [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]], dtype=torch.float64
    )
    out = smooth_posteriors(g, 5)
    assert out.shape == g.shape
    assert torch.allclose(out.sum(-1), torch.ones(g.shape[0], dtype=g.dtype), atol=1e-12)
    assert torch.all(out >= 0)


def test_source_variance_order_is_source_only():
    h = torch.tensor([[0.0, 0.0, 0.0], [10.0, 1.0, 3.0], [20.0, 2.0, 0.0]])
    order = source_variance_order(h)
    assert order[0].item() == 0


def test_bic_candidate_support_filter():
    rng = np.random.default_rng(0)
    z1 = rng.normal(-3.0, 0.2, size=(40, 2))
    z2 = rng.normal(+3.0, 0.2, size=(40, 2))
    z_s = np.concatenate([z1[:20], z2[:20]], axis=0)
    z_r = np.concatenate([z1[20:], z2[20:]], axis=0)
    cfg = StateVCConfig(cluster_dim=2, k_max=4, n_min=20)
    sel = fit_shared_gmm(z_s, z_r, cfg)
    assert 1 <= sel.k_eff <= 4
    assert np.all(sel.support >= 20)


def test_mean_identity_for_identical_sequences():
    torch.manual_seed(0)
    h = torch.randn(80, 16)
    cfg = StateVCConfig(
        cluster_dim=4,
        k_max=3,
        n_min=10,
        m_min=5,
        smooth_window=5,
        operator="mean",
        routing="soft",
        strength=1.0,
    )
    out = convert_features(h, h.clone(), cfg).features
    assert torch.allclose(out, h, atol=2e-5, rtol=2e-5)


def test_mean_global_shift_is_recovered():
    torch.manual_seed(1)
    h_s = torch.randn(100, 12)
    shift = torch.linspace(-0.5, 0.5, 12)
    h_r = h_s + shift
    cfg = StateVCConfig(
        cluster_dim=5,
        k_max=1,
        n_min=20,
        m_min=20,
        operator="mean",
        routing="soft",
        strength=1.0,
    )
    out = convert_features(h_s, h_r, cfg).features
    assert torch.allclose(out, h_r, atol=2e-5, rtol=2e-5)


def test_gaussian_map_identity():
    cov = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    a = gaussian_transport_map(cov, cov, 1e-8)
    assert torch.allclose(a, torch.eye(2, dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_all_operators_shape():
    torch.manual_seed(2)
    h_s = torch.randn(70, 10)
    h_r = torch.randn(75, 10) + 0.2
    for op in ["mean", "diagonal", "block", "none"]:
        cfg = StateVCConfig(
            cluster_dim=4,
            k_max=2,
            n_min=10,
            m_min=5,
            operator=op,
            block_size=2,
        )
        out = convert_features(h_s, h_r, cfg).features
        assert out.shape == h_s.shape
        assert torch.isfinite(out).all()
