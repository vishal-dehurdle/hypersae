import math
import torch
import pytest
from hypersae.utils.geometry import (
    project_to_poincare_ball,
    hyperbolic_geodesic_distance,
    hyperbolic_entailment_cone_penalty,
)

def test_project_to_poincare_ball_origin():
    # Setup directions and 0 depth
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    r = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    h = project_to_poincare_ball(w, r, c=1.0)
    assert torch.allclose(h, torch.zeros_like(h), atol=1e-6)

def test_project_to_poincare_ball_clipping():
    # Setup large depths that should approach boundary
    w = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    r = torch.tensor([1e5, 1e5], dtype=torch.float32)
    eps = 1e-5
    c = 1.0
    
    h = project_to_poincare_ball(w, r, c=c, eps=eps)
    norms = torch.norm(h, p=2, dim=-1)
    
    # Each norm should be capped at (1 - eps) / sqrt(c)
    max_expected = (1.0 - eps) / math.sqrt(c)
    for norm in norms:
        assert norm.item() <= max_expected + 1e-7
        assert norm.item() > 0.99  # Should be very close to the boundary
        assert not torch.isnan(norm)

def test_hyperbolic_geodesic_distance():
    c = 1.0
    # Test distance from a point to itself
    u = torch.tensor([[0.2, -0.3], [0.0, 0.5]], dtype=torch.float32)
    dist_self = hyperbolic_geodesic_distance(u, u, c=c)
    assert torch.allclose(dist_self, torch.zeros_like(dist_self), atol=1e-5)
    
    # Test distance between origin and a point u
    # Analytical: d(0, u) = 2/sqrt(c) * arctanh(sqrt(c) * ||u||)
    # For c=1.0, d(0, u) = 2 * arctanh(||u||)
    u_origin = torch.zeros((1, 2), dtype=torch.float32)
    u_point = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    dist = hyperbolic_geodesic_distance(u_origin, u_point, c=c)
    
    expected_dist = 2.0 * math.atanh(0.5)
    assert pytest.approx(dist.item(), abs=1e-5) == expected_dist

def test_hyperbolic_entailment_cone_penalty():
    c = 1.0
    K = 0.5
    
    # Parent near origin (root) should have 0 penalty for any child
    parent_root = torch.zeros((1, 2), dtype=torch.float32)
    child = torch.tensor([[0.8, 0.0]], dtype=torch.float32)
    penalty_root = hyperbolic_entailment_cone_penalty(parent_root, child, c=c, K=K)
    assert penalty_root.item() == 0.0
    
    # Child along the same ray (parent at 0.2, child at 0.5)
    # The child is downstream in the same direction, so it should be inside the cone (0 penalty)
    parent_aligned = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
    child_aligned = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    penalty_aligned = hyperbolic_entailment_cone_penalty(parent_aligned, child_aligned, c=c, K=K)
    assert penalty_aligned.item() == 0.0
    
    # Child in opposite direction (parent at 0.2, child at -0.5)
    # This should definitely violate the containment cone
    child_opposite = torch.tensor([[-0.5, 0.0]], dtype=torch.float32)
    penalty_opposite = hyperbolic_entailment_cone_penalty(parent_aligned, child_opposite, c=c, K=K)
    assert penalty_opposite.item() > 0.0

def test_autograd_gradients():
    c = 1.0
    w = torch.tensor([[0.6, 0.8]], dtype=torch.float32, requires_grad=True)
    r = torch.tensor([1.5], dtype=torch.float32, requires_grad=True)
    
    # 1. Check gradients for projection
    h = project_to_poincare_ball(w, r, c=c)
    loss = torch.sum(h ** 2)
    loss.backward()
    
    assert w.grad is not None
    assert r.grad is not None
    assert not torch.isnan(w.grad).any()
    assert not torch.isnan(r.grad).any()
    
    # Reset grads
    w.grad.zero_()
    r.grad.zero_()
    
    # 2. Check gradients for cone loss
    parent = project_to_poincare_ball(w, r, c=c)
    child = torch.tensor([[0.1, 0.5]], dtype=torch.float32, requires_grad=True)
    
    penalty = hyperbolic_entailment_cone_penalty(parent, child, c=c, K=0.5)
    # Add small epsilon to penalty sum to test gradients when violation is positive
    # Let's force a violation by placing the child in the opposite direction
    parent_viol = project_to_poincare_ball(w, r, c=c)
    child_viol = torch.tensor([[-0.2, -0.3]], dtype=torch.float32, requires_grad=True)
    penalty_viol = hyperbolic_entailment_cone_penalty(parent_viol, child_viol, c=c, K=0.5)
    
    penalty_viol.backward()
    assert w.grad is not None
    assert r.grad is not None
    assert child_viol.grad is not None
    assert not torch.isnan(w.grad).any()
    assert not torch.isnan(r.grad).any()
    assert not torch.isnan(child_viol.grad).any()
