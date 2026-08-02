import math
import torch
import geoopt

def project_to_poincare_ball(
    w: torch.Tensor,
    r: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Projects flat Euclidean concept vectors (w) and their learned radial depths (r)
    into the Poincaré ball of curvature c.
    
    Args:
        w: Unit-norm direction vectors, shape [..., D]
        r: Hierarchical depth scalars, shape [..., 1] or [...] (broadcastable)
        c: Curvature parameter of the Poincaré ball
        eps: Epsilon parameter for numerical boundary safety clipping
        
    Returns:
        h: Poincaré coordinates of the features, shape [..., D]
    """
    # Enforce float32 compute for numerical stability in hyperbolic space
    w_f32 = w.to(torch.float32)
    r_f32 = r.to(torch.float32)
    
    # Ensure r matches the last dimension of w
    if r_f32.dim() < w_f32.dim():
        r_f32 = r_f32.unsqueeze(-1)
        
    # Standard normalization of w just in case it drifts slightly from unit-norm
    w_norm = torch.norm(w_f32, p=2, dim=-1, keepdim=True)
    w_unit = w_f32 / torch.clamp(w_norm, min=1e-8)
    
    # Poincaré projection at the origin: exp_o^c(r * w)
    # Norm of projected vector is: tanh(sqrt(c) * r) / sqrt(c)
    sqrt_c = math.sqrt(c)
    norm_factor = torch.tanh(sqrt_c * r_f32) / sqrt_c
    h = norm_factor * w_unit
    
    # Clip coordinates to be strictly within the boundary (1 - eps) / sqrt(c)
    manifold = geoopt.manifolds.PoincareBall(c=c)
    h_proj = manifold.projx(h)
    
    # Additional manual clip for maximum boundary protection
    max_norm = (1.0 - eps) / sqrt_c
    h_proj_norm = torch.norm(h_proj, p=2, dim=-1, keepdim=True)
    scale = torch.where(
        h_proj_norm > max_norm,
        max_norm / torch.clamp(h_proj_norm, min=1e-8),
        torch.ones_like(h_proj_norm)
    )
    return h_proj * scale

def hyperbolic_geodesic_distance(
    u: torch.Tensor,
    v: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Computes the exact geodesic distance between points u and v in the Poincaré ball.
    
    Args:
        u: First set of coordinates, shape [..., D]
        v: Second set of coordinates, shape [..., D]
        c: Curvature parameter
        
    Returns:
        dist: Geodesic distance tensor
    """
    # Enforce float32 computation
    u_f32 = u.to(torch.float32)
    v_f32 = v.to(torch.float32)
    
    manifold = geoopt.manifolds.PoincareBall(c=c)
    return manifold.dist(u_f32, v_f32, dim=-1)

def hyperbolic_entailment_cone_penalty(
    parent: torch.Tensor,
    child: torch.Tensor,
    c: float = 1.0,
    K: float = 0.5,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Calculates the asymmetric containment cone penalty between a parent feature and a child feature.
    Penalizes if the child feature falls outside the parent's entailment cone.
    
    Args:
        parent: Parent coordinate tensor, shape [..., D]
        child: Child coordinate tensor, shape [..., D]
        c: Curvature parameter of the Poincaré ball
        K: Aperture tuning hyperparameter, 0 < K < 1 (typically ~0.5)
        eps: Epsilon for numerical stability
        
    Returns:
        penalty: Hinge penalty (Relu(xi - psi))
    """
    parent_f32 = parent.to(torch.float32)
    child_f32 = child.to(torch.float32)
    
    # Norm of parent
    parent_norm_sq = torch.sum(parent_f32 * parent_f32, dim=-1, keepdim=True)
    parent_norm = torch.sqrt(parent_norm_sq)
    
    # 1. Compute half-aperture: psi = arcsin( K * (1 - c * ||parent||^2) / ||parent|| )
    aperture_arg = K * (1.0 - c * parent_norm_sq) / torch.clamp(parent_norm, min=1e-7)
    aperture_arg = torch.clamp(aperture_arg, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    psi = torch.arcsin(aperture_arg)
    
    # 2. Compute angle Xi(parent, child)
    dot_pc = torch.sum(parent_f32 * child_f32, dim=-1, keepdim=True)
    child_norm_sq = torch.sum(child_f32 * child_f32, dim=-1, keepdim=True)
    diff_norm = torch.norm(parent_f32 - child_f32, p=2, dim=-1, keepdim=True)
    
    # Numerator of cos_xi
    num = dot_pc * (1.0 + c * parent_norm_sq) - parent_norm_sq * (1.0 + c * child_norm_sq)
    
    # Denominator term: sqrt(1 + c^2 ||p||^2 ||c||^2 - 2 c <p, c>)
    denom_term = torch.sqrt(torch.clamp(1.0 + (c**2) * parent_norm_sq * child_norm_sq - 2.0 * c * dot_pc, min=1e-8))
    denom = parent_norm * diff_norm * denom_term
    
    cos_xi = num / torch.clamp(denom, min=1e-8)
    cos_xi = torch.clamp(cos_xi, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    xi = torch.arccos(cos_xi)
    
    # 3. Handle edge case: Parent is at/near the origin (root covers the entire ball, so no violation)
    is_origin = (parent_norm < eps)
    xi = torch.where(is_origin, torch.zeros_like(xi), xi)
    psi = torch.where(is_origin, torch.ones_like(psi) * (math.pi / 2.0), psi)
    
    # 4. Hinge loss penalty
    violation = torch.relu(xi - psi)
    return violation.squeeze(-1)
