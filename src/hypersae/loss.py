import torch
import torch.nn as nn
from hypersae.utils.geometry import hyperbolic_entailment_cone_penalty

class TriPartiteLoss(nn.Module):
    def __init__(
        self,
        l1_coeff: float = 1e-3,
        entail_coeff: float = 1e-2,
        c: float = 1.0,
        K: float = 0.5,
        gamma: float = 0.1
    ):
        """
        Orchestrates training penalties across Euclidean and hyperbolic spaces.
        
        Args:
            l1_coeff: Sparsity multiplier (L1 penalty)
            entail_coeff: Hyperbolic entailment loss multiplier
            c: Poincaré ball curvature
            K: Hyperbolic cone aperture parameter
            gamma: Negative pair margin (prevents zero-energy collapse)
        """
        super().__init__()
        self.l1_coeff = l1_coeff
        self.entail_coeff = entail_coeff
        self.c = c
        self.K = K
        self.gamma = gamma

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        f: torch.Tensor,
        poincare_coords: torch.Tensor,
        pos_parents: torch.Tensor,
        pos_children: torch.Tensor,
        neg_parents: torch.Tensor,
        neg_children: torch.Tensor
    ):
        """
        Computes the composite loss: Recon MSE + Sparsity L1 + Hyperbolic Entailment.
        
        Args:
            x: Input activation stream, shape [..., d_model]
            x_hat: Reconstructed activation stream, shape [..., d_model]
            f: Sparse coding activations, shape [..., dict_size]
            poincare_coords: Features mapped to Poincare ball, shape [dict_size, d_model]
            pos_parents: Positive parent indices, shape [num_pairs]
            pos_children: Positive child indices, shape [num_pairs]
            neg_parents: Negative parent indices, shape [num_pairs]
            neg_children: Negative child indices, shape [num_pairs]
            
        Returns:
            total_loss: Differentiable combined loss tensor
            metrics: Dict containing individual component losses (floats) for tracking
        """
        # 1. Reconstruction Loss (Euclidean MSE)
        recon_loss = torch.mean((x - x_hat) ** 2)
        
        # 2. Sparsity Loss (L1 norm of feature activations per token)
        sparsity_loss = torch.mean(torch.sum(f, dim=-1))
        
        # 3. Hyperbolic Entailment Loss (Cone constraints in Poincare space)
        # Gather parent/child feature coordinates
        pos_p_coords = poincare_coords[pos_parents]
        pos_c_coords = poincare_coords[pos_children]
        neg_p_coords = poincare_coords[neg_parents]
        neg_c_coords = poincare_coords[neg_children]
        
        # Positive violations: minimize when child resides inside parent's cone
        pos_violations = hyperbolic_entailment_cone_penalty(
            pos_p_coords, pos_c_coords, c=self.c, K=self.K
        )
        
        # Negative violations: push unrelated concepts apart by at least margin gamma
        neg_violations = hyperbolic_entailment_cone_penalty(
            neg_p_coords, neg_c_coords, c=self.c, K=self.K
        )
        
        loss_pos = torch.mean(pos_violations)
        # Hinge loss penalty: penalize if negative violation is smaller than margin
        loss_neg = torch.mean(torch.relu(self.gamma - neg_violations))
        
        entail_loss = loss_pos + loss_neg
        
        # Combined objective
        total_loss = (
            recon_loss 
            + self.l1_coeff * sparsity_loss 
            + self.entail_coeff * entail_loss
        )
        
        metrics = {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_sparsity": sparsity_loss.item(),
            "loss_entail": entail_loss.item(),
            "loss_entail_pos": loss_pos.item(),
            "loss_entail_neg": loss_neg.item(),
        }
        
        return total_loss, metrics
