import math
import torch
import torch.nn as nn
import torch.nn.init as init
from hypersae.utils.geometry import project_to_poincare_ball

class HyperSAE(nn.Module):
    def __init__(self, d_model: int, dict_size: int):
        """
        Hyperbolic Sparse Autoencoder (HypSAE).
        
        Args:
            d_model: Dimensionality of the input (e.g. residual stream dimension).
            dict_size: Number of features in the dictionary (overcomplete dimension).
        """
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        
        # Encoder parameters
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        
        # Decoder parameters (rows represent features w_i)
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model))
        
        # Learnable radial depths (initialized to small positive constant)
        self.r = nn.Parameter(torch.empty(dict_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initializes weights using standard methods."""
        # Initialize encoder and decoder weights
        init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        
        # Initialize radial depths to small positive constant (root-ward initialization)
        init.constant_(self.r, 0.1)
        
        # Enforce unit norm constraint on decoder at initialization
        self.enforce_unit_norm()
        
    @torch.no_grad()
    def enforce_unit_norm(self):
        """
        Enforces unit-norm constraint on the decoder feature directions: ||w_i||_2 = 1.
        This must be called at the end of every optimization step to prevent weight inflation.
        """
        norms = torch.norm(self.W_dec, p=2, dim=-1, keepdim=True)
        self.W_dec.copy_(self.W_dec / torch.clamp(norms, min=1e-8))
        
    def forward(self, x: torch.Tensor):
        """
        Euclidean forward pass (Flat, fast-path).
        
        Args:
            x: Input activation stream of shape [..., d_model]
            
        Returns:
            x_hat: Reconstructed activation stream of shape [..., d_model]
            f: Sparse activations of shape [..., dict_size]
        """
        # Sparse encoding: f = ReLU(x @ W_enc^T + b_enc)
        f = torch.relu(x @ self.W_enc.t() + self.b_enc)
        
        # Reconstruct: x_hat = f @ W_dec
        x_hat = f @ self.W_dec
        
        return x_hat, f

    def get_poincare_coordinates(self, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """
        Computes the current projected Poincaré ball coordinates of all dictionary features.
        Decoupled compute: Upcasts internally to float32 for safety.
        
        Args:
            c: Manifold curvature parameter
            eps: Numerical boundary safety parameter
            
        Returns:
            h: Hyperbolic coordinates of shape [dict_size, d_model]
        """
        # Ensure r is broadcastable (dict_size, 1) during projection
        r_col = self.r.unsqueeze(-1)
        return project_to_poincare_ball(self.W_dec, r_col, c=c, eps=eps)

    def get_steering_vector(self, feature_id: int) -> torch.Tensor:
        """
        Extracts the flat, unit-norm steering vector for a target feature.
        Allows zero-friction manifold steering directly in Euclidean space during inference.
        
        Args:
            feature_id: The index of the dictionary feature
            
        Returns:
            w: Steering direction vector of shape [d_model]
        """
        return self.W_dec[feature_id]

    def export_ontology_graph(
        self,
        K: float = 0.5,
        c: float = 1.0,
        eps: float = 1e-5,
        tolerance: float = 0.0
    ):
        """
        Vectorized extraction of feature taxonomy.
        Constructs a NetworkX DAG of feature relationships.
        
        Args:
            K: Cone aperture tuning parameter (0 < K < 1)
            c: Poincaré ball curvature
            eps: Epsilon parameter for origin checks
            tolerance: Allowed violation threshold
            
        Returns:
            TR: Transitive-reduced NetworkX DiGraph representing the ontology
        """
        from hypersae.utils.ontology import extract_ontology_graph
        return extract_ontology_graph(self, K=K, c=c, eps=eps, tolerance=tolerance)


class FlatSAE(nn.Module):
    """
    Standard Flat (Euclidean) Sparse Autoencoder baseline.
    Used for comparative benchmarks against HyperSAE.
    """
    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self.enforce_unit_norm()

    @torch.no_grad()
    def enforce_unit_norm(self):
        norms = torch.norm(self.W_dec, p=2, dim=-1, keepdim=True)
        self.W_dec.copy_(self.W_dec / torch.clamp(norms, min=1e-8))

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        f = torch.relu(x_cent @ self.W_enc.t() + self.b_enc)
        x_hat = f @ self.W_dec + self.b_dec
        return x_hat, f

