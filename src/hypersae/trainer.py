import torch
import torch.optim as optim
from hypersae.models import HyperSAE
from hypersae.queue import CoActivationQueue
from hypersae.loss import TriPartiteLoss

class HyperSAETrainer:
    def __init__(
        self,
        model: HyperSAE,
        queue: CoActivationQueue,
        loss_fn: TriPartiteLoss,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        num_pairs: int = 128
    ):
        """
        Coordinates training steps across models, rolling buffers, and multi-space loss functions.
        
        Args:
            model: An instance of HyperSAE
            queue: An instance of CoActivationQueue
            loss_fn: An instance of TriPartiteLoss
            lr: Learning rate
            weight_decay: Optimizer weight decay
            num_pairs: Number of positive and negative co-activation pairs to sample per step
        """
        self.model = model
        self.queue = queue
        self.loss_fn = loss_fn
        self.num_pairs = num_pairs
        
        # Configure standard AdamW optimizer for all Euclidean parameters (W_enc, b_enc, W_dec, r)
        # Spherical constraints on W_dec are enforced post-step via projection
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_step(self, x: torch.Tensor):
        """
        Executes a single coordinated training step on a batch of activations.
        
        Args:
            x: Input activation stream tensor, shape [..., d_model]
            
        Returns:
            metrics: Dict containing loss component floats for tracking
        """
        self.optimizer.zero_grad()
        
        # 1. Forward Pass (Euclidean fast-path)
        x_hat, f = self.model(x)
        
        # 2. Update Queue buffer with current batch activations (Detached automatically)
        self.queue.update(f)
        
        # 3. Sample co-activation pairs based on current learned depths
        pos_p, pos_c, neg_p, neg_c = self.queue.sample_pairs(
            num_pairs=self.num_pairs,
            depths=self.model.r
        )
        
        # 4. Project coordinates to Poincare Ball (float32 slow-path)
        poincare_coords = self.model.get_poincare_coordinates(c=self.loss_fn.c)
        
        # 5. Compute combined losses
        loss, metrics = self.loss_fn(
            x=x,
            x_hat=x_hat,
            f=f,
            poincare_coords=poincare_coords,
            pos_parents=pos_p,
            pos_children=pos_c,
            neg_parents=neg_p,
            neg_children=neg_c
        )
        
        # 6. Backpropagation
        loss.backward()
        self.optimizer.step()
        
        # 7. Enforce unit-norm constraints on feature vectors
        self.model.enforce_unit_norm()
        
        return metrics
