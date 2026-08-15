import math
import torch
import pytest
from hypersae.models import HyperSAE
from hypersae.queue import CoActivationQueue
from hypersae.loss import TriPartiteLoss
from hypersae.trainer import HyperSAETrainer

def test_loss_forward():
    d_model = 16
    dict_size = 32
    num_pairs = 10
    
    loss_fn = TriPartiteLoss(l1_coeff=1e-3, entail_coeff=1e-2)
    
    x = torch.randn(5, d_model)
    x_hat = torch.randn(5, d_model)
    f = torch.relu(torch.randn(5, dict_size))
    poincare_coords = torch.randn(dict_size, d_model) * 0.1
    
    pos_p = torch.randint(0, dict_size, (num_pairs,))
    pos_c = torch.randint(0, dict_size, (num_pairs,))
    neg_p = torch.randint(0, dict_size, (num_pairs,))
    neg_c = torch.randint(0, dict_size, (num_pairs,))
    
    loss, metrics = loss_fn(
        x=x,
        x_hat=x_hat,
        f=f,
        poincare_coords=poincare_coords,
        pos_parents=pos_p,
        pos_children=pos_c,
        neg_parents=neg_p,
        neg_children=neg_c
    )
    
    assert loss.dim() == 0  # Should be scalar
    assert "loss_total" in metrics
    assert "loss_recon" in metrics
    assert "loss_sparsity" in metrics
    assert "loss_entail" in metrics
    assert not torch.isnan(loss)

def test_trainer_step():
    d_model = 8
    dict_size = 16
    num_pairs = 5
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    queue = CoActivationQueue(dict_size=dict_size, capacity=50)
    loss_fn = TriPartiteLoss(l1_coeff=1e-3, entail_coeff=1e-2)
    
    trainer = HyperSAETrainer(
        model=model,
        queue=queue,
        loss_fn=loss_fn,
        num_pairs=num_pairs,
        lr=1e-3
    )
    
    x = torch.randn(10, d_model)
    metrics = trainer.train_step(x)
    
    assert "loss_total" in metrics
    # Assert queue indices moved forward by 10 (batch size)
    assert queue.head_idx.item() == 10
    
    # Assert decoder remains unit-normalized
    norms = torch.norm(model.W_dec, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_trainer_convergence():
    d_model = 8
    dict_size = 16
    num_pairs = 10
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    queue = CoActivationQueue(dict_size=dict_size, capacity=200)
    loss_fn = TriPartiteLoss(l1_coeff=1e-3, entail_coeff=1e-2, gamma=0.1)
    
    trainer = HyperSAETrainer(
        model=model,
        queue=queue,
        loss_fn=loss_fn,
        num_pairs=num_pairs,
        lr=1e-2
    )
    
    # Generate simple synthetic data: a combination of random signals
    # We want to fill the queue first so that it doesn't fallback to random sampling
    warmup_x = torch.randn(120, d_model)
    with torch.no_grad():
        _, warmup_f = model(warmup_x)
    queue.update(warmup_f)
    
    # Track loss and depths
    initial_r = model.r.detach().clone()
    
    # Run 10 training steps
    losses = []
    for _ in range(10):
        x = torch.randn(32, d_model)
        metrics = trainer.train_step(x)
        losses.append(metrics["loss_total"])
        
    # Verify parameter update occurred on r (learned depths changed)
    assert not torch.equal(model.r, initial_r)
    # Norm constraint must still be exactly enforced
    norms = torch.norm(model.W_dec, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Ensure losses are valid numeric values
    for l in losses:
        assert not math.isnan(l)
