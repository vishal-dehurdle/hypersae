import torch
import pytest
from hypersae.models import HyperSAE
from hypersae.queue import CoActivationQueue

def test_hypersae_forward():
    d_model = 64
    dict_size = 128
    batch_size = 32
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    x = torch.randn(batch_size, d_model)
    
    x_hat, f = model(x)
    
    assert x_hat.shape == (batch_size, d_model)
    assert f.shape == (batch_size, dict_size)
    assert (f >= 0.0).all()  # ReLU output constraint

def test_enforce_unit_norm():
    d_model = 32
    dict_size = 64
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    # Manually corrupt weights by multiplying them by random scaling
    scaling = torch.rand(dict_size, 1) * 10.0 + 2.0
    with torch.no_grad():
        model.W_dec.mul_(scaling)
        
    # Enforce constraint
    model.enforce_unit_norm()
    
    # Assert each row has unit-norm 1.0
    norms = torch.norm(model.W_dec, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_get_poincare_coordinates():
    d_model = 16
    dict_size = 32
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    h = model.get_poincare_coordinates(c=1.0)
    
    assert h.shape == (dict_size, d_model)
    # Norm of projected points must be strictly less than 1.0 (since c=1.0)
    norms = torch.norm(h, p=2, dim=-1)
    assert (norms < 1.0).all()

def test_coactivation_queue_update():
    dict_size = 10
    capacity = 20
    
    queue = CoActivationQueue(dict_size=dict_size, capacity=capacity)
    assert queue.head_idx.item() == 0
    assert not queue.is_full.item()
    
    # 1. Update with smaller batch than capacity
    f1 = torch.zeros(5, dict_size)
    f1[0, 1] = 1.0
    f1[2, 3] = 2.0
    queue.update(f1)
    
    assert queue.head_idx.item() == 5
    assert not queue.is_full.item()
    assert queue.buffer[0, 1] == True
    assert queue.buffer[2, 3] == True
    
    # 2. Update to exceed capacity and trigger rollover
    f2 = torch.zeros(18, dict_size)
    f2[0, 5] = 1.0
    f2[15, 5] = 1.0  # This element falls in the second part and should wrap to index 0 of the buffer
    queue.update(f2)
    
    assert queue.is_full.item()
    assert queue.head_idx.item() == 3  # (5 + 18) % 20 = 3
    # Check that new write wrapped around to indices
    assert queue.buffer[5, 5] == True  # f2[0, 5] goes to buffer[5, 5]
    assert queue.buffer[0, 5] == True  # f2[15, 5] wraps around to buffer[0, 5]

def test_coactivation_queue_sampling():
    dict_size = 8
    capacity = 10
    
    queue = CoActivationQueue(dict_size=dict_size, capacity=capacity)
    depths = torch.arange(0, dict_size, dtype=torch.float32) * 0.1  # depth_i = i * 0.1
    
    # Initial sampling check (should fallback because buffer limit < 10)
    pos_p, pos_c, neg_p, neg_c = queue.sample_pairs(num_pairs=5, depths=depths)
    assert pos_p.shape == (5,)
    assert pos_c.shape == (5,)
    
    # Populate the queue to activate sampling (fill index above 10)
    # We create tokens where feature 2 and feature 5 are active
    f = torch.zeros(12, dict_size)
    f[:, 2] = 1.0
    f[:, 5] = 1.0
    queue.update(f)
    
    assert queue.head_idx.item() == 2  # wraps around (12 % 10)
    assert queue.is_full.item()
    
    # Sample actual co-activations
    pos_p, pos_c, neg_p, neg_c = queue.sample_pairs(num_pairs=5, depths=depths)
    
    assert pos_p.shape == (5,)
    assert pos_c.shape == (5,)
    
    # In all positive pairs, feature 2 and feature 5 should be mapped.
    # Since depth_2 = 0.2 and depth_5 = 0.5, parent should be 2 and child should be 5.
    for i in range(5):
        assert pos_p[i].item() == 2
        assert pos_c[i].item() == 5
