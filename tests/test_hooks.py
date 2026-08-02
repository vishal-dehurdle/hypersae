import torch
import torch.nn as nn
import pytest
from hypersae.models import HyperSAE
from hypersae.hooks import make_transformer_lens_hook, make_pytorch_forward_hook

def test_transformer_lens_hook():
    d_model = 4
    dict_size = 8
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    # Manually configure target steering vector direction
    # Set feature ID 2 direction to positive X axis
    w_dec = torch.zeros(dict_size, d_model)
    w_dec[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with torch.no_grad():
        model.W_dec.copy_(w_dec)
        model.enforce_unit_norm() # keeps column 2 at [1.0, 0.0, 0.0, 0.0]
        
    hook_fn = make_transformer_lens_hook(model, feature_id=2, alpha=2.5)
    
    # 3D Activation Stream [batch, seq, d_model]
    activations = torch.zeros(2, 3, d_model)
    
    steered = hook_fn(activations, None)
    
    # Verify shape
    assert steered.shape == (2, 3, d_model)
    # Check that steering vector [2.5, 0.0, 0.0, 0.0] was correctly added to all tokens
    expected = torch.zeros(2, 3, d_model)
    expected[..., 0] = 2.5
    assert torch.allclose(steered, expected, atol=1e-5)

def test_pytorch_forward_hook_tensor():
    d_model = 4
    dict_size = 8
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    w_dec = torch.zeros(dict_size, d_model)
    w_dec[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        model.W_dec.copy_(w_dec)
        model.enforce_unit_norm()
        
    hook_fn = make_pytorch_forward_hook(model, feature_id=1, alpha=-1.2)
    
    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(4))
                self.linear.bias.zero_()
        def forward(self, x):
            return self.linear(x)
            
    module = DummyModule()
    module.register_forward_hook(hook_fn)
    
    x = torch.zeros(3, 4)
    out = module(x)
    
    # Output should be identity mapped (zeros) + alpha * [0, 1, 0, 0] = [0, -1.2, 0, 0]
    expected = torch.zeros(3, 4)
    expected[:, 1] = -1.2
    assert torch.allclose(out, expected, atol=1e-5)

def test_pytorch_forward_hook_tuple():
    d_model = 4
    dict_size = 8
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    w_dec = torch.zeros(dict_size, d_model)
    w_dec[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    with torch.no_grad():
        model.W_dec.copy_(w_dec)
        model.enforce_unit_norm()
        
    hook_fn = make_pytorch_forward_hook(model, feature_id=0, alpha=3.0)
    
    # Dummy module returning a tuple of tensors (e.g. attention outputs)
    class TupleDummy(nn.Module):
        def forward(self, x):
            return (x, "metadata_string", {"info": 123})
            
    module = TupleDummy()
    module.register_forward_hook(hook_fn)
    
    x = torch.zeros(1, 2, 4)
    outputs = module(x)
    
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert outputs[1] == "metadata_string"
    assert outputs[2] == {"info": 123}
    
    # First output tensor must be steered
    steered_tensor = outputs[0]
    expected = torch.zeros(1, 2, 4)
    expected[..., 3] = 3.0
    assert torch.allclose(steered_tensor, expected, atol=1e-5)
