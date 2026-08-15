from typing import Callable, Any, Tuple, Union
import torch
from hypersae.models import HyperSAE

def make_transformer_lens_hook(
    model: HyperSAE,
    feature_id: int,
    alpha: float
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """
    Creates an activation steering hook compatible with TransformerLens modules.
    Adds a multiple alpha of the target concept's steering direction vector to
    the residual stream during model generation.
    
    Args:
        model: Trained HyperSAE instance
        feature_id: Feature ID index in the dictionary to steer
        alpha: Scaling coefficient multiplier
        
    Returns:
        hook_fn: A callable activation hook function with signature (activations, hook)
    """
    steering_vec = model.get_steering_vector(feature_id).detach().clone()
    
    def hook_fn(activations: torch.Tensor, hook: Any = None) -> torch.Tensor:
        # Move steering vector to active device and precision matching stream
        vec = steering_vec.to(device=activations.device, dtype=activations.dtype)
        
        # Broadcast dimensions to match stream (e.g. [batch, seq, d_model] -> [1, 1, d_model])
        dims_to_add = activations.dim() - 1
        view_shape = [1] * dims_to_add + [-1]
        vec_broadcast = vec.view(*view_shape)
        
        # Return a copy with steering offset added
        return activations + alpha * vec_broadcast
        
    return hook_fn

def make_pytorch_forward_hook(
    model: HyperSAE,
    feature_id: int,
    alpha: float
) -> Callable[[Any, Any, Union[torch.Tensor, Tuple[Any, ...]]], Union[torch.Tensor, Tuple[Any, ...]]]:
    """
    Creates a PyTorch standard forward hook compatible with nn.Module.register_forward_hook.
    Modifies module output streams in-place or returns a steered copy during inference.
    
    Args:
        model: Trained HyperSAE instance
        feature_id: Feature ID index in the dictionary to steer
        alpha: Scaling coefficient multiplier
        
    Returns:
        hook_fn: A callable forward hook function with signature (module, input, output)
    """
    steering_vec = model.get_steering_vector(feature_id).detach().clone()
    
    def hook_fn(module: Any, inputs: Any, outputs: Union[torch.Tensor, Tuple[Any, ...]]) -> Union[torch.Tensor, Tuple[Any, ...]]:
        if isinstance(outputs, torch.Tensor):
            vec = steering_vec.to(device=outputs.device, dtype=outputs.dtype)
            dims_to_add = outputs.dim() - 1
            view_shape = [1] * dims_to_add + [-1]
            return outputs + alpha * vec.view(*view_shape)
        elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            primary_output = outputs[0]
            vec = steering_vec.to(device=primary_output.device, dtype=primary_output.dtype)
            dims_to_add = primary_output.dim() - 1
            view_shape = [1] * dims_to_add + [-1]
            steered_primary = primary_output + alpha * vec.view(*view_shape)
            return (steered_primary,) + outputs[1:]
        else:
            return outputs
            
    return hook_fn
