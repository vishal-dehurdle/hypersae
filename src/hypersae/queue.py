import torch
import torch.nn as nn

class CoActivationQueue(nn.Module):
    buffer: torch.Tensor
    head_idx: torch.Tensor
    is_full: torch.Tensor

    def __init__(self, dict_size: int, capacity: int = 65536):
        """
        Rolling FIFO Co-Activation Queue for tracking feature activations.
        Pre-allocates buffers directly on VRAM to prevent dynamic allocation overhead.
        
        Args:
            dict_size: Number of features in the autoencoder dictionary
            capacity: Maximum number of tokens/activations to hold in rolling VRAM history
        """
        super().__init__()
        self.dict_size = dict_size
        self.capacity = capacity
        
        # Pre-allocate binary activity history buffer
        self.register_buffer(
            "buffer",
            torch.zeros(capacity, dict_size, dtype=torch.bool)
        )
        
        # Rolling head index tracker
        self.register_buffer(
            "head_idx",
            torch.tensor(0, dtype=torch.long)
        )
        
        # Buffer-filled flag
        self.register_buffer(
            "is_full",
            torch.tensor(False, dtype=torch.bool)
        )

    def update(self, f: torch.Tensor):
        """
        Updates the queue with new activations from the current batch.
        Gradients are explicitly detached to prevent VRAM memory leaks.
        
        Args:
            f: Activation tensor, shape [..., dict_size]
        """
        # Detach activations to sever PyTorch Autograd tracking
        f_detached = f.detach()
        
        # Flatten batch and sequence dimensions to extract token activations [N, dict_size]
        f_flat = f_detached.view(-1, self.dict_size)
        is_active = f_flat > 0.0
        
        N = is_active.size(0)
        if N == 0:
            return
            
        curr_idx = int(self.head_idx.item())
        
        # Perform rolling buffer write
        if curr_idx + N <= self.capacity:
            self.buffer[curr_idx : curr_idx + N] = is_active
            new_idx = curr_idx + N
        else:
            # Wrap-around write
            first_part = self.capacity - curr_idx
            self.buffer[curr_idx:] = is_active[:first_part]
            second_part = N - first_part
            
            # Simple modulo wrap-around write for remaining batch
            rem = second_part % self.capacity
            self.buffer[:rem] = is_active[first_part : first_part + rem]
            new_idx = rem
            self.is_full.copy_(torch.tensor(True, dtype=torch.bool))
            
        self.head_idx.copy_(torch.tensor(new_idx, dtype=torch.long))

    def sample_pairs(self, num_pairs: int, depths: torch.Tensor):
        """
        Samples positive (co-activating) and negative (contrastive) pairs for loss regularizers.
        
        Args:
            num_pairs: Number of positive and negative pairs to sample
            depths: Decoder depth tensor r of shape [dict_size]
            
        Returns:
            pos_parents: Parent indices for positive pairs, shape [num_pairs]
            pos_children: Child indices for positive pairs, shape [num_pairs]
            neg_parents: Parent indices for negative pairs, shape [num_pairs]
            neg_children: Child indices for negative pairs, shape [num_pairs]
        """
        limit = self.capacity if self.is_full.item() else int(self.head_idx.item())
        
        # Fallback if queue has too few items to sample from
        if limit < 10:
            fallback_parents = torch.randint(0, self.dict_size, (num_pairs,), device=self.buffer.device)
            fallback_children = torch.randint(0, self.dict_size, (num_pairs,), device=self.buffer.device)
            return fallback_parents, fallback_children, fallback_parents, fallback_children
            
        # 1. Sample Positive (Co-activating) Pairs
        active_counts = torch.sum(self.buffer[:limit], dim=-1)
        valid_token_indices = torch.nonzero(active_counts >= 2).squeeze(-1)
        
        if valid_token_indices.numel() == 0:
            # Fallback if no tokens have >= 2 active features
            pos_parents = torch.randint(0, self.dict_size, (num_pairs,), device=self.buffer.device)
            pos_children = torch.randint(0, self.dict_size, (num_pairs,), device=self.buffer.device)
        else:
            # Sample tokens with multiple active features
            sampled_token_idxs = valid_token_indices[
                torch.randint(0, valid_token_indices.numel(), (num_pairs,), device=self.buffer.device)
            ]
            
            pos_parents_list = []
            pos_children_list = []
            
            for t_idx in sampled_token_idxs:
                # Find which features were active for the selected token
                active_features = torch.nonzero(self.buffer[t_idx]).squeeze(-1)
                
                # Randomly pick two distinct active features
                perm = torch.randperm(active_features.numel(), device=self.buffer.device)[:2]
                feat_a = active_features[perm[0]]
                feat_b = active_features[perm[1]]
                
                # Assign parent/child roles based on radial depth scalar (smaller depth = parent)
                depth_a = depths[feat_a].item()
                depth_b = depths[feat_b].item()
                
                if depth_a <= depth_b:
                    pos_parents_list.append(feat_a)
                    pos_children_list.append(feat_b)
                else:
                    pos_parents_list.append(feat_b)
                    pos_children_list.append(feat_a)
                    
            pos_parents = torch.stack(pos_parents_list)
            pos_children = torch.stack(pos_children_list)
            
        # 2. Sample Negative (Contrastive) Pairs
        neg_parents = torch.randint(0, self.dict_size, (num_pairs,), device=self.buffer.device)
        neg_children = torch.randint(0, self.dict_size, (num_pairs,), device=self.buffer.device)
        
        return pos_parents, pos_children, neg_parents, neg_children
