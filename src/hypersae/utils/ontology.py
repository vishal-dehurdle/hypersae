import math
import torch
import networkx as nx

def extract_ontology_graph(
    model,
    K: float = 0.5,
    c: float = 1.0,
    eps: float = 1e-5,
    tolerance: float = 0.0
) -> nx.DiGraph:
    """
    Constructs a Directed Acyclic Graph (DAG) of features using batch-vectorized 
    hyperbolic containment checking and prunes redundant links via transitive reduction.
    
    Args:
        model: An instance of HyperSAE
        K: Cone aperture tuning parameter (0 < K < 1)
        c: Curvature parameter of the Poincaré ball
        eps: Epsilon parameter for origin checks and boundary thresholds
        tolerance: Allowed violation threshold (default 0.0)
        
    Returns:
        TR: A transitive-reduced NetworkX DiGraph representing the learned taxonomy
    """
    # Exclude grads and project coordinates to the Poincare ball (float32 compute)
    with torch.no_grad():
        h = model.get_poincare_coordinates(c=c, eps=eps)
        r = model.r.detach().clone()
        M = h.size(0)
        
        # 1. Compute parent norms and apertures
        norm_sq = torch.sum(h * h, dim=-1, keepdim=True)  # [M, 1]
        norm = torch.sqrt(norm_sq)  # [M, 1]
        
        # Half-apertures: psi = arcsin( K * (1 - c * ||parent||^2) / ||parent|| )
        aperture_arg = K * (1.0 - c * norm_sq) / torch.clamp(norm, min=1e-7)
        aperture_arg = torch.clamp(aperture_arg, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        psi = torch.arcsin(aperture_arg)  # [M, 1]
        
        # 2. Pairwise Ganea angle Xi(u, v) calculations
        # dot_pc[i, j] = <h_i, h_j>
        dot_pc = torch.matmul(h, h.t())  # [M, M]
        
        parent_norm_sq = norm_sq  # [M, 1]
        child_norm_sq = norm_sq.t()  # [1, M]
        
        parent_norm = norm  # [M, 1]
        
        # Euclidean distances: ||u - v|| = sqrt(||u||^2 + ||v||^2 - 2 <u, v>)
        diff_norm_sq = parent_norm_sq + child_norm_sq - 2.0 * dot_pc
        diff_norm = torch.sqrt(torch.clamp(diff_norm_sq, min=1e-8))  # [M, M]
        
        # Numerator of cos_xi
        num = dot_pc * (1.0 + c * parent_norm_sq) - parent_norm_sq * (1.0 + c * child_norm_sq)  # [M, M]
        
        # Denominator of cos_xi
        denom_term = torch.sqrt(torch.clamp(1.0 + (c**2) * parent_norm_sq * child_norm_sq - 2.0 * c * dot_pc, min=1e-8))
        denom = parent_norm * diff_norm * denom_term  # [M, M]
        
        cos_xi = num / torch.clamp(denom, min=1e-8)
        cos_xi = torch.clamp(cos_xi, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        xi = torch.arccos(cos_xi)  # [M, M]
        
        # Check containment violations: xi - psi
        violations = xi - psi  # [M, M]
        
        # 3. Containment conditions (violations <= tolerance) & (r_parent < r_child)
        r_parent = r.unsqueeze(-1)  # [M, 1]
        r_child = r.unsqueeze(0)  # [1, M]
        
        adj_matrix = (violations <= tolerance) & (r_parent < r_child)
        adj_matrix.fill_diagonal_(False)
        
        # Edge case: Parent near the origin entails all downstream nodes (r_parent < r_child)
        is_origin = (parent_norm < eps)  # [M, 1]
        adj_matrix = torch.where(
            is_origin & (r_parent < r_child),
            torch.ones_like(adj_matrix),
            adj_matrix
        )
        
        # Move to CPU to construct NetworkX Graph
        adj_matrix_cpu = adj_matrix.cpu()
        edges = torch.nonzero(adj_matrix_cpu)
        
        # 4. Construct base NetworkX DiGraph
        G = nx.DiGraph()
        for i in range(M):
            G.add_node(
                i,
                depth=float(r[i].item()),
                coordinate=str(h[i].cpu().tolist())
            )
            
        G.add_edges_from(edges.tolist())
        
        # 5. Compute Transitive Reduction to prune indirect redundant links
        TR = nx.transitive_reduction(G)
        
        # Restore node attributes lost during transitive_reduction
        node_attrs = {node: G.nodes[node] for node in TR.nodes}
        nx.set_node_attributes(TR, node_attrs)  # type: ignore
        
        return TR  # type: ignore
