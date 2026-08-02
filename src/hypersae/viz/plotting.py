import math
import torch
import numpy as np
import plotly.graph_objects as go
import networkx as nx

def plot_poincare_disk(
    model,
    G: nx.DiGraph | None = None,
    labels: list[str] | None = None,
    c: float = 1.0
) -> go.Figure:
    """
    Renders an interactive, premium 2D Poincaré disk visualization of dictionary features.
    
    Args:
        model: An instance of HyperSAE
        G: Optional NetworkX DiGraph representing hierarchical links (from Phase 4)
        labels: Optional list of strings to label each feature index
        c: Curvature parameter of the Poincaré ball
        
    Returns:
        fig: A Plotly Figure containing the Poincaré disk representation
    """
    # 1. Decoupled 2D coordinate projections
    # Extract Euclidean features [M, d_model] and learned depths r
    W = model.W_dec.detach().cpu().to(torch.float32)
    r = model.r.detach().cpu().to(torch.float32)
    M = W.size(0)
    
    # Run Singular Value Decomposition (SVD) on centered direction vectors
    W_mean = torch.mean(W, dim=0, keepdim=True)
    W_centered = W - W_mean
    
    U, S, _ = torch.linalg.svd(W_centered, full_matrices=False)
    coords_2d = U[:, :2] * S[:2]
    
    # Re-normalize 2D projected coordinates to unit circle
    norms_2d = torch.norm(coords_2d, p=2, dim=-1, keepdim=True)
    directions_2d = coords_2d / torch.clamp(norms_2d, min=1e-8)
    
    # Scale unit 2D directions by their exact hyperbolic Poincaré norm
    # d_i = tanh(sqrt(c) * r_i) / sqrt(c)
    poincare_norms = torch.tanh(math.sqrt(c) * r) / math.sqrt(c)
    coords_poincare_2d = directions_2d * poincare_norms.unsqueeze(-1)
    
    # Expose positions
    x_nodes = coords_poincare_2d[:, 0].numpy()
    y_nodes = coords_poincare_2d[:, 1].numpy()
    r_nodes = r.numpy()
    
    # 2. Render Canvas (Boundary and Layout)
    boundary_r = 1.0 / math.sqrt(c)
    theta = np.linspace(0, 2 * np.pi, 200)
    x_boundary = boundary_r * np.cos(theta)
    y_boundary = boundary_r * np.sin(theta)
    
    fig = go.Figure()
    
    # Add Bounding Poincaré Disk Outer Edge
    fig.add_trace(go.Scatter(
        x=x_boundary,
        y=y_boundary,
        mode="lines",
        line=dict(color="rgba(150, 150, 150, 0.4)", width=1.5, dash="dash"),
        name="Poincaré Boundary",
        hoverinfo="none",
        showlegend=True
    ))
    
    # 3. Render Entailment Edges (if taxonomy DAG is provided)
    if G is not None:
        edge_x = []
        edge_y = []
        for u, v in G.edges():
            edge_x.extend([x_nodes[u], x_nodes[v], None])
            edge_y.extend([y_nodes[u], y_nodes[v], None])
            
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(99, 102, 241, 0.25)", width=1.2),
            name="Entailment Links",
            hoverinfo="none",
            showlegend=True
        ))
        
    # 4. Render Features (Scatter Node Layer)
    hover_texts = []
    for i in range(M):
        lbl = labels[i] if labels is not None else f"Feature {i}"
        hover_texts.append(
            f"<b>Feature {i}</b><br>"
            f"Label: {lbl}<br>"
            f"Radial Depth r: {r_nodes[i]:.4f}<br>"
            f"Poincaré norm: {poincare_norms[i].item():.4f}<br>"
            f"2D Disk Coords: [{x_nodes[i]:.3f}, {y_nodes[i]:.3f}]"
        )
        
    # Premium color theme styling (Indigo-Purple-Yellow spectrum)
    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers",
        marker=dict(
            size=8.5,
            color=r_nodes,
            colorscale="Plasma",
            colorbar=dict(
                title=dict(
                    text="Radial Depth <i>r</i>",
                    font=dict(color="white", size=11)
                ),
                tickfont=dict(color="white")
            ),
            showscale=True,
            line=dict(width=0.6, color="rgba(255, 255, 255, 0.7)")
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Concept Nodes"
    ))
    
    # 5. Clean Layout Settings (Fixed Aspect Ratio to keep circle round)
    fig.update_layout(
        width=750,
        height=700,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1.0,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-boundary_r * 1.05, boundary_r * 1.05]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-boundary_r * 1.05, boundary_r * 1.05]
        ),
        plot_bgcolor="rgba(11, 15, 25, 1.0)",
        paper_bgcolor="rgba(11, 15, 25, 1.0)",
        legend=dict(
            font=dict(color="white", size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        ),
        margin=dict(l=15, r=15, t=55, b=15),
        title=dict(
            text="Interactive Poincaré Disk Concept Visualization",
            font=dict(color="white", size=15, family="Inter, sans-serif"),
            x=0.02,
            y=0.96
        )
    )
    
    return fig
