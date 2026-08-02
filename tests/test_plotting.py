import math
import torch
import plotly.graph_objects as go
import networkx as nx
import pytest
from hypersae.models import HyperSAE
from hypersae.viz.plotting import plot_poincare_disk

def test_plot_poincare_disk():
    d_model = 8
    dict_size = 12
    c = 1.0
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    # Render basic disk
    fig = plot_poincare_disk(model, c=c)
    
    assert isinstance(fig, go.Figure)
    
    # Cast fig.data to Any to avoid type checking issues with Plotly's stubs
    from typing import Any
    data: Any = fig.data
    assert len(data) >= 2
    
    # Boundary check
    boundary_trace = data[0]
    assert boundary_trace.name == "Poincaré Boundary"
    assert boundary_trace.mode == "lines"
    
    # Nodes check
    node_trace = data[-1]
    assert node_trace.name == "Concept Nodes"
    assert node_trace.mode == "markers"
    
    # Assert coordinates remain strictly inside Poincaré boundaries
    # Radius = 1.0 / sqrt(c) = 1.0
    x_coords = node_trace.x
    y_coords = node_trace.y
    assert len(x_coords) == dict_size
    assert len(y_coords) == dict_size
    
    for x, y in zip(x_coords, y_coords):
        radius = math.sqrt(x**2 + y**2)
        assert radius < 1.0 + 1e-5, f"Node projected coordinate outside the Poincaré disk boundary: {radius}"

def test_plot_poincare_disk_with_graph():
    d_model = 4
    dict_size = 6
    c = 1.0
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    # Initialize some depths and coordinates
    w_dec = torch.zeros(dict_size, d_model)
    w_dec[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w_dec[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w_dec[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    w_dec[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    
    r = torch.tensor([0.0, 0.2, 0.2, 0.5, 0.5, 0.6], dtype=torch.float32)
    
    with torch.no_grad():
        model.W_dec.copy_(w_dec)
        model.r.copy_(r)
        
    # Build taxonomy graph
    G = model.export_ontology_graph(K=0.5, c=c)
    
    # Render disk with edges
    fig = plot_poincare_disk(model, G=G, c=c)
    
    assert isinstance(fig, go.Figure)
    
    # Cast fig.data to Any to avoid type checking issues with Plotly's stubs
    from typing import Any
    data: Any = fig.data
    
    # Expecting 3 traces: Poincaré Boundary, Entailment Links, Concept Nodes
    trace_names = [trace.name for trace in data]
    assert "Poincaré Boundary" in trace_names
    assert "Entailment Links" in trace_names
    assert "Concept Nodes" in trace_names
    
    # Validate the edges trace contains line coordinates
    edge_trace = [t for t in data if t.name == "Entailment Links"][0]
    assert edge_trace.mode == "lines"
    assert len(edge_trace.x) > 0
    assert len(edge_trace.y) > 0
    
    # Validate coordinate values
    node_trace = [t for t in data if t.name == "Concept Nodes"][0]
    for x, y in zip(node_trace.x, node_trace.y):
        assert x**2 + y**2 < 1.0 + 1e-5
