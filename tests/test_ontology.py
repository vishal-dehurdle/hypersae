import torch
import networkx as nx
import pytest
from hypersae.models import HyperSAE
from hypersae.utils.ontology import extract_ontology_graph

def test_extract_ontology_dag():
    d_model = 4
    dict_size = 6
    
    model = HyperSAE(d_model=d_model, dict_size=dict_size)
    
    # We will build a manual, deterministic hierarchy:
    # Node 0: Root (at the origin, depth = 0.0)
    # Node 1: Category A parent (depth = 0.2, direction [1, 0, 0, 0])
    # Node 2: Category B parent (depth = 0.2, direction [0, 1, 0, 0])
    # Node 3: Category A leaf (depth = 0.5, direction [1, 0, 0, 0]) -> should be child of 1
    # Node 4: Category B leaf (depth = 0.5, direction [0, 1, 0, 0]) -> should be child of 2
    # Node 5: Disjoint category leaf (depth = 0.6, direction [0, 0, 1, 0]) -> should be child of root (0) only
    
    w_dec = torch.zeros(dict_size, d_model)
    w_dec[0] = torch.tensor([1.0, 0.0, 0.0, 0.0]) # Root direction doesn't matter since norm -> 0
    w_dec[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w_dec[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    w_dec[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w_dec[4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    w_dec[5] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    
    r = torch.tensor([0.0, 0.2, 0.2, 0.5, 0.5, 0.6], dtype=torch.float32)
    
    with torch.no_grad():
        model.W_dec.copy_(w_dec)
        model.r.copy_(r)
        
    # Extract Graph
    G = model.export_ontology_graph(K=0.5, c=1.0)
    
    # 1. Assert DAG properties
    assert isinstance(G, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)
    assert G.number_of_nodes() == dict_size
    
    # 2. Check Node Attributes
    for node in G.nodes:
        assert "depth" in G.nodes[node]
        assert "coordinate" in G.nodes[node]
        assert G.nodes[node]["depth"] == pytest.approx(r[node].item())
        
    # 3. Check Expected Hierarchical Paths
    # Node 0 (origin) has depth 0, so it entails everything else (1, 2, 3, 4, 5)
    # However, transitive reduction should prune direct links from 0 if indirect pathways exist.
    # So we should expect direct edges:
    # 0 -> 1
    # 0 -> 2
    # 0 -> 5 (no intermediary)
    # 1 -> 3
    # 2 -> 4
    
    expected_edges = {(0, 1), (0, 2), (0, 5), (1, 3), (2, 4)}
    actual_edges = set(G.edges)
    
    # Check that they match exactly or are highly aligned
    for edge in expected_edges:
        assert edge in actual_edges, f"Expected hierarchical edge {edge} was not found"
        
    # Check that redundant direct edges like (0, 3) or (0, 4) were successfully pruned
    assert (0, 3) not in actual_edges, "Redundant edge (0, 3) was not pruned by transitive reduction"
    assert (0, 4) not in actual_edges, "Redundant edge (0, 4) was not pruned by transitive reduction"
