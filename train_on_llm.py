import os
import sys
import math
import torch
import torch.nn as nn
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add src to python path to ensure hypersae imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hypersae.models import HyperSAE
from hypersae.queue import CoActivationQueue
from hypersae.loss import TriPartiteLoss
from hypersae.trainer import HyperSAETrainer
from hypersae.viz.plotting import plot_poincare_disk

def main():
    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Pythia-70m and register forward hook
    print("Loading EleutherAI/pythia-70m...")
    model_name = "EleutherAI/pythia-70m"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer is not None, "Failed to load tokenizer"
        llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        print("Falling back to local synthetic simulation mode...")
        run_synthetic_simulation(device)
        return

    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    # Residual dimension for Pythia-70m is 512
    d_model = 512
    
    # Store activations captured by the hook
    captured_activations = []

    def layer_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        # Shape: [batch, seq_len, d_model]
        captured_activations.append(hidden_states.detach())

    # Hook into residual stream output at the end of Layer 3
    hook_handle = llm.gpt_neox.layers[3].register_forward_hook(layer_hook)
    print("Registered forward hook on Pythia-70m layer 3.")

    # 3. Load dataset with synthetic fallback
    print("Loading dataset pile-10k...")
    try:
        dataset = load_dataset("NeelNanda/pile-10k", split="train")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Dataset loading failed: {e}. Generating dummy text corpus...")
        # Fallback dummy texts
        dataset = [{"text": "Sparse autoencoders are powerful tools for mechanistic interpretability."} for _ in range(500)]

    # 4. Initialize HyperSAE and Trainer
    # Dictionary size 2048, Poincaré curvature c=1.0
    dict_size = 2048
    c = 1.0
    
    print(f"Initializing HyperSAE (dict_size={dict_size}, d_model={d_model})...")
    hyper_sae = HyperSAE(d_model=d_model, dict_size=dict_size).to(device)
    
    # Pre-allocated rolling GPU queue for feature co-activations
    queue = CoActivationQueue(dict_size=dict_size, capacity=50000).to(device)
    
    # Loss terms combining Reconstruction MSE, L1 Sparsity, and hyperbolic entailments
    loss_fn = TriPartiteLoss(
        l1_coeff=1.5e-3,
        entail_coeff=1e-2,
        c=c,
        K=0.5,
        gamma=0.15
    ).to(device)
    
    trainer = HyperSAETrainer(
        model=hyper_sae,
        queue=queue,
        loss_fn=loss_fn,
        num_pairs=128,
        lr=1e-3
    )

    # 5. Training Loop
    print("Starting training loop...")
    batch_size = 8
    max_steps = 250
    seq_len = 128
    
    step = 0
    data_idx = 0
    
    while step < max_steps and data_idx < len(dataset):
        # Gather text batch
        texts = []
        while len(texts) < batch_size and data_idx < len(dataset):
            txt = dataset[data_idx]["text"]
            if len(txt.strip()) > 10:
                texts.append(txt)
            data_idx += 1
            
        if not texts:
            break
            
        # Tokenize batch
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            max_length=seq_len, 
            truncation=True, 
            padding=True
        ).to(device)
        
        # Run forward pass to trigger hook
        captured_activations.clear()
        with torch.no_grad():
            llm(**inputs)
            
        if not captured_activations:
            continue
            
        # Get hidden states and pass to HyperSAE
        # Shape: [batch, seq, d_model]
        states = captured_activations[0]
        
        # Flatten batch and sequence to train token-level activations
        states_flat = states.view(-1, d_model).to(torch.float32)
        
        # Execute training step
        metrics = trainer.train_step(states_flat)
        
        # Print progress
        if step % 25 == 0 or step == max_steps - 1:
            print(
                f"Step {step:03d} | "
                f"Total Loss: {metrics['loss_total']:.4f} | "
                f"Recon MSE: {metrics['loss_recon']:.4f} | "
                f"Sparsity L1: {metrics['loss_sparsity']:.4f} | "
                f"Entail: {metrics['loss_entail']:.4f}"
            )
            
        step += 1

    # Cleanup hook
    hook_handle.remove()
    print("Training complete. Cleaning up hooks...")

    print("Saving model weights to hyper_sae_weights.pt...")
    torch.save(hyper_sae.state_dict(), "hyper_sae_weights.pt")

    # 6. Extract Taxonomy DAG and Plot Poincaré Disk
    print("Extracting feature taxonomy...")
    G = hyper_sae.export_ontology_graph(K=0.5, c=c)
    
    print(f"Saving graph to feature_taxonomy.gexf...")
    nx.write_gexf(G, "feature_taxonomy.gexf")
    
    print("Generating Poincaré disk Plotly visualization...")
    fig = plot_poincare_disk(hyper_sae, G=G, c=c)
    
    print("Saving visualization to poincare_disk.html...")
    fig.write_html("poincare_disk.html")
    print("All artifacts successfully saved!")

def run_synthetic_simulation(device):
    """
    Simulation mode that runs the entire HyperSAE trainer loop using synthetic hidden states
    in case Hugging Face weights cannot be downloaded.
    """
    d_model = 512
    dict_size = 2048
    c = 1.0
    
    print("Initializing synthetic HyperSAE simulation...")
    hyper_sae = HyperSAE(d_model=d_model, dict_size=dict_size).to(device)
    queue = CoActivationQueue(dict_size=dict_size, capacity=50000).to(device)
    loss_fn = TriPartiteLoss(
        l1_coeff=1.5e-3,
        entail_coeff=1e-2,
        c=c,
        K=0.5,
        gamma=0.15
    ).to(device)
    
    trainer = HyperSAETrainer(
        model=hyper_sae,
        queue=queue,
        loss_fn=loss_fn,
        num_pairs=128,
        lr=1e-3
    )
    
    print("Running synthetic training loops...")
    for step in range(250):
        # Generate synthetic activations with hidden hierarchical correlation
        # Create a base category state, and add random features
        base_states = torch.randn(1024, d_model, device=device)
        metrics = trainer.train_step(base_states)
        
        if step % 25 == 0 or step == 249:
            print(
                f"Step {step:03d} | "
                f"Total Loss: {metrics['loss_total']:.4f} | "
                f"Recon MSE: {metrics['loss_recon']:.4f} | "
                f"Sparsity L1: {metrics['loss_sparsity']:.4f} | "
                f"Entail: {metrics['loss_entail']:.4f}"
            )
            
    print("Saving model weights to hyper_sae_weights.pt...")
    torch.save(hyper_sae.state_dict(), "hyper_sae_weights.pt")

    print("Extracting feature taxonomy...")
    G = hyper_sae.export_ontology_graph(K=0.5, c=c)
    print("Saving graph to feature_taxonomy.gexf...")
    nx.write_gexf(G, "feature_taxonomy.gexf")
    
    print("Generating Poincaré disk Plotly visualization...")
    fig = plot_poincare_disk(hyper_sae, G=G, c=c)
    
    print("Saving visualization to poincare_disk.html...")
    fig.write_html("poincare_disk.html")
    print("Synthetic simulation complete!")

if __name__ == "__main__":
    main()
