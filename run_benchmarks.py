import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add src to path for hypersae
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hypersae.models import HyperSAE
from hypersae.queue import CoActivationQueue
from hypersae.loss import TriPartiteLoss
from hypersae.trainer import HyperSAETrainer

# --- 1. Define standard Flat Euclidean SAE Baseline Architecture ---
class FlatSAE(nn.Module):
    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        
        # Initialize encoder weights and bias
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dict_size, d_model)))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        
        # Initialize decoder weights (flat Euclidean directions)
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dict_size, d_model)))
        self.enforce_unit_norm()
        
    def forward(self, x):
        # Euclidean sparse encoding forward pass
        f = torch.relu(x @ self.W_enc.t() + self.b_enc)
        x_hat = f @ self.W_dec
        return x_hat, f
        
    def enforce_unit_norm(self):
        with torch.no_grad():
            norms = torch.norm(self.W_dec, p=2, dim=-1, keepdim=True)
            self.W_dec.copy_(self.W_dec / torch.clamp(norms, min=1e-8))

# --- 2. Define Flat SAE Training Loop Step ---
def train_flat_sae_step(model, optimizer, x, l1_coeff):
    optimizer.zero_grad()
    x_hat, f = model(x)
    
    # Reconstruction loss
    recon_loss = torch.mean((x - x_hat) ** 2)
    # L1 sparsity
    sparsity_loss = torch.mean(torch.sum(f, dim=-1))
    
    loss = recon_loss + l1_coeff * sparsity_loss
    
    loss.backward()
    optimizer.step()
    model.enforce_unit_norm()
    
    return {
        "loss_total": loss.item(),
        "loss_recon": recon_loss.item(),
        "loss_sparsity": sparsity_loss.item()
    }

# --- 3. Perplexity & Reconstruction Evaluation Metrics ---
def evaluate_model(llm, tokenizer, sae, is_flat, test_dataset, device, d_model=512):
    # Prepare batch of test inputs
    texts = [test_dataset[i]["text"] for i in range(min(len(test_dataset), 16))]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
    input_ids = inputs["input_ids"]
    
    # 1. Baseline CE Loss
    with torch.no_grad():
        outputs = llm(input_ids, labels=input_ids)
        ce_baseline = outputs.loss.item()
        
    # 2. Zero Ablation CE Loss
    def zero_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (torch.zeros_like(outputs[0]),) + outputs[1:]
        return torch.zeros_like(outputs)
        
    handle_zero = llm.gpt_neox.layers[3].register_forward_hook(zero_hook)
    with torch.no_grad():
        outputs = llm(input_ids, labels=input_ids)
        ce_zero = outputs.loss.item()
    handle_zero.remove()
    
    # Extract actual layer-3 hidden states to compute SAE metrics
    captured_states = []
    def extract_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            captured_states.append(outputs[0].detach())
        else:
            captured_states.append(outputs.detach())
            
    handle_ext = llm.gpt_neox.layers[3].register_forward_hook(extract_hook)
    with torch.no_grad():
        llm(input_ids)
    handle_ext.remove()
    
    states = captured_states[0]
    states_flat = states.view(-1, d_model).to(torch.float32)
    
    # Compute Reconstruction MSE and Sparsity L0
    with torch.no_grad():
        x_hat, f = sae(states_flat)
        recon_mse = torch.mean((states_flat - x_hat) ** 2).item()
        
        # L0 = average number of active features per token
        active_counts = torch.sum(f > 0, dim=-1)
        l0_sparsity = torch.mean(active_counts.to(torch.float32)).item()
        
    # 3. Reconstructed CE Loss
    def recon_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            orig = outputs[0]
        else:
            orig = outputs
            
        shape = orig.shape
        flat = orig.view(-1, d_model).to(torch.float32)
        with torch.no_grad():
            x_hat_eval, _ = sae(flat)
            
        recon = x_hat_eval.view(shape).to(dtype=orig.dtype)
        if isinstance(outputs, tuple):
            return (recon,) + outputs[1:]
        return recon
        
    handle_recon = llm.gpt_neox.layers[3].register_forward_hook(recon_hook)
    with torch.no_grad():
        outputs = llm(input_ids, labels=input_ids)
        ce_recon = outputs.loss.item()
    handle_recon.remove()
    
    # CE Recovery = (ce_zero - ce_recon) / (ce_zero - ce_baseline)
    denom = ce_zero - ce_baseline
    recovery = (ce_zero - ce_recon) / denom if denom > 1e-5 else 0.0
    
    return recon_mse, l0_sparsity, ce_baseline, ce_recon, recovery

# --- 4. Main Benchmark Sweep Loop ---
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Base Model
    model_name = "EleutherAI/pythia-70m"
    print("Loading EleutherAI/pythia-70m...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None, "Failed to load tokenizer"
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    llm.eval()
    
    for param in llm.parameters():
        param.requires_grad = False

    # Hook to extract training activations
    captured_activations = []
    def layer_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            captured_activations.append(outputs[0].detach())
        else:
            captured_activations.append(outputs.detach())
    hook_handle = llm.gpt_neox.layers[3].register_forward_hook(layer_hook)

    # Load Pile-10k dataset
    print("Loading datasets...")
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    
    # Define validation slice
    val_dataset = [{"text": dataset[i]["text"]} for i in range(200, 250)]

    # Model specifications
    d_model = 512
    dict_size = 2048
    c = 1.0
    
    # Sweep over L1 coefficients
    l1_sweep = [1e-4, 5e-4, 1e-3, 5e-3]
    
    results = {
        "hypersae": [],
        "flatsae": []
    }

    # -- A. Train & Evaluate HyperSAE across L1 sweep --
    for l1 in l1_sweep:
        print(f"\n[Training HyperSAE with L1 coeff = {l1}]")
        hyper_sae = HyperSAE(d_model=d_model, dict_size=dict_size).to(device)
        queue = CoActivationQueue(dict_size=dict_size, capacity=50000).to(device)
        loss_fn = TriPartiteLoss(l1_coeff=l1, entail_coeff=1e-2, c=c, K=0.5, gamma=0.15).to(device)
        trainer = HyperSAETrainer(model=hyper_sae, queue=queue, loss_fn=loss_fn, num_pairs=128, lr=1e-3)
        
        # Run 100 training steps (short run for benchmarking iteration)
        data_idx = 0
        for step in range(100):
            texts = [dataset[data_idx + i]["text"] for i in range(8)]
            data_idx += 8
            inputs = tokenizer(texts, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
            
            captured_activations.clear()
            with torch.no_grad():
                llm(**inputs)
                
            states = captured_activations[0]
            states_flat = states.view(-1, d_model).to(torch.float32)
            trainer.train_step(states_flat)

        # Save checkpoint
        torch.save(hyper_sae.state_dict(), f"checkpoint_hypersae_{l1}.pt")
        
        # Evaluate model metrics
        recon_mse, l0_sparsity, _, _, recovery = evaluate_model(llm, tokenizer, hyper_sae, False, val_dataset, device)
        print(f"HyperSAE L1={l1} | L0={l0_sparsity:.2f} | MSE={recon_mse:.5f} | CE Recovery={recovery*100:.2f}%")
        results["hypersae"].append({"l1": l1, "l0": l0_sparsity, "mse": recon_mse, "recovery": recovery})

    # -- B. Train & Evaluate FlatSAE across L1 sweep --
    for l1 in l1_sweep:
        print(f"\n[Training FlatSAE Baseline with L1 coeff = {l1}]")
        flat_sae = FlatSAE(d_model=d_model, dict_size=dict_size).to(device)
        optimizer = optim.AdamW(flat_sae.parameters(), lr=1e-3)
        
        data_idx = 0
        for step in range(100):
            texts = [dataset[data_idx + i]["text"] for i in range(8)]
            data_idx += 8
            inputs = tokenizer(texts, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
            
            captured_activations.clear()
            with torch.no_grad():
                llm(**inputs)
                
            states = captured_activations[0]
            states_flat = states.view(-1, d_model).to(torch.float32)
            train_flat_sae_step(flat_sae, optimizer, states_flat, l1)

        # Save checkpoint
        torch.save(flat_sae.state_dict(), f"checkpoint_flatsae_{l1}.pt")
        
        # Evaluate model metrics
        recon_mse, l0_sparsity, _, _, recovery = evaluate_model(llm, tokenizer, flat_sae, True, val_dataset, device)
        print(f"FlatSAE L1={l1} | L0={l0_sparsity:.2f} | MSE={recon_mse:.5f} | CE Recovery={recovery*100:.2f}%")
        results["flatsae"].append({"l1": l1, "l0": l0_sparsity, "mse": recon_mse, "recovery": recovery})

    # Remove active hook
    hook_handle.remove()

    # Save summary log
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nBenchmark results saved to benchmark_results.json.")

    # --- 5. Generate Pareto Frontier Comparison Plots ---
    print("\nGenerating Pareto frontier comparison plots...")
    
    # 1. MSE vs L0 Plot
    plt.figure(figsize=(7, 5))
    h_l0 = [r["l0"] for r in results["hypersae"]]
    h_mse = [r["mse"] for r in results["hypersae"]]
    f_l0 = [r["l0"] for r in results["flatsae"]]
    f_mse = [r["mse"] for r in results["flatsae"]]
    
    plt.plot(h_l0, h_mse, marker='o', linestyle='-', color='#6366f1', label="HyperSAE (Ours)")
    plt.plot(f_l0, f_mse, marker='s', linestyle='--', color='#a8a29e', label="Flat Euclidean SAE (Baseline)")
    
    plt.xlabel("Sparsity (L0 Norm - Active Features/Token)")
    plt.ylabel("Reconstruction Fidelity (MSE)")
    plt.title("Reconstruction Fidelity vs. Sparsity Pareto Frontier")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig("pareto_mse_frontier.png", dpi=300)
    plt.close()
    
    # 2. Perplexity Recovery vs L0 Plot
    plt.figure(figsize=(7, 5))
    h_rec = [r["recovery"] * 100 for r in results["hypersae"]]
    f_rec = [r["recovery"] * 100 for r in results["flatsae"]]
    
    plt.plot(h_l0, h_rec, marker='o', linestyle='-', color='#8b5cf6', label="HyperSAE (Ours)")
    plt.plot(f_l0, f_rec, marker='s', linestyle='--', color='#a8a29e', label="Flat Euclidean SAE (Baseline)")
    
    plt.xlabel("Sparsity (L0 Norm - Active Features/Token)")
    plt.ylabel("Cross-Entropy Loss Recovery %")
    plt.title("LLM Downstream Perplexity Recovery vs. Sparsity")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig("pareto_perplexity_frontier.png", dpi=300)
    plt.close()

    print("Pareto plots saved to pareto_mse_frontier.png and pareto_perplexity_frontier.png.")
    print("Benchmark sweep completed successfully!")

if __name__ == "__main__":
    main()
