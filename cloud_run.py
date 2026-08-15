"""
cloud_run.py — Cloud-Scale Training & Downstream Evaluation Pipeline

Runs on a GCP VM with an NVIDIA GPU. Trains HyperSAE and FlatSAE on
Gemma-2-2B activations streamed from FineWeb-Edu, then evaluates
downstream reasoning on MMLU-Pro benchmarks.

Usage:
    source .venv/bin/activate
    PYTHONPATH=src python cloud_run.py

Environment:
    Requires .env with GOOGLE_CLOUD_PROJECT, GCS_BUCKET, HF_TOKEN,
    and GOOGLE_APPLICATION_CREDENTIALS.
"""

import os
import sys
import json
import time
import math
import gc
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dotenv import load_dotenv  # type: ignore[import-not-found]

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hypersae.models import HyperSAE
from hypersae.queue import CoActivationQueue
from hypersae.loss import TriPartiteLoss
from hypersae.trainer import HyperSAETrainer

load_dotenv()

# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "google/gemma-2-2b"
D_MODEL = 2304                    # Gemma-2-2B hidden dimension
HOOK_LAYER = 13                   # Middle layer of 26-layer model
DICT_SIZE = 16384                 # 7.1× overcomplete dictionary
TRAIN_STEPS = 10000               # ~50M tokens at batch_size=8, seq_len=256, tokens_per_step=~5120
BATCH_SIZE = 8
SEQ_LEN = 256
CHECKPOINT_INTERVAL = 5000        # Save checkpoint every N steps
LR = 3e-4
L1_SWEEP = [5e-4, 1e-3, 5e-3]    # Sparsity sweep coefficients

GCS_BUCKET = os.getenv("GCS_BUCKET", "hypersae-checkpoints")
HF_TOKEN = os.getenv("HF_TOKEN")

# ════════════════════════════════════════════════════════════════════════════════
# Flat Euclidean SAE Baseline
# ════════════════════════════════════════════════════════════════════════════════

class FlatSAE(nn.Module):
    """Standard Euclidean Sparse Autoencoder baseline (no hyperbolic component)."""

    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dict_size, d_model)))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dict_size, d_model)))
        self.enforce_unit_norm()

    def forward(self, x):
        f = torch.relu(x @ self.W_enc.t() + self.b_enc)
        x_hat = f @ self.W_dec
        return x_hat, f

    @torch.no_grad()
    def enforce_unit_norm(self):
        norms = torch.norm(self.W_dec, p=2, dim=-1, keepdim=True)
        self.W_dec.copy_(self.W_dec / torch.clamp(norms, min=1e-8))


def train_flat_sae_step(model, optimizer, x, l1_coeff):
    optimizer.zero_grad()
    x_hat, f = model(x)
    recon_loss = torch.mean((x - x_hat) ** 2)
    sparsity_loss = torch.mean(torch.sum(f, dim=-1))
    loss = recon_loss + l1_coeff * sparsity_loss
    loss.backward()
    optimizer.step()
    model.enforce_unit_norm()
    return {"loss_total": loss.item(), "loss_recon": recon_loss.item(), "loss_sparsity": sparsity_loss.item()}

# ════════════════════════════════════════════════════════════════════════════════
# GCS Upload Utility
# ════════════════════════════════════════════════════════════════════════════════

def upload_to_gcs(local_path: str, remote_path: str):
    """Uploads a local file to Google Cloud Storage."""
    try:
        from google.cloud import storage  # type: ignore[import-not-found]
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print(f"  ↑ Uploaded {local_path} → gs://{GCS_BUCKET}/{remote_path}")
    except Exception as e:
        print(f"  ⚠ GCS upload failed for {local_path}: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# Phase A: Training
# ════════════════════════════════════════════════════════════════════════════════

def get_gemma_hook_layer(llm):
    """Returns the target transformer layer module for Gemma-2."""
    return llm.model.layers[HOOK_LAYER]


def train_all_models(llm, tokenizer, device):
    """
    Trains HyperSAE and FlatSAE on Gemma-2-2B layer-13 activations
    across a sparsity sweep, returning all trained model checkpoints.
    """
    print("\n" + "═" * 72)
    print("  PHASE A — TRAINING ON GEMMA-2-2B ACTIVATIONS")
    print("═" * 72)

    # Freeze the LLM
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    # Activation capture hook
    captured = []

    def capture_hook(module, inputs, outputs):
        # Gemma layers return a tuple: (hidden_states, ...) 
        if isinstance(outputs, tuple):
            captured.append(outputs[0].detach())
        else:
            captured.append(outputs.detach())

    hook_handle = get_gemma_hook_layer(llm).register_forward_hook(capture_hook)

    # Stream FineWeb-Edu training data
    print("Streaming FineWeb-Edu dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    data_iter = iter(dataset)

    def get_batch():
        """Fetch a batch of tokenized sequences from the stream."""
        texts = []
        for _ in range(BATCH_SIZE):
            try:
                sample = next(data_iter)
                texts.append(sample["text"])
            except StopIteration:
                break
        if not texts:
            return None
        return tokenizer(
            texts,
            return_tensors="pt",
            max_length=SEQ_LEN,
            truncation=True,
            padding=True
        ).to(device)

    results = {"hypersae": {}, "flatsae": {}}

    # ─── Train HyperSAE configs ────────────────────────────────────────
    for l1 in L1_SWEEP:
        tag = f"hypersae_l1_{l1}"
        final_path = f"checkpoint_{tag}_final.pt"

        if os.path.exists(final_path):
            print(f"\n── Skipping {tag} — final checkpoint {final_path} already exists ──")
            results["hypersae"][l1] = {"model_path": final_path}
            continue

        print(f"\n── Training {tag} ({TRAIN_STEPS} steps) ──")

        hyper_sae = HyperSAE(d_model=D_MODEL, dict_size=DICT_SIZE).to(device)
        queue = CoActivationQueue(dict_size=DICT_SIZE, capacity=20000).to(device)
        loss_fn = TriPartiteLoss(l1_coeff=l1, entail_coeff=1e-2, c=1.0, K=0.5, gamma=0.15).to(device)
        trainer = HyperSAETrainer(model=hyper_sae, queue=queue, loss_fn=loss_fn, num_pairs=256, lr=LR)

        t0 = time.time()
        for step in range(1, TRAIN_STEPS + 1):
            batch = get_batch()
            if batch is None:
                print(f"  Data stream exhausted at step {step}")
                break

            captured.clear()
            with torch.no_grad():
                llm(**batch)

            if not captured:
                continue

            states = captured[0].view(-1, D_MODEL).to(torch.float32)
            metrics = trainer.train_step(states)

            if step % 500 == 0:
                elapsed = time.time() - t0
                tps = (step * BATCH_SIZE * SEQ_LEN) / elapsed
                print(f"  Step {step:>5d} | loss={metrics['loss_total']:.4f} "
                      f"recon={metrics['loss_recon']:.4f} | {tps:.0f} tok/s")

            if step % CHECKPOINT_INTERVAL == 0:
                ckpt_path = f"checkpoint_{tag}_step{step}.pt"
                torch.save(hyper_sae.state_dict(), ckpt_path)
                upload_to_gcs(ckpt_path, f"checkpoints/{ckpt_path}")

        # Save final checkpoint
        torch.save(hyper_sae.state_dict(), final_path)
        upload_to_gcs(final_path, f"checkpoints/{final_path}")
        results["hypersae"][l1] = {"model_path": final_path}
        print(f"  ✓ {tag} completed in {time.time() - t0:.1f}s")

        # Cleanup memory
        del hyper_sae, queue, loss_fn, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ─── Train FlatSAE configs ──────────────────────────────────────────
    # Reset data iterator for FlatSAE training (fresh stream)
    data_iter = iter(dataset)

    for l1 in L1_SWEEP:
        tag = f"flatsae_l1_{l1}"
        final_path = f"checkpoint_{tag}_final.pt"

        if os.path.exists(final_path):
            print(f"\n── Skipping {tag} — final checkpoint {final_path} already exists ──")
            results["flatsae"][l1] = {"model_path": final_path}
            continue

        print(f"\n── Training {tag} ({TRAIN_STEPS} steps) ──")

        flat_sae = FlatSAE(d_model=D_MODEL, dict_size=DICT_SIZE).to(device)
        optimizer = optim.AdamW(flat_sae.parameters(), lr=LR)

        t0 = time.time()
        for step in range(1, TRAIN_STEPS + 1):
            batch = get_batch()
            if batch is None:
                print(f"  Data stream exhausted at step {step}")
                break

            captured.clear()
            with torch.no_grad():
                llm(**batch)

            if not captured:
                continue

            states = captured[0].view(-1, D_MODEL).to(torch.float32)
            metrics = train_flat_sae_step(flat_sae, optimizer, states, l1)

            if step % 500 == 0:
                elapsed = time.time() - t0
                tps = (step * BATCH_SIZE * SEQ_LEN) / elapsed
                print(f"  Step {step:>5d} | loss={metrics['loss_total']:.4f} "
                      f"recon={metrics['loss_recon']:.4f} | {tps:.0f} tok/s")

            if step % CHECKPOINT_INTERVAL == 0:
                ckpt_path = f"checkpoint_{tag}_step{step}.pt"
                torch.save(flat_sae.state_dict(), ckpt_path)
                upload_to_gcs(ckpt_path, f"checkpoints/{ckpt_path}")

        torch.save(flat_sae.state_dict(), final_path)
        upload_to_gcs(final_path, f"checkpoints/{final_path}")
        results["flatsae"][l1] = {"model_path": final_path}
        print(f"  ✓ {tag} completed in {time.time() - t0:.1f}s")

        # Cleanup memory
        del flat_sae, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    hook_handle.remove()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Phase B: Downstream Evaluation (MMLU-Pro)
# ════════════════════════════════════════════════════════════════════════════════

def make_sae_hook(sae, d_model):
    """Creates a forward hook that replaces the final-token activation
    with the SAE's reconstruction (decision-point substitution)."""
    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, tuple):
            orig = outputs[0]
        else:
            orig = outputs
        recon = orig.clone()
        last_token = orig[:, -1, :].to(torch.float32)
        with torch.no_grad():
            x_hat, _ = sae(last_token)
        recon[:, -1, :] = x_hat.to(dtype=orig.dtype)
        if isinstance(outputs, tuple):
            return (recon,) + outputs[1:]
        return recon
    return hook_fn


def evaluate_mcq_benchmark(llm, tokenizer, device, dataset, choice_labels, hook_fn=None):
    """
    Runs logit-based multiple-choice evaluation on a dataset.

    Args:
        llm: The language model.
        tokenizer: The tokenizer.
        device: Compute device.
        dataset: List of dicts with 'prompt', 'correct_idx', 'choice_tokens'.
        choice_labels: List of token IDs for A, B, C, D (or more).
        hook_fn: Optional forward hook to apply to target layer.
    
    Returns:
        accuracy: Float accuracy as percentage.
    """
    hook_handle = None
    if hook_fn is not None:
        hook_handle = get_gemma_hook_layer(llm).register_forward_hook(hook_fn)

    correct = 0
    total = 0

    try:
        for item in dataset:
            inputs = tokenizer(item["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = llm(inputs["input_ids"])

            logits = outputs.logits[0, -1, :]
            choice_logits = logits[choice_labels[:item["num_choices"]]]
            pred_idx = torch.argmax(choice_logits).item()

            if pred_idx == item["correct_idx"]:
                correct += 1
            total += 1

            if total <= 3:
                print(f"  [{total:03d}] pred={pred_idx} correct={item['correct_idx']} "
                      f"logits={[f'{x:.3f}' for x in choice_logits.tolist()]}")
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return acc




def prepare_mmlu_pro(tokenizer):
    """Loads and formats the MMLU-Pro dataset for MCQ evaluation."""
    print("Loading MMLU-Pro dataset...")
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  MMLU-Pro unavailable ({e}). Using MMLU fallback.")
        try:
            ds = load_dataset("cais/mmlu", "all", split="test")
            return _format_mmlu_standard(ds)
        except Exception as e2:
            print(f"  MMLU also unavailable ({e2}). Using synthetic fallback.")
            return _mmlu_synthetic_fallback()

    formatted = []
    option_letters = "ABCDEFGHIJ"

    for item in ds:
        q = item.get("question", "")  # type: ignore[union-attr]
        options = item.get("options", [])  # type: ignore[union-attr]
        answer_idx = item.get("answer_index", item.get("answer", 0))  # type: ignore[union-attr]

        # Handle string answers (letter → index)
        if isinstance(answer_idx, str):
            answer_idx = option_letters.index(answer_idx) if answer_idx in option_letters else 0

        num_choices = min(len(options), 10)
        if num_choices < 2:
            continue

        prompt_lines = [f"Question: {q}"]
        for i, opt in enumerate(options[:num_choices]):
            prompt_lines.append(f"{option_letters[i]}) {opt}")
        prompt_lines.append("Answer:")
        prompt = "\n".join(prompt_lines)

        formatted.append({"prompt": prompt, "correct_idx": answer_idx, "num_choices": num_choices})

    print(f"  Loaded {len(formatted)} MMLU-Pro questions.")
    return formatted


def _format_mmlu_standard(ds):
    """Formats standard 4-choice MMLU as a fallback for MMLU-Pro."""
    formatted = []
    for item in ds:
        q = item.get("question", "")
        choices = item.get("choices", [])
        answer = item.get("answer", 0)
        if len(choices) != 4:
            continue
        prompt = f"Question: {q}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        formatted.append({"prompt": prompt, "correct_idx": answer, "num_choices": 4})
    return formatted


def _mmlu_synthetic_fallback():
    """Minimal synthetic fallback if MMLU-Pro is completely unavailable."""
    items = [
        {"q": "What is the capital of France?", "choices": ["London", "Paris", "Berlin", "Madrid"], "ans": 1},
        {"q": "Who developed the theory of general relativity?", "choices": ["Newton", "Einstein", "Bohr", "Feynman"], "ans": 1},
        {"q": "What is the powerhouse of the cell?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi"], "ans": 2},
    ]
    formatted = []
    for item in items:
        choices: list[str] = item["choices"]  # type: ignore[assignment]
        prompt = f"Question: {item['q']}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        formatted.append({"prompt": prompt, "correct_idx": item["ans"], "num_choices": 4})
    return formatted * 15


def run_evaluations(llm, tokenizer, device, train_results):
    """
    Evaluates baseline, FlatSAE, and HyperSAE hooks on MMLU-Pro benchmark.
    """
    print("\n" + "═" * 72)
    print("  PHASE B — DOWNSTREAM REASONING EVALUATIONS")
    print("═" * 72)

    # Encode choice tokens for Gemma-2 tokenizer
    choice_tokens_4 = [
        tokenizer.encode(" A", add_special_tokens=False)[-1],
        tokenizer.encode(" B", add_special_tokens=False)[-1],
        tokenizer.encode(" C", add_special_tokens=False)[-1],
        tokenizer.encode(" D", add_special_tokens=False)[-1],
    ]
    choice_tokens_10 = choice_tokens_4 + [
        tokenizer.encode(" E", add_special_tokens=False)[-1],
        tokenizer.encode(" F", add_special_tokens=False)[-1],
        tokenizer.encode(" G", add_special_tokens=False)[-1],
        tokenizer.encode(" H", add_special_tokens=False)[-1],
        tokenizer.encode(" I", add_special_tokens=False)[-1],
        tokenizer.encode(" J", add_special_tokens=False)[-1],
    ]
    print(f"Choice token IDs (A-D): {choice_tokens_4}")

    # Load benchmark datasets
    mmlu_data = prepare_mmlu_pro(tokenizer)

    # Use the best L1 config (lowest L1 = most faithful reconstruction)
    best_l1 = L1_SWEEP[0]
    hyper_path = train_results["hypersae"][best_l1]["model_path"]
    flat_path = train_results["flatsae"][best_l1]["model_path"]

    hyper_sae = HyperSAE(d_model=D_MODEL, dict_size=DICT_SIZE).to(device)
    hyper_sae.load_state_dict(torch.load(hyper_path, map_location=device))
    hyper_sae.eval()

    flat_sae = FlatSAE(d_model=D_MODEL, dict_size=DICT_SIZE).to(device)
    flat_sae.load_state_dict(torch.load(flat_path, map_location=device))
    flat_sae.eval()

    hyper_hook = make_sae_hook(hyper_sae, D_MODEL)
    flat_hook = make_sae_hook(flat_sae, D_MODEL)

    eval_results = {}

    # ── MMLU-Pro ────────────────────────────────────────────────────────
    print("\n── MMLU-Pro Evaluation ──")

    print("  [Baseline]")
    mmlu_baseline = evaluate_mcq_benchmark(llm, tokenizer, device, mmlu_data, choice_tokens_10)
    print(f"  Baseline Accuracy: {mmlu_baseline:.2f}%")

    print("  [FlatSAE]")
    mmlu_flat = evaluate_mcq_benchmark(llm, tokenizer, device, mmlu_data, choice_tokens_10, hook_fn=flat_hook)
    print(f"  FlatSAE Accuracy:  {mmlu_flat:.2f}%")

    print("  [HyperSAE]")
    mmlu_hyper = evaluate_mcq_benchmark(llm, tokenizer, device, mmlu_data, choice_tokens_10, hook_fn=hyper_hook)
    print(f"  HyperSAE Accuracy: {mmlu_hyper:.2f}%")

    eval_results["mmlu_pro"] = {
        "baseline": mmlu_baseline,
        "flatsae": mmlu_flat,
        "hypersae": mmlu_hyper,
        "num_questions": len(mmlu_data)
    }

    return eval_results


# ════════════════════════════════════════════════════════════════════════════════
# Phase C: Pareto Frontier Reconstruction Metrics
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_reconstruction_metrics(llm, tokenizer, device, train_results):
    """Computes MSE, L0, and CE Recovery for all trained configs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "═" * 72)
    print("  PHASE C — PARETO FRONTIER RECONSTRUCTION METRICS")
    print("═" * 72)

    # Activation capture
    captured = []
    def capture_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            captured.append(outputs[0].detach())
        else:
            captured.append(outputs.detach())

    hook_handle = get_gemma_hook_layer(llm).register_forward_hook(capture_hook)

    # Prepare validation data
    val_texts = []
    val_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, sample in enumerate(val_dataset):
        if i >= 64:  # 64 validation samples
            break
        val_texts.append(sample["text"])

    val_inputs = tokenizer(val_texts, return_tensors="pt", max_length=SEQ_LEN, truncation=True, padding=True).to(device)
    # Function to compute average CE loss with mini-batching to prevent CUDA OOM
    def compute_ce_loss(hook_func=None):
        total_loss = 0.0
        count = 0
        batch_size = 4
        for b in range(0, len(val_texts), batch_size):
            b_texts = val_texts[b:b+batch_size]
            inputs = tokenizer(b_texts, return_tensors="pt", max_length=SEQ_LEN, truncation=True, padding=True).to(device)
            b_ids = inputs["input_ids"]
            h = None
            if hook_func:
                h = get_gemma_hook_layer(llm).register_forward_hook(hook_func)
            with torch.no_grad():
                outputs = llm(b_ids, labels=b_ids)
                total_loss += outputs.loss.item() * len(b_texts)
                count += len(b_texts)
            if h:
                h.remove()
        return total_loss / count

    # Baseline CE
    ce_baseline = compute_ce_loss()

    # Zero ablation CE
    def zero_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (torch.zeros_like(outputs[0]),) + outputs[1:]
        return torch.zeros_like(outputs)

    ce_zero = compute_ce_loss(zero_hook)

    # Extract raw states with mini-batching
    raw_states_list = []
    batch_size = 4
    for b in range(0, len(val_texts), batch_size):
        b_texts = val_texts[b:b+batch_size]
        inputs = tokenizer(b_texts, return_tensors="pt", max_length=SEQ_LEN, truncation=True, padding=True).to(device)
        b_ids = inputs["input_ids"]
        captured.clear()
        with torch.no_grad():
            llm(b_ids)
        raw_states_list.append(captured[0].view(-1, D_MODEL).to(torch.float32))
    raw_states = torch.cat(raw_states_list, dim=0)
    hook_handle.remove()

    pareto_results = {"hypersae": [], "flatsae": []}

    for sae_type in ["hypersae", "flatsae"]:
        for l1 in L1_SWEEP:
            model_path = train_results[sae_type][l1]["model_path"]
            if sae_type == "hypersae":
                sae = HyperSAE(d_model=D_MODEL, dict_size=DICT_SIZE).to(device)
            else:
                sae = FlatSAE(d_model=D_MODEL, dict_size=DICT_SIZE).to(device)
            sae.load_state_dict(torch.load(model_path, map_location=device))
            sae.eval()

            with torch.no_grad():
                x_hat, f = sae(raw_states)
                mse = torch.mean((raw_states - x_hat) ** 2).item()
                l0 = torch.mean(torch.sum(f > 0, dim=-1).float()).item()

            # CE with reconstruction hook
            def recon_hook(module, inputs, outputs, _sae=sae):
                if isinstance(outputs, tuple):
                    orig = outputs[0]
                else:
                    orig = outputs
                flat = orig.view(-1, D_MODEL).to(torch.float32)
                with torch.no_grad():
                    x_hat_eval, _ = _sae(flat)
                recon = x_hat_eval.view(orig.shape).to(dtype=orig.dtype)
                if isinstance(outputs, tuple):
                    return (recon,) + outputs[1:]
                return recon

            ce_recon = compute_ce_loss(recon_hook)

            denom = ce_zero - ce_baseline
            recovery = (ce_zero - ce_recon) / denom if denom > 1e-5 else 0.0

            entry = {"l1": l1, "l0": l0, "mse": mse, "recovery": recovery, "ce_recon": ce_recon}
            pareto_results[sae_type].append(entry)
            print(f"  {sae_type} L1={l1} | L0={l0:.1f} | MSE={mse:.5f} | Recovery={recovery*100:.1f}%")

    # ── Generate Pareto Plots ───────────────────────────────────────────
    print("\nGenerating Pareto frontier plots...")

    # MSE vs L0
    plt.figure(figsize=(8, 5))
    h_l0 = [r["l0"] for r in pareto_results["hypersae"]]
    h_mse = [r["mse"] for r in pareto_results["hypersae"]]
    f_l0 = [r["l0"] for r in pareto_results["flatsae"]]
    f_mse = [r["mse"] for r in pareto_results["flatsae"]]

    plt.plot(h_l0, h_mse, "o-", color="#6366f1", label="HyperSAE (Ours)", linewidth=2, markersize=8)
    plt.plot(f_l0, f_mse, "s--", color="#a8a29e", label="Flat SAE (Baseline)", linewidth=2, markersize=8)
    plt.xlabel("Sparsity (L0 — Active Features/Token)", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Gemma-2-2B Layer 13 — MSE vs Sparsity Pareto Frontier", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("cloud_pareto_mse.png", dpi=300)
    upload_to_gcs("cloud_pareto_mse.png", "results/cloud_pareto_mse.png")
    plt.close()

    # Recovery vs L0
    plt.figure(figsize=(8, 5))
    h_rec = [r["recovery"] * 100 for r in pareto_results["hypersae"]]
    f_rec = [r["recovery"] * 100 for r in pareto_results["flatsae"]]

    plt.plot(h_l0, h_rec, "o-", color="#8b5cf6", label="HyperSAE (Ours)", linewidth=2, markersize=8)
    plt.plot(f_l0, f_rec, "s--", color="#a8a29e", label="Flat SAE (Baseline)", linewidth=2, markersize=8)
    plt.xlabel("Sparsity (L0 — Active Features/Token)", fontsize=12)
    plt.ylabel("CE Loss Recovery %", fontsize=12)
    plt.title("Gemma-2-2B Layer 13 — Downstream Recovery vs Sparsity", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("cloud_pareto_recovery.png", dpi=300)
    upload_to_gcs("cloud_pareto_recovery.png", "results/cloud_pareto_recovery.png")
    plt.close()

    print("  ✓ Pareto plots saved.")
    return pareto_results


# ════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 72)
    print("  HyperSAE — Cloud-Scale Training & Evaluation Pipeline")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Hook: Layer {HOOK_LAYER} | Dict Size: {DICT_SIZE}")
    print(f"  Train Steps: {TRAIN_STEPS} | GCS Bucket: {GCS_BUCKET}")
    print("═" * 72)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Training will be extremely slow.")

    # Authenticate HF
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)

    # Load Gemma-2-2B in float16
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    if tokenizer.pad_token is None:  # type: ignore[union-attr]
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore[union-attr]

    print(f"Model loaded. Parameters: {sum(p.numel() for p in llm.parameters()) / 1e9:.2f}B")

    # Phase A: Training
    train_results = train_all_models(llm, tokenizer, device)

    # Phase B: Downstream Evaluation
    eval_results = run_evaluations(llm, tokenizer, device, train_results)

    # Phase C: Pareto Frontier Metrics
    pareto_results = evaluate_reconstruction_metrics(llm, tokenizer, device, train_results)

    # ── Final Report ────────────────────────────────────────────────────
    final_report = {
        "model": MODEL_NAME,
        "hook_layer": HOOK_LAYER,
        "dict_size": DICT_SIZE,
        "train_steps": TRAIN_STEPS,
        "l1_sweep": L1_SWEEP,
        "evaluations": eval_results,
        "pareto": pareto_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    report_path = "cloud_eval_results.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    upload_to_gcs(report_path, f"results/{report_path}")

    print("\n" + "═" * 72)
    print("  ✓ ALL PHASES COMPLETE")
    print(f"  Results: {report_path}")
    print(f"  GCS: gs://{GCS_BUCKET}/results/")
    print("═" * 72)

    # Print summary table
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│             DOWNSTREAM EVALUATION SUMMARY           │")
    print("├──────────────┬──────────┬──────────┬────────────────┤")
    print("│ Benchmark    │ Baseline │ FlatSAE  │ HyperSAE      │")
    print("├──────────────┼──────────┼──────────┼────────────────┤")
    for bench_name, bench_data in eval_results.items():
        b = bench_data["baseline"]
        fl = bench_data["flatsae"]
        hy = bench_data["hypersae"]
        print(f"│ {bench_name:<12s} │ {b:6.2f}%  │ {fl:6.2f}%  │ {hy:6.2f}%        │")
    print("└──────────────┴──────────┴──────────┴────────────────┘")


if __name__ == "__main__":
    main()
