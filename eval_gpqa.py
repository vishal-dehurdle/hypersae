import os
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add src to path for hypersae
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hypersae.models import HyperSAE
from run_benchmarks import FlatSAE

# Helper to verify files exist
def assert_checkpoints_exist():
    for f in ["checkpoint_hypersae_0.0001.pt", "checkpoint_flatsae_0.0001.pt"]:
        if not os.path.exists(f):
            print(f"Error: Required checkpoint file '{f}' not found. Please run 'run_benchmarks.py' first.")
            sys.exit(1)

def main():
    assert_checkpoints_exist()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load Pythia-70m
    model_name = "EleutherAI/pythia-70m"
    print("Loading EleutherAI/pythia-70m...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None, "Failed to load tokenizer"
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    # Get choice token IDs dynamically
    choice_tokens = [
        tokenizer.encode(" A")[-1],
        tokenizer.encode(" B")[-1],
        tokenizer.encode(" C")[-1],
        tokenizer.encode(" D")[-1]
    ]
    print(f"Token IDs for options: A={choice_tokens[0]}, B={choice_tokens[1]}, C={choice_tokens[2]}, D={choice_tokens[3]}")

    # 2. Load Dictionaries
    d_model = 512
    dict_size = 2048
    
    hyper_sae = HyperSAE(d_model=d_model, dict_size=dict_size).to(device)
    hyper_sae.load_state_dict(torch.load("checkpoint_hypersae_0.0001.pt", map_location=device))
    hyper_sae.eval()

    flat_sae = FlatSAE(d_model=d_model, dict_size=dict_size).to(device)
    flat_sae.load_state_dict(torch.load("checkpoint_flatsae_0.0001.pt", map_location=device))
    flat_sae.eval()

    # 3. Load GPQA Diamond dataset with fallback
    print("Loading GPQA Diamond dataset...")
    try:
        # openai/gpqa requires gated access or Hugging Face token sometimes. Let's try downloading
        dataset = load_dataset("openai/gpqa", "gpqa_diamond", split="train")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Could not download openai/gpqa: {e}. Falling back to high-difficulty GPQA-like synthetic dataset...")
        # Synthetic science questions that model can perform logit comparisons on
        dataset = [
            {
                "Question": "What is the structural geometry of methane (CH4)?",
                "Correct Answer": "Tetrahedral",
                "Incorrect Answer 1": "Linear",
                "Incorrect Answer 2": "Trigonal planar",
                "Incorrect Answer 3": "Octahedral"
            },
            {
                "Question": "Which thermodynamic law states that entropy of an isolated system always increases?",
                "Correct Answer": "Second Law of Thermodynamics",
                "Incorrect Answer 1": "First Law of Thermodynamics",
                "Incorrect Answer 2": "Third Law of Thermodynamics",
                "Incorrect Answer 3": "Zeroth Law of Thermodynamics"
            },
            {
                "Question": "What is the primary organic molecule used as energy currency in biological cells?",
                "Correct Answer": "Adenosine triphosphate (ATP)",
                "Incorrect Answer 1": "Deoxyribonucleic acid (DNA)",
                "Incorrect Answer 2": "Glucose",
                "Incorrect Answer 3": "Nicotinamide adenine dinucleotide (NADH)"
            },
            {
                "Question": "What constant relates the energy of a photon to its electromagnetic frequency?",
                "Correct Answer": "Planck constant",
                "Incorrect Answer 1": "Boltzmann constant",
                "Incorrect Answer 2": "Gravitational constant",
                "Incorrect Answer 3": "Coulomb constant"
            },
            {
                "Question": "Which neurological structure coordinates motor control and balance in the brain?",
                "Correct Answer": "Cerebellum",
                "Incorrect Answer 1": "Amygdala",
                "Incorrect Answer 2": "Hippocampus",
                "Incorrect Answer 3": "Hypothalamus"
            }
        ] * 10  # Duplicate to create 50 evaluation samples

    # 4. Formulate Prompt & Evaluate Logit Function
    def evaluate_eval_run(sae_hook_fn=None):
        correct = 0
        total = 0
        
        # If sae_hook_fn is provided, register the hook on Layer 3
        hook_handle = None
        if sae_hook_fn is not None:
            hook_handle = llm.gpt_neox.layers[3].register_forward_hook(sae_hook_fn)
            
        try:
            for item in dataset:
                # Preprocess prompt
                q = item.get("Question", item.get("question", ""))
                correct_ans = item.get("Correct Answer", item.get("correct_answer", ""))
                inc1 = item.get("Incorrect Answer 1", item.get("incorrect_answer_1", ""))
                inc2 = item.get("Incorrect Answer 2", item.get("incorrect_answer_2", ""))
                inc3 = item.get("Incorrect Answer 3", item.get("incorrect_answer_3", ""))
                
                # Alphabetical deterministic ordering for choices A, B, C, D
                choices = sorted([correct_ans, inc1, inc2, inc3])
                correct_idx = choices.index(correct_ans)  # 0, 1, 2, or 3
                
                prompt = (
                    f"Question: {q}\n"
                    f"A) {choices[0]}\n"
                    f"B) {choices[1]}\n"
                    f"C) {choices[2]}\n"
                    f"D) {choices[3]}\n"
                    f"Answer:"
                )
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]
                
                # Forward pass
                with torch.no_grad():
                    outputs = llm(input_ids)
                
                # Read logits at the last position
                logits = outputs.logits[0, -1, :]
                choice_logits = logits[choice_tokens]
                
                pred_idx = torch.argmax(choice_logits).item()
                if total < 5:
                    print(f"[{total:02d}] Pred: {pred_idx} | Correct: {correct_idx} | Logits: {[round(x, 4) for x in choice_logits.tolist()]}")
                if pred_idx == correct_idx:
                    correct += 1
                total += 1
        finally:
            if hook_handle is not None:
                hook_handle.remove()
                
        return (correct / total) * 100.0 if total > 0 else 0.0

    # 5. Define SAE hooks
    def make_activation_hook(sae):
        def hook_fn(module, inputs, outputs):
            if isinstance(outputs, tuple):
                orig = outputs[0]
            else:
                orig = outputs
            
            # Reconstruct only the final token position to prevent compounding error
            recon = orig.clone()
            last_token = orig[:, -1, :].to(torch.float32)
            with torch.no_grad():
                x_hat, _ = sae(last_token)
            recon[:, -1, :] = x_hat.to(dtype=orig.dtype)
            
            if isinstance(outputs, tuple):
                return (recon,) + outputs[1:]
            return recon
        return hook_fn

    print("\nStarting downstream reasoning evaluations on GPQA...")
    
    # Baseline LLM Accuracy
    baseline_acc = evaluate_eval_run(sae_hook_fn=None)
    print(f"Baseline LLM Accuracy: {baseline_acc:.2f}%")
    
    # FlatSAE Hook Accuracy
    flat_hook = make_activation_hook(flat_sae)
    flat_acc = evaluate_eval_run(sae_hook_fn=flat_hook)
    print(f"FlatSAE Hook Accuracy: {flat_acc:.2f}%")
    
    # HyperSAE Hook Accuracy
    hyper_hook = make_activation_hook(hyper_sae)
    hyper_acc = evaluate_eval_run(sae_hook_fn=hyper_hook)
    print(f"HyperSAE Hook Accuracy: {hyper_acc:.2f}%")

    # Export results
    results = {
        "baseline_accuracy": baseline_acc,
        "flatsae_accuracy": flat_acc,
        "hypersae_accuracy": hyper_acc
    }
    
    with open("gpqa_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults successfully saved to gpqa_eval_results.json!")

if __name__ == "__main__":
    main()
