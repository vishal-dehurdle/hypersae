import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to python path to ensure hypersae imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hypersae.models import HyperSAE
from hypersae.hooks import make_pytorch_forward_hook

def main():
    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Pythia-70m
    model_name = "EleutherAI/pythia-70m"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None, "Failed to load tokenizer"
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    llm.eval()

    # 3. Load trained HyperSAE weights
    d_model = 512
    dict_size = 2048
    hyper_sae = HyperSAE(d_model=d_model, dict_size=dict_size).to(device)
    
    weights_path = "hyper_sae_weights.pt"
    if not os.path.exists(weights_path):
        print(f"Weights file '{weights_path}' not found. Please run 'train_on_llm.py' first.")
        return
        
    print(f"Loading HyperSAE weights from {weights_path}...")
    hyper_sae.load_state_dict(torch.load(weights_path, map_location=device))
    hyper_sae.eval()

    # 4. Find which feature fires on a prompt
    prompt = "The scientific study of the brain and nervous"
    print(f"\nPrompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    captured = []
    def extract_hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            captured.append(outputs[0].detach())
        else:
            captured.append(outputs.detach())
            
    handle = llm.gpt_neox.layers[3].register_forward_hook(extract_hook)
    with torch.no_grad():
        llm(**inputs)
    handle.remove()
    
    # Cast to float32 and pass to HyperSAE
    activations = captured[0].view(-1, d_model).to(torch.float32)
    with torch.no_grad():
        _, f = hyper_sae(activations)
        
    # Get top activating feature index over all tokens in the prompt
    feature_sums = torch.sum(f, dim=0)
    top_vals, top_indices = torch.topk(feature_sums, k=5)
    
    print("\nTop 5 Activating Features on Prompt:")
    for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_indices.tolist())):
        print(f"Rank {rank+1} | Feature ID: {idx:04d} | Activation Sum: {val:.4f}")

    target_feature = int(top_indices[0].item())
    print(f"\nSteering target: Feature {target_feature:04d}")

    # 5. Generate Completions under different steering configurations
    def generate_completion(steering_alpha=0.0):
        assert tokenizer is not None, "Failed to resolve tokenizer"
        if steering_alpha != 0.0:
            # Generate the forward hook
            steer_hook = make_pytorch_forward_hook(
                model=hyper_sae,
                feature_id=target_feature,
                alpha=steering_alpha
            )
            hook_handle = llm.gpt_neox.layers[3].register_forward_hook(steer_hook)
        else:
            hook_handle = None
            
        try:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            with torch.no_grad():
                output_ids = llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=25,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False  # greedy decoding to keep comparisons deterministic
                )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

    print("\n--- Running Generation Comparisons ---")
    
    print("\n[Baseline (No Steering)]")
    baseline = generate_completion(0.0)
    print(baseline)
    
    print(f"\n[Steered Positive (Alpha = +8.0 on Feature {target_feature:04d})]")
    pos_steered = generate_completion(8.0)
    print(pos_steered)
    
    print(f"\n[Steered Negative (Alpha = -8.0 on Feature {target_feature:04d})]")
    neg_steered = generate_completion(-8.0)
    print(neg_steered)

if __name__ == "__main__":
    main()
