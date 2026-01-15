from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Simple inference script for consolidated Smoltron models")
    parser.add_argument("--model_path", type=str, default="./hf_model_250m", help="Path to the consolidated HF model")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little bird named", help="Prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for sampling")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).cuda()

    # Encode prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    # Generate
    print(f"Generating (max {args.max_tokens} tokens)...")
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=args.max_tokens, 
            do_sample=True, 
            temperature=args.temp, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\n" + "="*20 + " MODEL OUTPUT " + "="*20)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("="*54 + "\n")

if __name__ == "__main__":
    main()
