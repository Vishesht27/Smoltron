from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the converted model
model = AutoModelForCausalLM.from_pretrained("./hf_model_250m")
tokenizer = AutoTokenizer.from_pretrained("./hf_model_250m")
# Push to Hugging Face Hub
model.push_to_hub("rtaAILabs/Smoltron-250M-Ablation")
tokenizer.push_to_hub("rtaAILabs/Smoltron-250M-Ablation")