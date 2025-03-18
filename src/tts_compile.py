import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

torch_device = "cuda:0"
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-mini-v1"

# Set padding max length
max_length = 50

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="eager"
).to(torch_device, dtype=torch_dtype)

# Adjust cache implementation if needed
model.generation_config.cache_implementation = "static"  # Use "dynamic" instead of "static"

# Compile the forward pass
compile_mode = "reduce-overhead"  # Changed to "default" for compatibility
model.forward = torch.compile(model.forward, mode=compile_mode)

# Warmup
inputs = tokenizer("This is for compilation", return_tensors="pt", padding="max_length", max_length=max_length).to(torch_device)
attention_mask = inputs.attention_mask  # Ensure attention_mask is provided
model_kwargs = {
    **inputs,
    "prompt_input_ids": inputs.input_ids,
    "prompt_attention_mask": inputs.attention_mask,
    "attention_mask": attention_mask,  # Add attention_mask explicitly
}

n_steps = 1 if compile_mode == "default" else 2
for _ in range(n_steps):
    _ = model.generate(**model_kwargs)
