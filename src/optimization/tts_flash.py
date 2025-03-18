import os
import gc
import torch 
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Set environment variables for CUDA optimization and memory management.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Clear GPU memory utility function.
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

clear_gpu_memory()

# Device selection and precision.
torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16  # Use fp16 for compatibility.
model_name = "ai4bharat/indic-parler-tts"

# Load tokenizer and model with Flash Attention 2 enabled.
tokenizer = AutoTokenizer.from_pretrained(model_name)
description_tokenizer = AutoTokenizer.from_pretrained(model_name)

model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"  # Enable Flash Attention 2.
).to(torch_device, dtype=torch_dtype)

# Input preparation with fixed padding length.
max_length = 50

prompt = "ಕರ್ನಾಟಕದಲ್ಲಿ ಆರೋಗ್ಯ ಸೌಲಭ್ಯಗಳು ಹೆಚ್ಚುತ್ತಿವೆ. ಆಸ್ಪತ್ರೆಗಳು ಮತ್ತು ಕ್ಲಿನಿಕ್‌ಗಳು ರೋಗಿಗಳಿಗೆ ಉತ್ತಮ ಸೇವೆಯನ್ನು ಒದಗಿಸುತ್ತಿವೆ."
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."

description_input_ids = description_tokenizer(
    description,
    return_tensors="pt",
    padding="max_length",
    max_length=max_length
).to(torch_device)

prompt_input_ids = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    max_length=max_length
).to(torch_device)

# Generate audio.
generation_kwargs = {
    "input_ids": description_input_ids.input_ids,
    "attention_mask": description_input_ids.attention_mask,
    "prompt_input_ids": prompt_input_ids.input_ids,
    "prompt_attention_mask": prompt_input_ids.attention_mask,
}

try:
    generation = model.generate(**generation_kwargs)
except Exception as e:
    print(f"Error during generation: {e}")
    clear_gpu_memory()
    exit()

# Save audio to file.
audio_arr = generation.float().cpu().numpy().squeeze()  # Convert BFloat16 to float32.
sf.write("indic_tts_out.wav", audio_arr, model.config.sampling_rate)

# Clear GPU memory after generation.
clear_gpu_memory()
