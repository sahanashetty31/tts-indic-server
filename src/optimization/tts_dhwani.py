import os
import gc
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import time

# Set environment variable for reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Utility function to clear GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Device selection
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Clear GPU memory before loading the model
clear_gpu_memory()

# Model and tokenizer initialization with mixed precision
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Input preparation (shortened inputs if necessary)
prompt = "ಕರ್ನಾಟಕದಲ್ಲಿ ಆರೋಗ್ಯ ಸೌಲಭ್ಯಗಳು ಹೆಚ್ಚುತ್ತಿವೆ. ಆಸ್ಪತ್ರೆಗಳು ಮತ್ತು ಕ್ಲಿನಿಕ್‌ಗಳು ರೋಗಿಗಳಿಗೆ ಉತ್ತಮ ಸೇವೆಯನ್ನು ಒದಗಿಸುತ್ತಿವೆ."
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# Clear GPU memory before generation
clear_gpu_memory()

# Measure time for generation
start_time = time.time()

generation = model.generate(
    input_ids=description_input_ids.input_ids,
    attention_mask=description_input_ids.attention_mask,
    prompt_input_ids=prompt_input_ids.input_ids,
    prompt_attention_mask=prompt_input_ids.attention_mask,
)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to generate audio: {elapsed_time:.2f} seconds")

# Save audio to file
audio_arr = generation.cpu().numpy().squeeze()
sf.write("indic_tts_out_2.wav", audio_arr, model.config.sampling_rate)

# Clear GPU memory after generation
clear_gpu_memory()
