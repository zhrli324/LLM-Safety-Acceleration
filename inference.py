import torch
from model_utils import load_model
import time

def chunkify(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def perform_inference(model, tokenizer, inputs, batch_size=32, max_length=50):
    torch.cuda.empty_cache()
    device = model.model.device  # 使用模型的设备
    inputs = [inputs] if isinstance(inputs, str) else inputs

    batched_inputs = chunkify(inputs, batch_size)
    all_outputs = []

    start_time = time.time()

    with torch.no_grad():
        for batch in batched_inputs:
            for batch_input in batch:
                outputs = model.generate(batch_input, max_length)
                all_outputs.extend(outputs)

    end_time = time.time()
    print(f"Text generation time: {end_time - start_time:.2f} seconds")

    return all_outputs

def generate_text(model, tokenizer, input_text, batch_size=32, max_length=50):
    generated_tokens = perform_inference(model, tokenizer, input_text, batch_size, max_length)
    generated_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
    return generated_texts
