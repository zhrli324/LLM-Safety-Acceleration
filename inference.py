import torch
from model_utils import load_model
import time

def chunkify(lst, n):
    """将列表分割成大小为n的子列表"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def perform_inference(model, tokenizer, inputs, batch_size=32, max_length=50):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(inputs, str):
        inputs = [inputs]  # 如果inputs是字符串，将其转为包含该字符串的列表
    inputs = [inputs] if isinstance(inputs, str) else inputs

    batched_inputs = chunkify(inputs, batch_size)
    all_outputs = []

    start_time = time.time()  # 记录开始时间

    with torch.no_grad():
        for batch in batched_inputs:
            outputs = model.generate(batch, device, max_length)
            all_outputs.extend(outputs)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总耗时
    print(f"Inference time: {elapsed_time:.2f} seconds")  # 打印推理时间

    return all_outputs

def get_response(tokenizer, generated_tokens):
    # 使用tokenizer将生成的token解码成可读的文本
    responses = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
    return responses