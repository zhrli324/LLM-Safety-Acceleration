import torch
from model_utils import load_model

def perform_inference(model, inputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    inputs = [inputs] if isinstance(inputs, str) else inputs

    with torch.no_grad():
        outputs = model.forward_with_classification(inputs)
    return outputs

def get_response(hidden_states):
    # 假设我们返回最后一个隐藏状态作为模型的响应
    return hidden_states[-1]
