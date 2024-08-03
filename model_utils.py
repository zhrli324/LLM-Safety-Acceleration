import torch
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

class CustomModelForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.classifiers = {}
        self.scalers = {}

    def set_classifiers_and_scalers(self, classifiers, scalers):
        self.classifiers = classifiers
        self.scalers = scalers

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        hidden_states = outputs.hidden_states
        skip_even_layers = False
        filtered_hidden_states = []

        device = input_ids.device  # 获取输入数据所在的设备

        for i, hidden_state in enumerate(hidden_states):
            if i in self.classifiers and i in self.scalers:
                last_token_hidden_state = hidden_state[:, -1, :]  # 只取最后一个token的隐藏状态
                hidden_state_np = last_token_hidden_state.detach().cpu().numpy()
                hidden_state_np_scaled = self.scalers[i].transform(hidden_state_np)
                #hidden_state_np = hidden_state.mean(dim=1).detach().cpu().numpy()
                #hidden_state_np_scaled = self.scalers[i].transform(hidden_state_np)
                if self.classifiers[i].predict(hidden_state_np_scaled)[0]:
                    skip_even_layers = True
            if not (skip_even_layers and i % 2 == 0):
                print(f"Layer {i} not skipped")  # 添加打印语句
                filtered_hidden_states.append(hidden_state.to(device))  # 确保张量被移动回原设备

        outputs.hidden_states = tuple(filtered_hidden_states)
        return outputs

class TransformerModelWithClassifier:
    def __init__(self, transformer_model_name, classifier_paths, scaler_paths):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        classifiers = {i: joblib.load(path) for i, path in enumerate(classifier_paths)}
        scalers = {i: joblib.load(path) for i, path in enumerate(scaler_paths)}

        self.model = CustomModelForCausalLM.from_pretrained(transformer_model_name, device_map="auto")
        self.model.set_classifiers_and_scalers(classifiers, scalers)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, inputs, max_length=50):
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        generated_tokens = self.model.generate(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, max_length=max_length)
        return generated_tokens

def load_model(transformer_model_name, classifier_paths, scaler_paths):
    return TransformerModelWithClassifier(transformer_model_name, classifier_paths, scaler_paths)
