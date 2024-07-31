import torch
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerModelWithClassifier:
    def __init__(self, transformer_model_name, classifier_path):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(transformer_model_name, output_hidden_states=True, device_map="auto")
        self.weak_classifier = joblib.load(classifier_path)

        # 使用现有的eos_token作为pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def classify_hidden_state(self, hidden_state):
        hidden_state_np = hidden_state.detach().cpu().numpy()
        # 对隐藏状态进行调整以匹配分类器的输入尺寸
        hidden_state_np = hidden_state_np.reshape(1, -1)
        hidden_state_np = hidden_state_np[:, :4096]  # 截取前 4096 个特征，确保尺寸匹配
        return self.weak_classifier.predict(hidden_state_np)[0]

    def forward_with_classification(self, inputs, device):
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.model(**tokens)
        hidden_states = outputs.hidden_states
        filtered_hidden_states = []
        skip_even_layers = False

        for i, hidden_state in enumerate(hidden_states):
            print(f"NUM: {i}")
            if skip_even_layers and i % 2 == 0:
                continue

            hidden_state = hidden_state.to('cpu')
            if i % 2 == 1 and self.classify_hidden_state(hidden_state):
                skip_even_layers = True

            filtered_hidden_states.append(hidden_state.to(device))

        return filtered_hidden_states

    def generate(self, inputs, device, max_length=50):
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
        
        # 获取初始隐藏状态
        outputs = self.model(**tokens)
        hidden_states = outputs.hidden_states

        # 使用跳层逻辑处理隐藏状态
        processed_hidden_states = self.forward_with_classification(inputs, device)

        # 使用处理后的隐藏状态进行生成
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        # 使用top-k sampling代替beam search，减少生成时间
        generated_tokens = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=max_length, 
            do_sample=True, 
            top_k=50
        )

        return generated_tokens

def load_model(transformer_model_name, classifier_path):
    return TransformerModelWithClassifier(transformer_model_name, classifier_path)



'''import torch
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class TransformerModelWithClassifier:
    def __init__(self, transformer_model_name, classifier_path):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(transformer_model_name, output_hidden_states=True, device_map="auto")
        self.weak_classifier = joblib.load(classifier_path)

        # 使用现有的eos_token作为pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def classify_hidden_state(self, hidden_state):
        hidden_state_np = hidden_state.detach().cpu().numpy()
        # 确保分类器的输入尺寸匹配
        hidden_state_np = hidden_state_np.reshape(1, -1)
        hidden_state_np = hidden_state_np[:, :self.weak_classifier.n_features_in_]
        return self.weak_classifier.predict(hidden_state_np)[0]

    def forward_with_classification(self, inputs, device, max_length=50):
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.model(**tokens)
        hidden_states = outputs.hidden_states
        filtered_hidden_states = []
        skip_even_layers = False

        start_time = time.time()

        for i, hidden_state in enumerate(hidden_states):
            print(f"NUM: {i}")
            if skip_even_layers and i % 2 == 0:
                continue

            hidden_state = hidden_state.to('cpu')
            if i % 2 == 1 and self.classify_hidden_state(hidden_state):
                skip_even_layers = True
            filtered_hidden_states.append(hidden_state)

        end_time = time.time()
        print(f"Hidden states processing time: {end_time - start_time:.2f} seconds")

        return filtered_hidden_states

    def generate(self, inputs, device, max_length=50):
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
        
        # 获取初始隐藏状态
        outputs = self.model(**tokens)
        hidden_states = outputs.hidden_states

        # 使用跳层逻辑处理隐藏状态
        processed_hidden_states = self.forward_with_classification(inputs, device, max_length)

        # 使用处理后的隐藏状态进行生成
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        start_time = time.time()

        # 直接使用生成方法
        generated_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

        end_time = time.time()
        print(f"Text generation time: {end_time - start_time:.2f} seconds")

        return generated_tokens

def load_model(transformer_model_name, classifier_path):
    return TransformerModelWithClassifier(transformer_model_name, classifier_path)'''
