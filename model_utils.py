import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TransformerModelWithClassifier:
    def __init__(self, transformer_model_name, classifier_path):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(transformer_model_name, output_hidden_states=True)
        self.weak_classifier = joblib.load(classifier_path)

    def classify_hidden_state(self, hidden_state):
        hidden_state_np = hidden_state.detach().cpu().numpy()
        return self.weak_classifier.predict(hidden_state_np.reshape(1, -1))[0]

    def forward_with_classification(self, inputs):
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        hidden_states = self.model(**tokens).hidden_states
        for i, hidden_state in enumerate(hidden_states):
            if i % 2 == 1:  # 奇数层
                if self.classify_hidden_state(hidden_state):
                    return hidden_states[:i+1]  # 跳过剩余偶数层
        return hidden_states

def load_model(transformer_model_name, classifier_path):
    return TransformerModelWithClassifier(transformer_model_name, classifier_path)
