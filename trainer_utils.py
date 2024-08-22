import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from modeling_llama_supervised import LlamaForCausalLM
from transformers import LlamaTokenizer
from joblib import dump
from load_data import get_data, load_conv
from visualization import accuracy_line


norm_prompt_path = './exp_data/normal_prompt.csv'
jailbreak_prompt_path = './exp_data/jailbreak_prompt.csv'
malicious_prompt_path = './exp_data/malicious_prompt.csv'


def load_exp_data(shuffle_seed=None, use_conv=False, model_name=None):
    normal_inputs = get_data(norm_prompt_path, shuffle_seed)
    malicious_inputs = get_data(malicious_prompt_path, shuffle_seed)
    if os.path.exists(jailbreak_prompt_path):
        jailbreak_inputs = get_data(jailbreak_prompt_path, shuffle_seed)
    else:
        jailbreak_inputs = None
    if use_conv and model_name is None:
        raise ValueError("please set model name for load")
    if use_conv:
        normal_inputs = [load_conv(model_name, _) for _ in normal_inputs]
        malicious_inputs = [load_conv(model_name, _) for _ in malicious_inputs]
        jailbreak_inputs = [load_conv(model_name, _) for _ in jailbreak_inputs] if jailbreak_inputs is not None else None
    return normal_inputs, malicious_inputs, jailbreak_inputs


def get_layer(forward_info, layer):
    new_forward_info = {}
    for k, v in forward_info.items():
        new_forward_info[k] = {"hidden_states": v["hidden_states"][layer], "label": v["label"]}
    return new_forward_info


class SafetyClassifier:
    def __init__(self, return_report=True, return_visual=False):
        self.return_report = return_report
        self.return_visual = return_visual

    @staticmethod
    def _process_data(forward_info):
        features = []
        labels = []
        for key, value in forward_info.items():
            for hidden_state in value["hidden_states"]:
                features.append(hidden_state.flatten())
                labels.append(value["label"])
        features = np.array(features)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def svm(self, forward_info, layer, path_template):
        X_train, X_test, y_train, y_test = self._process_data(forward_info)
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        dump(svm_model, path_template.format(layer))
        y_pred = svm_model.predict(X_test)
        report = None
        if self.return_report:
            print("SVM Test Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0.0))
        if self.return_visual:
            report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
        return X_test, y_pred, report

    def mlp(self, forward_info, layer, path_template, scaler_path_template):
        X_train, X_test, y_train, y_test = self._process_data(forward_info)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        dump(scaler, scaler_path_template.format(layer))
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.01,
                            solver='adam', verbose=0, random_state=42,
                            learning_rate_init=.01)
        mlp.fit(X_train_scaled, y_train)
        dump(mlp, path_template.format(layer))
        y_pred = mlp.predict(X_test_scaled)
        report = None
        if self.return_report:
            print(f"MLP Test Classification Report for Layer {layer}:")
            print(classification_report(y_test, y_pred, zero_division=0.0))
        if self.return_visual:
            report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
        return X_test, y_pred, report


class ClassifierTrainer:
    def __init__(self, model_path, layer_nums, skip=False, return_report=True, return_visual=True):
        self.model = LlamaForCausalLM.from_pretrained(model_path, skip=skip, device_map="auto")
        self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model_name = model_path.split("/")[-1]
        self.forward_info = {}
        self.return_report = return_report
        self.return_visual = return_visual
        self.layer_sums = layer_nums

    def step_forward(self, model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.to(model.device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        res_hidden_states = []
        for _ in outputs.hidden_states:
            res_hidden_states.append(_.detach().cpu().numpy())
        return res_hidden_states

    def get_forward_info(self, inputs_dataset, class_label, debug=True):
        offset = len(self.forward_info)
        for _, i in enumerate(inputs_dataset):
            if debug and _ > 100:
                break
            list_hs = self.step_forward(self.model, self.tokenizer, i)
            last_hs = [hs[:, -1, :] for hs in list_hs]
            self.forward_info[_ + offset] = {"hidden_states": last_hs, "label": class_label}
    
    def forward(self, datasets, debug=True):
        if isinstance(datasets, list):
            for class_num, dataset in enumerate(datasets):
                self.get_forward_info(dataset, class_num, debug=debug)
        elif isinstance(datasets, dict):
            for class_key, dataset in datasets.items():
                self.get_forward_info(dataset, class_key, debug=debug)

    def train_classifier(self, classifier_path_svm, classifier_path_mlp, scaler_path_mlp,
                  classifier_list=None, accuracy=True):
        classifier = SafetyClassifier(self.return_report, self.return_visual)
        if classifier_list is None:
            classifier_list = ["svm", "mlp"]
        rep_dict = {}
        if "svm" in classifier_list:
            rep_dict["svm"] = {}
            for layer in range(0, self.layer_sums):
                x, y, rep = classifier.svm(get_layer(self.forward_info, layer), layer, classifier_path_svm)
                rep_dict["svm"][layer] = rep
        if "mlp" in classifier_list:
            rep_dict["mlp"] = {}
            for layer in range(0, self.layer_sums):
                x, y, rep = classifier.mlp(get_layer(self.forward_info, layer), layer, classifier_path_mlp, scaler_path_mlp)
                rep_dict["mlp"][layer] = rep
        if not self.return_visual:
            return
        if accuracy and classifier_list != []:
            accuracy_line(rep_dict, self.model_name)