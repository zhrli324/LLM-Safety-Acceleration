import torch

malicious = False
classifier_results = []
i = 1
is_harmful = False
sparsified = False
least_sparse_layers = []


def decide_malicious(results):
    classifier_results = results[5:10]
    mali_count = classifier_results.count(['mali'])
    norm_count = classifier_results.count(['norm'])
    total_count = len(classifier_results)
    mali_ratio = mali_count / total_count
    norm_ratio = norm_count / total_count
    if mali_ratio > 0.6:
       return True
    elif norm_ratio > 0.6:
        return False
    else:
        classifier_results = results[5:20]
        mali_count = classifier_results.count(['mali'])
        norm_count = classifier_results.count(['norm'])
        total_count = len(classifier_results)
        mali_ratio = mali_count / total_count
        norm_ratio = norm_count / total_count
        if mali_ratio > 0.5:
            return True
        elif norm_ratio > 0.5:
            return False
        

def sparsify_model(model, sparsity_threshold=1e-3):
    """
    对模型进行稀疏化处理，将小于阈值的权重置零。

    Args:
        model: 要进行稀疏化的模型。
        sparsity_threshold: 小于该阈值的权重将被置零。

    Returns:
        稀疏化处理后的模型。
    """
    # 确保不在梯度计算模式下进行剪枝
    with torch.no_grad():
        for name, param in model.named_parameters():
            # 检查参数名中是否包含 "weight"，以避免剪枝偏置项
            if "weight" in name:
                # 创建掩码，只保留大于阈值的权重
                mask = torch.abs(param) > sparsity_threshold
                # 应用掩码，将小于阈值的权重置零
                param *= mask

    return model


def calculate_sparsity(param):
    """
    计算给定参数的稀疏度（非零元素的比例）。
    
    Args:
        param: 一个PyTorch张量，表示层的权重参数。

    Returns:
        稀疏度值（非零元素的比例）。
    """
    non_zero_elements = torch.sum(param != 0).item()
    total_elements = param.numel()
    sparsity = non_zero_elements / total_elements
    return sparsity


def calculate_layer_sparsity(model):
    """
    计算模型中每一层的综合稀疏度（各组件稀疏度的平均值）。
    
    Args:
        model: 一个PyTorch模型。

    Returns:
        一个字典，键是层的数字序号，值是该层的综合稀疏度。
    """
    layer_sparsity = {}
    current_layer_name = None
    current_layer_sparsities = []

    for name, param in model.named_parameters():
        # 提取层名，保留前三个部分，例如 "model.layers.0"
        parts = name.split(".")
        if len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit():
            layer_number = int(parts[2])  # 提取层的数字序号
            layer_name = layer_number  # 使用数字序号作为键
            
            # 如果我们已经处理完当前层的所有参数，计算该层的平均稀疏度
            if current_layer_name is not None and current_layer_name != layer_name:
                average_sparsity = sum(current_layer_sparsities) / len(current_layer_sparsities)
                layer_sparsity[current_layer_name] = average_sparsity
                current_layer_sparsities = []  # 重置列表
            
            current_layer_name = layer_name
            
            # 计算当前组件的稀疏度并存储
            if "weight" in name:
                sparsity = calculate_sparsity(param)
                current_layer_sparsities.append(sparsity)
    
    # 处理最后一层
    if current_layer_name is not None:
        average_sparsity = sum(current_layer_sparsities) / len(current_layer_sparsities)
        layer_sparsity[current_layer_name] = average_sparsity
    
    return layer_sparsity


def find_least_sparse_layers(model, num_layers=5):
    """
    找到模型中综合稀疏度最低的几层，并返回它们的数字序号。
    
    Args:
        model: 一个PyTorch模型。
        num_layers: 要返回的最不稀疏的层的数量。

    Returns:
        一个包含层的数字序号的列表，按照稀疏度从低到高排序。
    """
    layer_sparsity = calculate_layer_sparsity(model)
    
    # 按稀疏度从高到低排序，选择最不稀疏的几层
    sorted_sparsity = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
    
    # 返回最不稀疏的层的数字序号
    least_sparse_layers = [layer_number for layer_number, sparsity in sorted_sparsity[:num_layers]]
    
    return least_sparse_layers

