import torch
from load_data import get_loaders
import torch.nn as nn
from transformers import LlamaTokenizer
from sparsegpt import SparseGPT 

malicious = False
classifier_results = []
i = 1
is_harmful = False
sparsified = False
least_sparse_layers = []
pruning = False


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


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, dataloader, device):

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    # inps = torch.zeros((128, 2048, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(model, sparsity_ratio, device=torch.device("cuda:0"), prune_n=0, prune_m=0, nsamples=128, use_variant=False):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    tokenizer = LlamaTokenizer.from_pretrained("/root/autodl-tmp/llama/Llama-2-7b-chat-hf")

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return model


@torch.no_grad()
def prune_sparsegpt(model, sparsity_ratio, dev=torch.device("cuda:0"), nsamples=128, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    tokenizer = LlamaTokenizer.from_pretrained("/root/autodl-tmp/llama/Llama-2-7b-chat-hf")
    # model.seqlen = model.config.max_position_embeddings 
    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=0, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    # torch.cuda.empty_cache()


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
    sorted_layers = []
    for layer_number, sparsity in sorted_sparsity:
        sorted_layers.append(layer_number)
    
    sorted_layers.remove(31)

    least_sparse_layers = [layer_number for layer_number in sorted_layers[:num_layers]]

    # least_sparse_layers = []
    # for layer_number, sparsity in sorted_sparsity:
    #     if layer_number != 31:
    #         least_sparse_layers.append(layer_number)
    #         left_layer -= 1
    #         if left_layer < 0:
    #            break


    # 返回最不稀疏的层的数字序号
    # least_sparse_layers = [layer_number for layer_number in sorted_sparsity[:num_layers]]
    
    return least_sparse_layers, sorted_sparsity

