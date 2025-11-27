import torch
from pathlib import Path

file_path = Path("checkpoint_399999.pt")  # 你的 checkpoint 文件路径
output_path = Path("RQVAE_tensors.pt")  # 提取后保存的 tensor 文件

# 1. 定义占位类，避免找不到原模型报错
class RqVaePlaceholder:
    def __init__(self, *args, **kwargs):
        pass

# 2. 将占位类加入 safe globals
torch.serialization.add_safe_globals([RqVaePlaceholder])

# 3. 加载 checkpoint，只提取 tensor
state = torch.load(file_path, map_location="cpu", weights_only=False)

# 4. 提取 model.state_dict 或递归提取所有 tensor
def extract_tensors(obj, prefix=""):
    result = {}
    if isinstance(obj, torch.Tensor):
        result[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            result.update(extract_tensors(v, prefix=f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(extract_tensors(v, prefix=f"{prefix}.{i}" if prefix else str(i)))
    return result

if "model" in state:
    tensors = extract_tensors(state["model"])
else:
    tensors = extract_tensors(state)

# 5. 保存提取的 tensor
torch.save(tensors, output_path)
print(f"提取了 {len(tensors)} 个 tensor，已保存到 {output_path}")
