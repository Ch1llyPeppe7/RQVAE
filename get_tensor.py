import torch
from pathlib import Path

# checkpoint 路径
file_path = Path("checkpoint_399999.pt")  # 根据你的实际路径修改
save_dir = Path("extracted_codebooks")
save_dir.mkdir(exist_ok=True)

# 占位类，避免原模型类报错
class RqVaePlaceholder:
    def __init__(self, *args, **kwargs):
        pass

torch.serialization.add_safe_globals([RqVaePlaceholder])

# 加载 checkpoint
checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

# 提取 state_dict
state_dict = checkpoint.get("model", checkpoint)

# 遍历 state_dict 找出 codebook 张量
for k, v in state_dict.items():
    if "codebook" in k.lower() or "embed" in k.lower():
        tensor_path = save_dir / f"{k.replace('.', '_')}.pt"
        torch.save(v, tensor_path)
        print(f"Saved {k} with shape {v.shape} -> {tensor_path}")

print(f"Done! All codebook tensors are saved in {save_dir}/")
