import torch

# 假设文件路径
path = "/public/home/wangdj2/Jin.Qian/RQ-VAE-Recommender/dataset/amazon/processed/data_beauty.pt"

data = torch.load(path, map_location="cpu",weights_only=False)  # 推荐用 CPU

print(len(data))
for i, x in enumerate(data):
    print(i, type(x), getattr(x, "shape", len(x) if isinstance(x, (list, tuple)) else ""))
