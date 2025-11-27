import argparse
import gin
import torch
from data.schemas import TokenizedSeqBatch
from einops import rearrange
from torch import Tensor


def reset_kv_cache(fn):
    def inner(self, *args, **kwargs):
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out
    
    return inner


def reset_encoder_cache(fn):
    def inner(self, *args, **kwargs):
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        out = fn(self, *args, **kwargs)
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        return out
    
    return inner


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def select_columns_per_row(x: Tensor, indices: Tensor) -> torch.Tensor:
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]

    B = x.shape[0]
    return x[
        rearrange(torch.arange(B, device=x.device), "B -> B 1"), indices
    ]


def maybe_repeat_interleave(x, repeats, dim):
    if not isinstance(x, Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)


# def parse_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config_path", type=str, help="Path to gin config file.")
#     args = parser.parse_args()
#     gin.parse_config_file(args.config_path)
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    
    # 新增可选覆盖参数
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--runname", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--rqvae_path", type=str, default=None)
    parser.add_argument("--noise_test", type=str, default=None)
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)

    # 覆盖 gin 参数
    if args.dataset_split is not None:
        gin.bind_parameter("train.dataset_split", args.dataset_split)
    if args.noise_test is not None:
        gin.bind_parameter("train.noise_test", args.dataset_split)
    if args.rqvae_path is not None:
        gin.bind_parameter("train.pretrained_rqvae_path", args.rqvae_path)
    if args.project is not None:
        gin.bind_parameter("train.project", args.project)
    if args.runname is not None:
        gin.bind_parameter("train.runname", args.runname)
    if args.save_dir is not None:
        gin.bind_parameter("train.save_dir_root", args.save_dir)

@torch.no_grad
def compute_debug_metrics(batch: TokenizedSeqBatch, model_output = None, prefix: str = "") -> dict:
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix + f"seq_length_p{q}": torch.quantile(seq_lengths, q=q).detach().cpu().item() 
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item() for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics