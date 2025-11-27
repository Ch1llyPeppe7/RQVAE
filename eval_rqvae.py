import gin
import os
import torch
import numpy as np
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
       iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,   
    runname="",
    project="",
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    dataset_split="beauty",
    noise_test='False'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- wandb init ---
    if wandb_logging:
        wandb.login()
        run = wandb.init(name=runname, project=project)

    # --- Dataset ---
    eval_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        train_test_split="all",
        split=dataset_split
    )
    if noise_test:
        
    eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    # --- RQ-VAE ---
    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=False,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats
    ).to(device)

    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)

    model.eval()
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats
    )
    tokenizer.rq_vae = model

    id_diversity_log = {}
    total_loss_list = []
    reconstruction_loss_list = []
    rqvae_loss_list = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating RQ-VAE"):
            data = batch_to(batch, device)
            out = model(data,0.2)
            total_loss_list.append(out.loss.cpu().item())
            reconstruction_loss_list.append(out.reconstruction_loss.cpu().item())
            rqvae_loss_list.append(out.rqvae_loss.cpu().item())

        # --- Losses ---
        id_diversity_log["eval_total_loss"] = np.mean(total_loss_list)
        id_diversity_log["eval_reconstruction_loss"] = np.mean(reconstruction_loss_list)
        id_diversity_log["eval_rqvae_loss"] = np.mean(rqvae_loss_list)

        
      # --- Corpus ids for entropy / usage ---
        corpus_ids = tokenizer.precompute_corpus_ids(eval_dataset)
        max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]
        id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()

        # --- Global entropy over all layers ---
        _, counts = torch.unique(corpus_ids[:, :vae_n_layers], dim=0, return_counts=True)
        p_global = counts / corpus_ids.shape[0]
        rqvae_entropy = -(p_global * torch.log(p_global)).sum()
        id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()

        # --- Per-layer entropy / usage / conditional entropy / mutual information ---
        for cid in range(vae_n_layers):
            # --- marginal entropy ---
            _, counts = torch.unique(corpus_ids[:, cid], return_counts=True)
            p_layer = counts / corpus_ids.shape[0]
            H_layer = -(p_layer * torch.log(p_layer)).sum().cpu().item()
            id_diversity_log[f"codebook_entropy_{cid}"] = H_layer
            id_diversity_log[f"codebook_usage_{cid}"] = len(counts) / vae_codebook_size

            # --- conditional entropy H(C_cid | C_<cid>) ---
            if cid > 0:
                context = corpus_ids[:, :cid]
                current = corpus_ids[:, cid].unsqueeze(1)
                joint = torch.cat([context, current], dim=1)

                # joint entropy
                _, joint_counts = torch.unique(joint, dim=0, return_counts=True)
                p_joint = joint_counts / corpus_ids.shape[0]
                H_joint = -(p_joint * torch.log(p_joint)).sum()

                # context entropy
                _, context_counts = torch.unique(context, dim=0, return_counts=True)
                p_context = context_counts / corpus_ids.shape[0]
                H_context = -(p_context * torch.log(p_context)).sum()

                # conditional entropy
                H_cond = (H_joint - H_context).cpu().item()
                id_diversity_log[f"codebook_cond_entropy_{cid}"] = H_cond

                # mutual information with previous layers
                I = H_layer - H_cond
                id_diversity_log[f"codebook_mutual_info_{cid}"] = I
            else:
                # first layer has no previous context
                id_diversity_log[f"codebook_cond_entropy_{cid}"] = H_layer
                id_diversity_log[f"codebook_mutual_info_{cid}"] = 0.0
    if wandb_logging:
        wandb.log(id_diversity_log)
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
