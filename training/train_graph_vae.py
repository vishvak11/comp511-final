import numpy as np
import torch
from tqdm import trange
from absl import logging
from absl.flags import FLAGS
from pathlib import Path
from data.utils import cast_loader_to_iterator
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import time
from sklearn.metrics import adjusted_rand_score  # <<< MODIFIED >>>
from sklearn.cluster import KMeans               # <<< MODIFIED >>>
import pandas as pd  
import scanpy as sc
from scipy.stats import rankdata
from .eval import calculate_ari, calculate_nmi


def load_lr_scheduler(optim, gamma = 0.5, step_size = 100000):
    return torch.optim.lr_scheduler.StepLR(optim, step_size, gamma)

def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError

def load_item_from_save(path, key, default, device):
    path = Path(path)
    if not path.exists():
        return default

    #ckpt = torch.load(path)
    ckpt = torch.load(path, map_location=device)
    if key not in ckpt:
        logging.warn(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]

def build_neighbor_loader(iterator_split, num_neighbors=10, batch_size=256, shuffle=True):
    """
    Build a NeighborLoader (train or test) from one split of your iterator.

    Args:
        iterator_split: one of iterator.train or iterator.test
        num_neighbors (list[int]): neighbor sampling sizes
        batch_size (int): mini-batch size
        shuffle (bool): whether to shuffle batches

    Returns:
        neighbor_loader (NeighborLoader)
        indices (torch.Tensor) – node indices returned by your iterator
    """
    # Unpack a full graph batch from your existing iterator
    full_inputs, adj_sparse, indices = next(iterator_split)
    row, col, val = adj_sparse.coo()

    # Assemble PyG graph
    full_edge_index = torch.stack([row, col], dim=0)
    full_edge_weight = val
    data = Data(x=full_inputs, edge_index=full_edge_index, edge_attr=full_edge_weight)

    # Create the loader
    neighbor_loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors],
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return neighbor_loader, indices


def train_graph_vae(
    model,
    optim,
    loader,
    outdir,
    n_iters=100,
    eval_freq=5,
    cache_freq=10,
    device="cpu",
    # Scheduler
    gamma=0.5, 
    step_size=100000,
    # Graph
    num_neighbors=10,
    batch_size=256,
    # Evaluation
    num_clusters=None, 
    labels=None, 
    compute_ari=False, 
    compute_nmi=False,
    seed_limit=256
):

    outdir = Path(outdir)
    cachedir = outdir / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)

    if device != 'cpu' and torch.cuda.is_available():
        device = torch.device(device)
        print("Using GPU!")
    else:
        device = torch.device('cpu')
        print("Using CPU!")

    model = model.to(device)  

    def state_dict(model, optim, **kwargs):
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        state.update(kwargs)
        return state

    def evaluate(
            neighbor_loader_test, 
            num_clusters=None, 
            labels=None, 
            compute_ari=False, 
            compute_nmi=False,
            seed_limit=256
        ):

        model.eval()
        all_losses = []
        if compute_ari:
            all_aris = []
        if compute_nmi:
            all_nmis = []
        
        with torch.no_grad():
            for i, batch in enumerate(neighbor_loader_test):
                batch_inputs = batch.x.to(device).squeeze(1)
                sampled_edge_index = batch.edge_index.to(device)
                batch_indices = batch.n_id  # maps into indices_test

                # Compute Evaluation Loss
                loss, comps, outs = model(batch_inputs, sampled_edge_index)
                loss = loss.mean().to(device)
                check_loss(loss)
                all_losses.append(loss.item())

                # Compute metrics as requested
                if (compute_ari or compute_nmi) and num_clusters is not None:
                    n = min(seed_limit, outs.z.shape[0])
                    if n >= max(2, num_clusters):
                        emb = outs.z[:n].squeeze(1).detach().cpu().numpy()

                        # extract true labels for these nodes
                        true_idx = indices_test[batch_indices[:n].cpu().numpy()]
                        lbls = labels[true_idx]

                        if compute_ari:
                            all_aris.append(calculate_ari(emb, lbls, num_clusters))
                        if compute_nmi:
                            all_nmis.append(calculate_nmi(emb, lbls, num_clusters))

        mean_loss = float(np.mean(all_losses)) if all_losses else float("nan")
        mean_ari = float(np.mean(all_aris)) if compute_ari and all_aris else None
        mean_nmi = float(np.mean(all_nmis)) if compute_nmi and all_nmis else None

        results = {"eval_loss": mean_loss}
        if mean_ari is not None:
            results["eval_ari"] = mean_ari
        if mean_nmi is not None:
            results["eval_nmi"] = mean_nmi

        #print("Evaluation results:", results)
        return results


    # *********** model, optim, loader = load(config, restore=cachedir / "last.pt") ***********

    iterator = cast_loader_to_iterator(loader, cycle_mode="all")
    scheduler = load_lr_scheduler(optim, gamma, step_size)

    step = load_item_from_save(cachedir / "last.pt", "step", 0, device=device)
    if scheduler is not None and step > 0:
        # keep LR schedule in sync
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(cachedir / "model.pt", "best_eval_loss", np.inf, device=device)
    eval_loss = best_eval_loss

    neighbor_loader, indices = build_neighbor_loader(
        iterator.train, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=True
    )

    # -------- Build full test graph once --------
    neighbor_loader_test, indices_test = build_neighbor_loader(
        iterator.test, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False
    )

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for epoch in ticker:
        model.train()
        model = model.to(device)

        # Make sure optimizer state is on the right device (kept)
        for state in optim.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(device).float()

        epoch_losses = []

        for batch in neighbor_loader:
            batch_inputs = batch.x.to(device).squeeze(1)
            sampled_edge_index = batch.edge_index.to(device)
            #batch_gene_weights = geneWeights_all[batch.n_id].to(device)

            optim.zero_grad()
            loss, comps, _ = model(batch_inputs, sampled_edge_index)
            loss = loss.mean().to(device)
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            loss.backward()
            optim.step()
            check_loss(loss)

            epoch_losses.append(loss.item())

        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # ---- EVAL (epoch-based) ----
        if epoch % eval_freq == 0:
            # TODO:
            results = evaluate(neighbor_loader_test, num_clusters, labels, compute_ari, compute_nmi, seed_limit)
            eval_loss_epoch = results['eval_loss']
            eval_ari_epoch  = results.get("eval_ari", None)
            eval_nmi_epoch  = results.get("eval_nmi", None)

            print("ARI: ", eval_ari_epoch)

            if eval_loss_epoch < best_eval_loss:
                best_eval_loss = eval_loss_epoch
                sd = state_dict(
                    model, optim,
                    step=(epoch + 1),
                    eval_loss=eval_loss_epoch,
                    best_eval_loss=best_eval_loss
                )
                torch.save(sd, cachedir / "model.pt")

        # ---- CACHE (epoch-based) ----
        if epoch % cache_freq == 0:
            torch.save(state_dict(model, optim, step=(epoch + 1)), cachedir / "last.pt")

        if scheduler is not None:
            scheduler.step()

        # tqdm line: show train/eval
        postfix = {"train_loss": f"{mean_train_loss:.4f}"}
        if eval_loss_epoch is not None: postfix["eval_loss"] = f"{eval_loss_epoch:.4f}"
        if eval_ari_epoch  is not None: postfix["eval_ari"]  = f"{eval_ari_epoch:.4f}"
        if eval_nmi_epoch  is not None: postfix["eval_nmi"]  = f"{eval_nmi_epoch:.4f}"
        ticker.set_postfix(postfix)

    # final safety save
    torch.save(state_dict(model, optim, step=epoch), cachedir / "last.pt")  