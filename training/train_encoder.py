import numpy as np
import torch
from tqdm import trange
from absl import logging
from absl.flags import FLAGS
from pathlib import Path
from data.utils import cast_loader_to_iterator
import pandas as pd
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

def train_encoder(
    model,
    optim,
    loader,
    outdir,
    n_iters=100,
    eval_freq=5,
    cache_freq=10,
    device="cpu",
    iteration =0,
    gene_weights = None,
    # Scheduler
    gamma=0.5, 
    step_size=100000,
    # Evaluation
    num_clusters=None, 
    labels=None, 
    compute_ari=False, 
    compute_nmi=False
):
    
    outdir = Path(outdir)
    cachedir = outdir / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)

    if device != 'cpu' and torch.cuda.is_available():
        device = torch.device(device)
        print("Using GPU!")
        print("Device:", device)
    else:
        device = torch.device('cpu')
        print("Using CPU!")
        print("Device:", device)

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

    def evaluate(num_clusters=None, labels=None, compute_ari=False, compute_nmi=False):
        """
        Evaluate model reconstruction loss, ARI, and/or NMI.

        Args:
            num_clusters (str): Number of clusters for ARI and NMI
            labels: Ground truth labels for evaluation
            compute_ari (bool): whether to compute ARI
            compute_nmi (bool): whether to compute NMI
        """
        model.eval()
        all_losses = []
        if compute_ari:
            all_aris = []
        if compute_nmi:
            all_nmis = []

        with torch.no_grad():
            for vinputs in iterator.test:
                inputs = vinputs[0].to(device)
                batch_indices = vinputs[1].cpu().numpy().flatten()

                # Compute Evaluation Loss
                loss, comps, outs = model(inputs)
                loss = loss.mean().to(device)
                check_loss(loss)
                all_losses.append(loss.item())

                # embeddings
                outs_np = outs.z.cpu().squeeze(1).numpy()

                # Compute metrics as requested
                if compute_ari:
                    if num_clusters is not None and labels is not None:
                        try: 
                            batch_labels = labels[batch_indices]
                            ari = calculate_ari(outs_np, batch_labels, num_clusters)
                            all_aris.append(ari)
                        except ValueError as e:
                            # KMeans failure: batch too small for num_clusters
                            print(
                                f"[WARN] ARI skipped: batch_size={outs_np.shape[0]}, "
                                f"num_clusters={num_clusters} ({e})"
                            )
                    else: 
                        print("Unable to compute ARI. Please provide number of clusters and/or ground truth labels.")

                if compute_nmi:
                    if num_clusters is not None and labels is not None:
                        try: 
                            nmi = calculate_nmi(outs_np, labels, num_clusters)
                            all_nmis.append(nmi)
                        except ValueError as e:
                            # KMeans failure: batch too small for num_clusters
                            print(
                                f"[WARN] NMI skipped: batch_size={outs_np.shape[0]}, "
                                f"num_clusters={num_clusters} ({e})"
                            )
                    else:
                        print("Unable to compute NMI. Please provide number of clusters and/or ground truth labels.")

        mean_loss = float(np.mean(all_losses)) if all_losses else float("nan")
        mean_ari = float(np.mean(all_aris)) if compute_ari and all_aris else None
        mean_nmi = float(np.mean(all_nmis)) if compute_nmi and all_nmis else None

        print(mean_ari)

        results = {"eval_loss": mean_loss}
        if mean_ari is not None:
            results["eval_ari"] = mean_ari
        if mean_nmi is not None:
            results["eval_nmi"] = mean_nmi

        #print("Evaluation results:", results)
        return results
        

    # *********** model, optim, loader = load(config, restore=cachedir / "last.pt") ***********

    iterator = cast_loader_to_iterator(loader, cycle_mode="none")
    scheduler = load_lr_scheduler(optim, gamma, step_size)

    step = load_item_from_save(cachedir / "last.pt", "step", 0, device)
    if scheduler is not None and step > 0:
        # keep LR schedule in sync
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(cachedir / "model.pt", "best_eval_loss", np.inf, device)
    eval_loss = best_eval_loss

    # Restored Optimizer on same device
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device).float()

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for epoch in ticker:
        model.train()
        model = model.to(device)

        epoch_losses = []
        """
        batch = next(iterator.train)
        inputs = batch[0].to(device)
        optim.zero_grad()

        loss, comps, _ = model(inputs)
        loss = loss.mean().to(device)
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(loss)
        check_loss(loss)

        """
        for batch in iterator.train:
            inputs = batch[0].to(device)
            optim.zero_grad()

            if gene_weights != None:
                batch_gene_weights = gene_weights[batch.n_id].to(device)
            else:
                batch_gene_weights = None

            loss, comps, _ = model(inputs, batch_gene_weights)
            loss = loss.mean().to(device)
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}

            loss.backward()
            optim.step()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            check_loss(loss)

            epoch_losses.append(loss.item())

        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        #print("Train Loss: ", mean_train_loss)

        if epoch % eval_freq == 0:
            model.eval()
            model = model.to(device)
            results = evaluate(num_clusters, labels, compute_ari, compute_nmi)
            eval_loss_epoch = results['eval_loss']
            #print("Eval Loss: ", eval_loss_epoch)
            eval_ari_epoch  = results.get("eval_ari", None)
            eval_nmi_epoch  = results.get("eval_nmi", None)

            if eval_loss_epoch < best_eval_loss:
                best_eval_loss = eval_loss_epoch
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss, best_eval_loss=best_eval_loss)
                torch.save(sd, cachedir / f"model_iter{iteration}.pt")
        else:
            eval_loss_epoch = None
            eval_ari_epoch  = None
            eval_nmi_epoch  = None

        if epoch % cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / f"last_iter{iteration}.pt")

        if scheduler is not None:
            scheduler.step()

        # Update tqdm line with latest metrics
        postfix = {"train_loss": f"{mean_train_loss:.4f}"}
        if eval_loss_epoch is not None: postfix["eval_loss"] = f"{eval_loss_epoch:.4f}"
        if eval_ari_epoch  is not None: postfix["eval_ari"]  = f"{eval_ari_epoch:.4f}"
        #if eval_nmi_epoch  is not None: postfix["eval_nmi"]  = f"{eval_nmi_epoch:.4f}"
        ticker.set_postfix(postfix)

    torch.save(state_dict(model, optim, step=step), cachedir / f"last_iter{iteration}.pt")