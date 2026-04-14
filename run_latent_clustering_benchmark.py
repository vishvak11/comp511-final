import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA

from data.cell import build_cell_data, build_dataloaders
from models.ae import load_autoencoder_model
from models.vae import load_vae_model
from perturbation.encode_decode import encode_adata
from training.eval import calculate_ari, calculate_nmi
from training.train_encoder import train_encoder
from training.train_graph_encoder import train_graph_encoder
from training.train_graph_vae import train_graph_vae
from training.train_vae import train_vae


DATASET_PRESETS = {
    "axolotl_regeneration": "../ot/datasets/axolotl/Regeneration_subset_hvg2.h5ad",
    "axolotl_development": "../ot/datasets/axolotl/Development_hvg2.h5ad",
    "mosta": "../ot/datasets/mosta/mosta_hvg.h5ad",
}

MODEL_ALIASES = {
    "AE": "AE",
    "GAE": "GraphAE",
    "VAE": "VAE",
    "VGAE": "GraphVAE",
    "GraphAE": "GraphAE",
    "GraphVAE": "GraphVAE",
}

GRAPH_MODELS = {"GraphAE", "GraphVAE"}
DENSE_MODELS = {"AE", "VAE"}


def parse_args():
    # Keep the CLI flexible enough for full runs and smaller smoke tests.
    parser = argparse.ArgumentParser(
        description="Benchmark dense and graph autoencoders across timepoints."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        default="mosta",
        help="Named dataset preset.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional explicit .h5ad path. Overrides --dataset.",
    )
    parser.add_argument("--timepoint-key", default="timepoint")
    parser.add_argument("--label-key", default="annotation")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--output-root", default="./comp511")
    parser.add_argument(
        "--train-modes",
        nargs="+",
        #choices=["shared", "per_timepoint"],
        #default=["shared", "per_timepoint"],
        choices=["shared", "per_timepoint", "pca"],
        default=["shared", "per_timepoint", "pca"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["VAE"],
        help="Model names or aliases: VAE.",
        #default=["GAE", "VGAE"],
        #help="Model names or aliases: GAE, VGAE.",
        default=["AE", "GAE", "VAE", "VGAE"],
        help="Model names or aliases: AE, GAE, VAE, VGAE.",
    )
    parser.add_argument(
        "--graph-k",
        nargs="+",
        type=int,
        #default=[5, 10],
        default=[5, 10, 15, 20],
        help="Spatial graph k values for GAE/VGAE only.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        #default=list(range(5)),
        help="Random seeds for learned models.",
    )
    parser.add_argument(
        "--pca-seed",
        type=int,
        default=None,
        help="Seed used for the PCA baseline. Defaults to the first seed.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--hidden-units", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--cache-freq", type=int, default=10)
    parser.add_argument("--vae-loss-type", default="ziln")
    parser.add_argument("--vae-beta", type=float, default=0.00001)
    parser.add_argument("--leiden-resolution", type=float, default=0.3)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument(
        "--timepoints",
        nargs="+",
        default=None,
        help="Optional explicit list of timepoints to evaluate.",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=None,
        help="Optional cap on the number of timepoints after sorting/filtering.",
    )
    return parser.parse_args()


def set_global_seed(seed: int):
    # Match the repo's training behavior closely so repeated runs are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_dataset_path(args) -> Path:
    return Path(args.data_path or DATASET_PRESETS[args.dataset]).resolve()


def infer_dataset_tag(args) -> str:
    if args.data_path:
        return Path(args.data_path).stem
    return args.dataset


def canonical_model_names(model_names):
    resolved = []
    for model_name in model_names:
        key = MODEL_ALIASES.get(model_name, model_name)
        if key not in {"AE", "VAE", "GraphAE", "GraphVAE"}:
            raise ValueError(f"Unsupported model '{model_name}'.")
        resolved.append(key)
    return resolved


def sorted_timepoints(adata, timepoint_key, requested=None, max_timepoints=None):
    # Sort numeric-looking labels naturally (e.g. 2DPI, 5DPI, 10DPI).
    values = adata.obs[timepoint_key].astype(str).unique().tolist()

    def sort_key(value):
        digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
        if digits:
            try:
                return (0, float(digits), value)
            except ValueError:
                pass
        return (1, value)

    values = sorted(values, key=sort_key)
    if requested is not None:
        requested = [str(v) for v in requested]
        values = [v for v in values if v in requested]
    if max_timepoints is not None:
        values = values[: max(0, int(max_timepoints))]
    return values


def ensure_label_column(adata, label_key):
    if label_key not in adata.obs:
        raise KeyError(f"obs['{label_key}'] not found in dataset.")


def ensure_spatial_for_graph_models(adata, spatial_key, models, train_modes):
    if any(model in GRAPH_MODELS for model in models) and any(
        mode in {"shared", "per_timepoint"} for mode in train_modes
    ):
        if spatial_key not in adata.obsm:
            raise KeyError(f"obsm['{spatial_key}'] is required for graph models.")


def build_model(model_name, input_dim, args):
    # Reuse the existing model builders so the benchmark stays aligned with repo defaults.
    common_kwargs = dict(
        name=model_name,
        restore=None,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer="Adam",
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
    )
    if model_name in {"AE", "GraphAE"}:
        return load_autoencoder_model(use_gat=True, **common_kwargs)
    return load_vae_model(
        **common_kwargs,
        loss_type=args.vae_loss_type,
        beta=args.vae_beta
    )


def build_training_dataset(adata, model_name, args, graph_k=None):
    # Dense models ignore spatial data; graph models use the existing spatial graph path.
    is_graph = model_name in GRAPH_MODELS
    return build_cell_data(
        adata=adata,
        device=args.device,
        condition_key=args.timepoint_key,
        spatial=is_graph,
        source=None,
        target="all",
        groupby=args.timepoint_key,
        split="none",
        model_name=model_name,
        graph_spatial_key=args.spatial_key,
        graph_k=graph_k if graph_k is not None else 10,
    )


def build_training_loader(adata, dataset, model_name, args):
    if model_name in GRAPH_MODELS:
        return build_dataloaders(
            dataset,
            model_name=model_name,
            batch_size=adata.n_obs,
            shuffle_train=False,
        )
    return build_dataloaders(
        dataset,
        model_name=model_name,
        batch_size=args.batch_size,
        shuffle_train=True,
    )


def train_model(model_name, train_adata, outdir, args, graph_k=None):
    # Train one model instance and return it in-memory for immediate encoding/evaluation.
    dataset = build_training_dataset(train_adata, model_name, args, graph_k=graph_k)
    loader = build_training_loader(train_adata, dataset, model_name, args)
    model, optim = build_model(model_name, input_dim=train_adata.n_vars, args=args)

    trainer_kwargs = dict(
        model=model,
        optim=optim,
        loader=loader,
        outdir=outdir,
        n_iters=args.n_iters,
        eval_freq=args.eval_freq,
        cache_freq=args.cache_freq,
        device=args.device,
        num_clusters=None,
        labels=None,
        compute_ari=False,
        compute_nmi=False,
    )

    if model_name == "AE":
        train_encoder(**trainer_kwargs)
    elif model_name == "VAE":
        train_vae(**trainer_kwargs)
    elif model_name == "GraphAE":
        train_graph_encoder(
            **trainer_kwargs,
            num_neighbors=graph_k,
            batch_size=args.batch_size,
            seed_limit=args.batch_size,
        )
    elif model_name == "GraphVAE":
        train_graph_vae(
            **trainer_kwargs,
            num_neighbors=graph_k,
            batch_size=args.batch_size,
            seed_limit=args.batch_size,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    return model


def make_latent_adata(embedding, template_adata):
    # Wrap the embedding in AnnData so Scanpy can run neighbors/Leiden/UMAP directly on it.
    latent = sc.AnnData(X=np.asarray(embedding, dtype=np.float32), obs=template_adata.obs.copy())
    latent.uns = {}
    for key, value in template_adata.uns.items():
        if key.endswith("_colors"):
            latent.uns[key] = value
    return latent


def evaluate_embedding(
    embedding,
    adata_tp,
    seed,
    label_key,
    leiden_resolution,
    umap_neighbors,
    plot_path,
):
    # ARI/NMI follow the existing KMeans-based evaluation path in training/eval.py.
    labels = adata_tp.obs[label_key].astype("category")
    label_codes = labels.cat.codes.to_numpy()
    n_clusters = int(max(1, min(labels.nunique(), adata_tp.n_obs)))

    np.random.seed(seed)
    ari = calculate_ari(embedding, label_codes, n_clusters)
    np.random.seed(seed)
    nmi = calculate_nmi(embedding, label_codes, n_clusters)

    latent_adata = make_latent_adata(embedding, adata_tp)
    if latent_adata.n_obs >= 3:
        # Leiden is used only for visualization; metrics come from KMeans above.
        sc.pp.neighbors(
            latent_adata,
            n_neighbors=min(umap_neighbors, latent_adata.n_obs - 1),
        )
        sc.tl.leiden(
            latent_adata,
            resolution=leiden_resolution,
            random_state=seed,
            key_added="leiden",
        )
        sc.tl.umap(latent_adata, random_state=seed)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc.pl.umap(latent_adata, color=label_key, ax=axes[0], show=False, title="Annotation")
        sc.pl.umap(latent_adata, color="leiden", ax=axes[1], show=False, title="Leiden")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        leiden_clusters = int(latent_adata.obs["leiden"].astype(str).nunique())
    else:
        latent_adata.obs["leiden"] = pd.Categorical(["0"] * latent_adata.n_obs)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"UMAP skipped: only {latent_adata.n_obs} cells",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        leiden_clusters = int(latent_adata.obs["leiden"].astype(str).nunique())

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "n_clusters": n_clusters,
        "n_cells": int(adata_tp.n_obs),
        "n_label_classes": int(labels.nunique()),
        "leiden_clusters": leiden_clusters,
    }


def encode_timepoint(model, model_name, adata_tp, args, graph_k=None):
    # Encode each timepoint independently after training, as requested.
    latent_adata = encode_adata(
        model=model,
        adata=adata_tp,
        encoder_name=model_name,
        device=args.device,
        condition_key=args.timepoint_key,
        graph_spatial_key=args.spatial_key,
        graph_k=graph_k if graph_k is not None else 10,
        batch_size=args.batch_size,
    )
    return np.asarray(latent_adata.X, dtype=np.float32)


def pca_embedding(adata_tp, seed, latent_dim):
    # PCA is the non-neural baseline and never uses the spatial graph sweep.
    n_components = max(1, min(latent_dim, adata_tp.n_obs, adata_tp.n_vars))
    X = adata_tp.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    elif hasattr(X, "todense"):
        X = np.asarray(X.todense())
    else:
        X = np.asarray(X)
    X = np.asarray(X, dtype=np.float32)
    return PCA(n_components=n_components, random_state=seed).fit_transform(X)


def timepoint_subset(adata, timepoint_key, timepoint):
    mask = adata.obs[timepoint_key].astype(str) == str(timepoint)
    return adata[mask].copy()


def append_result(rows, **kwargs):
    row = dict(kwargs)
    rows.append(row)


def run_shared_models(adata, timepoints, models, args, outroot, rows):
    # Train one model across all timepoints, then evaluate each timepoint separately.
    for seed in args.seeds:
        set_global_seed(seed)
        for model_name in models:
            k_values = args.graph_k if model_name in GRAPH_MODELS else [None]
            for graph_k in k_values:
                run_name = f"seed{seed}"
                if graph_k is not None:
                    run_name += f"_k{graph_k}"
                run_dir = outroot / "shared" / model_name / run_name
                ckpt_dir = run_dir / "checkpoints"
                fig_dir = run_dir / "umaps"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                fig_dir.mkdir(parents=True, exist_ok=True)

                print(f"[shared] training {model_name} seed={seed} k={graph_k}")
                model = train_model(
                    model_name=model_name,
                    train_adata=adata,
                    outdir=ckpt_dir,
                    args=args,
                    graph_k=graph_k,
                )

                for timepoint in timepoints:
                    print("Timepoint:", timepoint)
                    adata_tp = timepoint_subset(adata, args.timepoint_key, timepoint)
                    if adata_tp.n_obs == 0:
                        continue
                    embedding = encode_timepoint(
                        model=model,
                        model_name=model_name,
                        adata_tp=adata_tp,
                        args=args,
                        graph_k=graph_k,
                    )
                    plot_path = fig_dir / f"{timepoint}.png"
                    metrics = evaluate_embedding(
                        embedding=embedding,
                        adata_tp=adata_tp,
                        seed=seed,
                        label_key=args.label_key,
                        leiden_resolution=args.leiden_resolution,
                        umap_neighbors=args.umap_neighbors,
                        plot_path=plot_path,
                    )
                    print(metrics)
                    append_result(
                        rows,
                        dataset=infer_dataset_tag(args),
                        train_mode="shared",
                        model=model_name,
                        seed=seed,
                        graph_k=graph_k,
                        timepoint=str(timepoint),
                        checkpoint_dir=str(ckpt_dir),
                        umap_path=str(plot_path),
                        **metrics,
                    )


def run_per_timepoint_models(adata, timepoints, models, args, outroot, rows):
    # Train a separate model for each timepoint and evaluate on that same timepoint.
    for seed in args.seeds:
        set_global_seed(seed)
        for timepoint in timepoints:
            print("Timepoint:", timepoint)
            adata_tp = timepoint_subset(adata, args.timepoint_key, timepoint)
            if adata_tp.n_obs == 0:
                continue
            for model_name in models:
                k_values = args.graph_k if model_name in GRAPH_MODELS else [None]
                for graph_k in k_values:
                    print("Model:", model_name, "Graph k:", graph_k)
                    run_name = f"seed{seed}"
                    if graph_k is not None:
                        run_name += f"_k{graph_k}"
                    run_dir = outroot / "per_timepoint" / model_name / str(timepoint) / run_name
                    ckpt_dir = run_dir / "checkpoints"
                    fig_dir = run_dir / "umaps"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    fig_dir.mkdir(parents=True, exist_ok=True)

                    print(f"[per_timepoint] training {model_name} seed={seed} tp={timepoint} k={graph_k}")
                    model = train_model(
                        model_name=model_name,
                        train_adata=adata_tp,
                        outdir=ckpt_dir,
                        args=args,
                        graph_k=graph_k,
                    )
                    embedding = encode_timepoint(
                        model=model,
                        model_name=model_name,
                        adata_tp=adata_tp,
                        args=args,
                        graph_k=graph_k,
                    )
                    plot_path = fig_dir / "umap.png"
                    metrics = evaluate_embedding(
                        embedding=embedding,
                        adata_tp=adata_tp,
                        seed=seed,
                        label_key=args.label_key,
                        leiden_resolution=args.leiden_resolution,
                        umap_neighbors=args.umap_neighbors,
                        plot_path=plot_path,
                    )
                    print(metrics)
                    append_result(
                        rows,
                        dataset=infer_dataset_tag(args),
                        train_mode="per_timepoint",
                        model=model_name,
                        seed=seed,
                        graph_k=graph_k,
                        timepoint=str(timepoint),
                        checkpoint_dir=str(ckpt_dir),
                        umap_path=str(plot_path),
                        **metrics,
                    )


def run_pca_baseline(adata, timepoints, args, outroot, rows):
    # PCA runs once on a single seed and does not iterate over graph k values.
    seed = args.pca_seed if args.pca_seed is not None else args.seeds[0]
    set_global_seed(seed)
    for timepoint in timepoints:
        adata_tp = timepoint_subset(adata, args.timepoint_key, timepoint)
        if adata_tp.n_obs == 0:
            continue
        run_dir = outroot / "pca" / str(timepoint) / f"seed{seed}"
        fig_dir = run_dir / "umaps"
        fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"[pca] evaluating tp={timepoint} seed={seed}")
        embedding = pca_embedding(adata_tp=adata_tp, seed=seed, latent_dim=args.latent_dim)
        plot_path = fig_dir / "umap.png"
        metrics = evaluate_embedding(
            embedding=embedding,
            adata_tp=adata_tp,
            seed=seed,
            label_key=args.label_key,
            leiden_resolution=args.leiden_resolution,
            umap_neighbors=args.umap_neighbors,
            plot_path=plot_path,
        )
        append_result(
            rows,
            dataset=infer_dataset_tag(args),
            train_mode="pca",
            model="PCA",
            seed=seed,
            graph_k=None,
            timepoint=str(timepoint),
            checkpoint_dir="",
            umap_path=str(plot_path),
            **metrics,
        )


def write_results(rows, outroot):
    # Save both the raw run table and a grouped summary for downstream analysis.
    outroot.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    if results_df.empty:
        results_df = pd.DataFrame(
            columns=[
                "dataset",
                "train_mode",
                "model",
                "seed",
                "graph_k",
                "timepoint",
                "ari",
                "nmi",
                "n_clusters",
                "n_cells",
                "n_label_classes",
                "leiden_clusters",
                "checkpoint_dir",
                "umap_path",
            ]
        )
    results_path = outroot / "metrics_per_run.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["dataset", "train_mode", "model", "graph_k", "timepoint"], dropna=False)[
            ["ari", "nmi", "n_clusters", "leiden_clusters"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary_df.columns
    ]
    summary_df.to_csv(outroot / "metrics_summary.csv", index=False)
    print(f"Saved per-run metrics to {results_path}")
    print(f"Saved summary metrics to {outroot / 'metrics_summary.csv'}")


def main():
    # The script keeps all outputs under ot2/comp511/<dataset_tag>/...
    args = parse_args()
    data_path = infer_dataset_path(args)
    outroot = (Path(__file__).resolve().parent / args.output_root).resolve()
    benchmark_root = outroot / infer_dataset_tag(args)

    print(f"Loading dataset from {data_path}")
    adata = sc.read_h5ad(str(data_path))
    if args.timepoint_key not in adata.obs:
        raise KeyError(f"obs['{args.timepoint_key}'] not found in dataset.")
    ensure_label_column(adata, args.label_key)

    models = canonical_model_names(args.models)
    ensure_spatial_for_graph_models(adata, args.spatial_key, models, args.train_modes)
    timepoints = sorted_timepoints(
        adata,
        timepoint_key=args.timepoint_key,
        requested=args.timepoints,
        max_timepoints=args.max_timepoints,
    )
    if not timepoints:
        raise ValueError("No timepoints selected for benchmarking.")

    rows = []

    if "shared" in args.train_modes:
        run_shared_models(adata, timepoints, models, args, benchmark_root, rows)
    if "per_timepoint" in args.train_modes:
        run_per_timepoint_models(adata, timepoints, models, args, benchmark_root, rows)
    if "pca" in args.train_modes:
        run_pca_baseline(adata, timepoints, args, benchmark_root, rows)

    write_results(rows, benchmark_root)


if __name__ == "__main__":
    main()
