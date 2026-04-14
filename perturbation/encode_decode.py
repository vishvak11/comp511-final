"""Thin inference adapters built on the shared training/eval data flow."""

from __future__ import annotations

import anndata
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from data.cell import AnnDataDataset, AnnDataGraphDataset, build_dataloaders
from data.utils import cast_loader_to_iterator

from utils.helpers import nest_dict


def _build_iterator(dataset, model_name: str, batch_size: int, cycle_mode: str = "all"):
    """Create a finite, non-shuffled iterator using the project data utilities."""
    print("batch size: ", batch_size)
    loader = build_dataloaders(
        dataset,
        model_name=model_name,
        batch_size=max(1, int(batch_size)),
        shuffle_train=False,
    )
    return cast_loader_to_iterator(loader, cycle_mode=cycle_mode)


def _latent_adata(latent: np.ndarray, template_adata) -> anndata.AnnData:
    """Wrap a latent matrix in AnnData so downstream stages keep one shared data abstraction."""
    return anndata.AnnData(
        X=np.asarray(latent, dtype=np.float32),
        obs=template_adata.obs.copy(),
        uns=template_adata.uns.copy(),
        obsm={key: np.asarray(value).copy() for key, value in template_adata.obsm.items()},
    )


def _build_neighbor_loader(iterator_split, num_neighbors: int, batch_size: int, shuffle: bool = False):
    """Mirror the graph-model test loader path used during training evaluation."""
    
    full_inputs, adj_sparse, indices = next(iterator_split)
    row, col, val = adj_sparse.coo()
    full_edge_index = torch.stack([row, col], dim=0)

    data = Data(x=full_inputs, edge_index=full_edge_index, edge_attr=val)
    neighbor_loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors],
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
    )
    return neighbor_loader, indices


def _encode_dense_adata(model, adata, encoder_name: str, device: str, batch_size: int) -> anndata.AnnData:
    """Encode AE/VAE inputs through the same dataset/loader flow used in evaluation."""
    dataset = AnnDataDataset(adata, spatial=False)
    iterator = _build_iterator(dataset, model_name=encoder_name, batch_size=batch_size, cycle_mode="none")
    latents: list[np.ndarray] = []

    with torch.no_grad():
        for batch_inputs, _ in iterator:
            inputs = batch_inputs.to(device=device, dtype=torch.float32)
            if encoder_name == "AE":
                z = model.encode(inputs)
            elif encoder_name == "VAE":
                z, _ = model.encode(inputs)
            else:
                raise ValueError(f"Unsupported dense encoder '{encoder_name}'.")
            latents.append(np.asarray(z.detach().cpu().numpy(), dtype=np.float32))

    if not latents:
        latent_dim = getattr(model, "latent_dim", 0)
        return _latent_adata(np.zeros((0, latent_dim), dtype=np.float32), adata)
    return _latent_adata(np.concatenate(latents, axis=0), adata)


def _encode_graph_adata(
    model,
    adata,
    encoder_name: str,
    device: str,
    condition_key: str,
    graph_spatial_key: str,
    graph_k: int,
    batch_size: int,
) -> anndata.AnnData:
    """Encode graph models through the same neighbor-sampled eval path used in training."""
    dataset = AnnDataGraphDataset(
        adata,
        group_key=condition_key,
        spatial=True,
        spatial_key=graph_spatial_key,
        k=graph_k,
    )
    dataset = nest_dict({"test": dataset}, as_dot_dict=True)
    iterator = _build_iterator(dataset, model_name=encoder_name, batch_size=adata.n_obs)
    neighbor_loader, indices = _build_neighbor_loader(
        iterator.test,
        num_neighbors=graph_k,
        batch_size=batch_size,
        shuffle=False,
    )

    latent = None
    with torch.no_grad():
        for batch in neighbor_loader:
            batch_inputs = batch.x.to(device).squeeze(1)
            sampled_edge_index = batch.edge_index.to(device)
            seed_count = int(batch.batch_size)
            seed_nodes = batch.n_id[:seed_count].detach().cpu().numpy()
            ordered_idx = indices[seed_nodes].detach().cpu().numpy().astype(int)

            if encoder_name == "GraphAE":
                z = model.encode(batch_inputs, sampled_edge_index)
            elif encoder_name == "GraphVAE":
                z, _ = model.encode(batch_inputs, sampled_edge_index)
            else:
                raise ValueError(f"Unsupported graph encoder '{encoder_name}'.")

            z_seed = np.asarray(z[:seed_count].detach().cpu().numpy(), dtype=np.float32)
            if latent is None:
                latent = np.zeros((adata.n_obs, z_seed.shape[1]), dtype=np.float32)
            latent[ordered_idx] = z_seed

    if latent is None:
        latent_dim = getattr(model, "latent_dim", 0)
        return _latent_adata(np.zeros((0, latent_dim), dtype=np.float32), adata)
    return _latent_adata(latent, adata)


def encode_adata(
    model,
    adata,
    encoder_name: str,
    device: str,
    condition_key: str,
    graph_spatial_key: str = "spatial",
    graph_k: int = 10,
    batch_size: int | None = None,
) -> anndata.AnnData:
    """Encode an AnnData subset using the shared training/eval data flow."""
    model = model.to(device)
    model.eval()
    batch_size = int(batch_size or adata.n_obs or 1)

    if encoder_name in {"AE", "VAE"}:
        return _encode_dense_adata(model, adata, encoder_name=encoder_name, device=device, batch_size=batch_size)
    if encoder_name in {"GraphAE", "GraphVAE"}:
        return _encode_graph_adata(
            model,
            adata,
            encoder_name=encoder_name,
            device=device,
            condition_key=condition_key,
            graph_spatial_key=graph_spatial_key,
            graph_k=graph_k,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported encoder '{encoder_name}'.")


def decode_latent(
    model,
    latent,
    encoder_name: str,
    device: str,
    batch_size: int | None = None,
) -> np.ndarray:
    """Decode latent batches through the same shared loader/iterator utilities."""
    model = model.to(device)
    model.eval()
    if isinstance(latent, anndata.AnnData):
        latent_adata = latent.copy()
    else:
        latent_adata = anndata.AnnData(X=np.asarray(latent, dtype=np.float32))
    latent_matrix = np.asarray(latent_adata.X, dtype=np.float32)
    if latent_matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    dataset = AnnDataDataset(latent_adata, spatial=False)
    iterator = _build_iterator(dataset, model_name="AE", batch_size=int(batch_size or latent_matrix.shape[0]), cycle_mode="none")
    decoded_batches: list[np.ndarray] = []

    with torch.no_grad():
        for z_batch, _ in iterator:
            decoded = model.decode(z_batch.to(device=device, dtype=torch.float32))
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            decoded_batches.append(np.asarray(decoded.detach().cpu().numpy(), dtype=np.float32))

    return np.concatenate(decoded_batches, axis=0)
