import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
from math import ceil

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_sparse import SparseTensor


from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

#from models.ae import load_autoencoder_model
#from models.vae import load_vae_model
from data.utils import cast_dataset_to_loader
from utils.helpers import nest_dict

class AnnDataDataset(Dataset):
    """
    Minimal tensor dataset for AnnData.
    Optionally returns spatial coords and/or a categorical obs label.
    """
    def __init__(self, adata, spatial=False, obs=None, categories=None, include_index=False):
        self.adata = adata.copy()
        # Ensure float32 matrix
        X = self.adata.X
        if sparse.issparse(X):
            X = X.todense()
        self.adata.X = np.asarray(X, dtype=np.float32)

        self.obs = obs
        self.categories = categories
        self.include_index = include_index
        self.spatial = spatial

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        gene_expression = self.adata.X[idx]

        # spatial if present and requested
        spatial_coords = None
        if self.spatial and ('spatial' in self.adata.obsm):
            spatial_coords = self.adata.obsm['spatial'][idx]

        # optional obs label -> integer category index
        if self.obs is not None:
            meta_val = self.adata.obs[self.obs].iloc[idx]
            meta = self.categories.index(meta_val) if self.categories is not None else meta_val
            if spatial_coords is not None:
                return gene_expression, spatial_coords, int(meta)
            else:
                return gene_expression, int(meta)

        if self.include_index:
            if spatial_coords is not None:
                return self.adata.obs_names[idx], (gene_expression, spatial_coords)
            else:
                return self.adata.obs_names[idx], gene_expression

        if spatial_coords is not None:
            return gene_expression, spatial_coords
        else:
            return gene_expression, idx


class AnnDataGraphDataset(Dataset):
    """
    Graph flavored dataset that builds a group-wise KNN adjacency (source/target) from spatial coords.
    Returns (feature, full_adjacency, idx) to be compatible with prior downstream code.
    """
    def __init__(self, adata, group_key, obs=None, categories=None, include_index=False, spatial = True,
                 spatial_key="spatial", k=10, self_loop=False, normalize_weights=True, metric="euclidean"):
        self.adata = adata.copy()
        X = self.adata.X
        if sparse.issparse(X):
            X = X.todense()
        self.X = np.asarray(X, dtype=np.float32)

        self.obs = obs
        self.categories = categories
        self.include_index = include_index
        self.n_cells = self.X.shape[0]

        self.adj = self._build_adjacency(
            self.adata, spatial_key=spatial_key, group_key=group_key, k=k,
            self_loop=self_loop, normalize_weights=normalize_weights, metric=metric
        )

    def _build_adjacency(self, adata, spatial_key, group_key, k=10, self_loop=False, 
                        normalize_weights=True, metric="euclidean"):
        if spatial_key not in adata.obsm:
            raise ValueError(f"'{spatial_key}' not found in adata.obsm")

        if group_key not in adata.obs:
            raise ValueError(f"'{group_key}' column not found in adata.obs.")

        coords = adata.obsm[spatial_key]
        groups = adata.obs[group_key].astype("category")
        N = coords.shape[0]

        row_list, col_list, weight_list = [], [], []
        for g in groups.cat.categories:
            mask = (groups.values == g)
            group_coords = coords[mask]
            group_indices = np.where(mask)[0]
            if group_coords.shape[0] <= k:
                continue

            nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(group_coords)
            distances, indices = nbrs.kneighbors(group_coords)

            row_idx = np.repeat(group_indices, k)
            col_idx = group_indices[indices.flatten()]
            # Avoid div-by-zero
            weights = 1.0 / (distances.flatten() + 1e-8)

            if normalize_weights:
                row_sums = np.zeros(N, dtype=np.float64)
                np.add.at(row_sums, row_idx, weights)
                weights = weights / (row_sums[row_idx] + 1e-8)

            row_list.append(row_idx)
            col_list.append(col_idx)
            weight_list.append(weights)

        if len(row_list) == 0:
            # no edges (edge case: tiny groups)
            return SparseTensor.sparse_diag(torch.ones(N))

        row = torch.tensor(np.concatenate(row_list), dtype=torch.long)
        col = torch.tensor(np.concatenate(col_list), dtype=torch.long)
        val = torch.tensor(np.concatenate(weight_list), dtype=torch.float32)

        adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N, N))
        adj = adj.fill_diag(1.0) if self_loop else adj.remove_diag()
        return adj

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        feature = self.X[idx]
        adj_all = self.adj  # full adjacency (kept same as original)
        if self.obs is not None:
            label_val = self.adata.obs[self.obs].iloc[idx]
            label = self.categories.index(label_val) if self.categories is not None else label_val
            feature = (feature, int(label))

        if self.include_index:
            return self.adata.obs_names[idx], feature, adj_all

        return feature, adj_all, idx


def label_cohort_as_train_or_eval(samples, random_state=0, holdout=None, trainset=None, evalset=None):
    if trainset is None:
        trainset = []
    elif isinstance(trainset, str):
        trainset = [trainset]
    assert isinstance(trainset, list)

    if evalset is None:
        evalset = []
    elif isinstance(evalset, str):
        evalset = [evalset]
    assert isinstance(evalset, list)

    assert not np.in1d(trainset, evalset).any()
    cohort = pd.Series("train", samples)

    if not holdout:
        return cohort

    if holdout < 1:
        holdout = ceil(len(cohort) * holdout)

    nremove = holdout - len(evalset)
    assert nremove >= 0

    decide = [x for x in samples if (x not in evalset) and (x not in trainset)]
    rng = np.random.default_rng(random_state)
    evalset.extend(rng.choice(decide, nremove, replace=False).tolist())

    cohort.loc[evalset] = "eval"
    return cohort


def split_cell_data_none(adata):
    split = pd.Series("train_test", index=adata.obs.index, dtype=object)
    return split


def split_cell_data_train_test(adata, groupby=None, holdout=None, test_size=0.02, random_state=0):
    split = pd.Series(None, index=adata.obs.index, dtype=object)
    groups = {None: adata.obs.index} if groupby is None else adata.obs.groupby(groupby).groups

    for _, index in groups.items():
        trainobs, testobs = train_test_split(index, test_size=test_size, random_state=random_state)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            # update cells in holdout to ood
            split.loc[adata.obs[key].isin(value)] = "ood"

    return split


def split_cell_data_train_test_eval(adata, test_size=0.0, eval_size=0.2, groupby=None, holdout=None, random_state=0, **kwargs):
    split = pd.Series(None, index=adata.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            # updates cells that are in the holdout to ood
            split.loc[adata.obs[key].isin(value)] = "ood"

    groups = {None: adata.obs.index} if groupby is None else adata.obs.groupby(groupby).groups

    for _, index in groups.items():
        training, evalobs = train_test_split(index, random_state=random_state, test_size=eval_size)
        trainobs, testobs = train_test_split(training, random_state=random_state, test_size=test_size)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"
    return split


def split_cell_data_train_test_eval2(
    adata,
    test_size=0.0,
    eval_size=0.2,
    groupby=None,
    holdout=None,
    random_state=0,
    **kwargs,
):
    split = pd.Series(None, index=adata.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[adata.obs[key].isin(value)] = "ood"

    groups = {None: adata.obs.index} if groupby is None else adata.obs.groupby(groupby).groups

    for _, index in groups.items():
        if eval_size <= 0:
            trainobs = index
            evalobs = []
        else:
            trainobs, evalobs = train_test_split(
                index, random_state=random_state, test_size=eval_size
            )

        split.loc[trainobs] = "train"
        if len(evalobs) > 0:
            split.loc[evalobs] = "eval"

    return split



def split_cell_data_toggle_ood(adata, holdout, key, mode, random_state=0, **kwargs):
    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """
    split = split_cell_data_train_test(adata, random_state=random_state, **kwargs)
    value = holdout if isinstance(holdout, list) else [holdout]
    ood_idx = adata.obs_names[adata.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood_idx, random_state=random_state, test_size=0.5)
    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"
    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"
    else:
        raise ValueError("mode must be 'ood' or 'iid'")
    return split


def split_cell_data(adata, name="none", groupby=None, test_size=0.02, holdout=None, random_state=42, eval_size=0.2):
    #TODO: NEED TO UPDTE the last two approaches
    if name == "train_test":
        split = split_cell_data_train_test(adata, groupby=groupby, holdout=holdout, test_size=test_size, random_state=random_state)
    elif name == "none":
        split = split_cell_data_none(adata)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(adata, groupby, test_size, random_state)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval2(adata, groupby=groupby, test_size=test_size, 
                                                    eval_size=eval_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown split mode: {name}")
    return split.astype("category")


def compute_ari(embeddings, adata, label_key="annotation"):
    if label_key not in adata.obs:
        print(f"[compute_ari] '{label_key}' not found in adata.obs. Skipping ARI.")
        return
    labels = adata.obs[label_key].astype("category").cat.codes.to_numpy()
    print(labels)
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=0)
    pred_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, pred_labels)
    print(f"ARI: {ari:.4f}")


def _ensure_dense_float32(adata: anndata.AnnData):
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def apply_embedding( 
    adata: anndata.AnnData,
    graph_group_key,
    device='cpu',
    embedding_type: str | None = None,
    embedding_model=None,               # pre-initialized model object (preferred)
    embedding_path: str | Path | None = None,  # or a path we can load from
    graph_spatial_key="spatial",
    graph_k=10,
    graph_self_loop=True,
    graph_normalize=True,
    ari_label_key="annotation"
):
    """
    Apply AE/VAE/GraphAE/GraphVAE encoders (optional).
    Choose ONE of (embedding_model, embedding_path). If both are None, returns adata unchanged.
    """
    if embedding_type is None or (embedding_model is None and embedding_path is None):
        return adata  # silently do nothing

    adata = _ensure_dense_float32(adata)
    model = embedding_model.to(device)

    if embedding_type in ["AE", "VAE"]:
        inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
        with torch.no_grad():
            if embedding_type == "AE":
                Z = model.encode(inputs)
            elif embedding_type == "VAE":
                mu, logvar = model.encode(inputs)
                Z = mu
        compute_ari(Z, adata, label_key=ari_label_key)
        out = anndata.AnnData(np.asarray(Z.detach().cpu().numpy(), dtype=np.float32),
                              obs=adata.obs.copy(), uns=adata.uns.copy(), obsm=adata.obsm.copy())
        return out

    if embedding_type in ["GraphAE", "GraphVAE"]:
        if graph_spatial_key not in adata.obsm:
            raise ValueError(f"{embedding_type} requires adata.obsm['{graph_spatial_key}']")
        if graph_group_key not in adata.obs:
            raise ValueError(f"'{graph_group_key}' column not found in adata.obs.")
        coords = adata.obsm[graph_spatial_key]
        groups = adata.obs[graph_group_key].astype("category")
        N = coords.shape[0]

        rows, cols, vals = [], [], []
        for g in groups.cat.categories:
            mask = (groups.values == g)
            group_coords = coords[mask]
            group_idx = np.where(mask)[0]
            if group_coords.shape[0] <= graph_k:
                continue
            nbrs = NearestNeighbors(n_neighbors=graph_k, metric="euclidean").fit(group_coords)
            d, ind = nbrs.kneighbors(group_coords)
            r = np.repeat(group_idx, graph_k)
            c = group_idx[ind.flatten()]
            w = 1.0 / (d.flatten() + 1e-8)
            if graph_normalize:
                row_sums = np.zeros(N)
                np.add.at(row_sums, r, w)
                w = w / (row_sums[r] + 1e-8)
            rows.append(r); cols.append(c); vals.append(w)

        if not rows:
            adj = SparseTensor.sparse_diag(torch.ones(N))
        else:
            row = torch.tensor(np.concatenate(rows), dtype=torch.long)
            col = torch.tensor(np.concatenate(cols), dtype=torch.long)
            val = torch.tensor(np.concatenate(vals), dtype=torch.float32)
            adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N, N))
            adj = adj.fill_diag(1.0) if graph_self_loop else adj.remove_diag()

        r, c, v = adj.coo()
        edge_index = torch.stack([r, c], dim=0)

        inputs = torch.tensor(adata.X, dtype=torch.float32)
        pyg_data = Data(x=inputs, edge_index=edge_index, edge_attr=v)

        neighbor_loader = NeighborLoader(pyg_data, num_neighbors=[graph_k], batch_size=256, shuffle=False)
        model.eval()
        outs = []
        with torch.no_grad():
            for batch in neighbor_loader:
                if embedding_type == "GraphAE":
                    enc_all = model.encode(batch.x.to(device), batch.edge_index.to(device))
                    enc_seed = enc_all[:batch.batch_size]
                elif embedding_type == "GraphVAE":
                    mu, logvar = model.encode(batch.x, batch.edge_index)
                    enc_seed = mu[:batch.batch_size]
                outs.append(enc_seed.cpu())
        Z = torch.cat(outs, dim=0)
        compute_ari(Z, adata, label_key=ari_label_key)
        return anndata.AnnData(np.asarray(Z.detach().cpu().numpy(), dtype=np.float32),
                               obs=adata.obs.copy(), uns=adata.uns.copy(), obsm=adata.obsm.copy())

    raise ValueError(f"Unknown embedding type: {embedding_type}") 


def prepare_data(
    adata, 
    condition,                 # e.g., "timepoint"
    source,                    # e.g., "E15.5" or ["E15.5", "E14.5"]
    target,                    # e.g., "E16.5" or ["E16.5"] or "all"
    groupby,
    split,
    test_size=0.2,
    random_state=0, 
    eval_size=0.2,
    # anything extra for split_cell_data (e.g. holdout=...) can go inside datasplit
):
    """
    Config-free version of the original read_single_anndata.

    Returns
    -------
    filtered_adata : anndata.AnnData
        Filtered AnnData with:
          - obs["transport"] set to "source"/"target"
          - obs["split"] set if datasplit is provided
          - obsm["spatial"] preserved/aligned if present
    """
    # 1) build transport mapping from condition/source/target
    src_vals = source if isinstance(source, list) else [source]
    if target == "all":
        adata.obs["transport"] = "target"
        adata.obs.loc[adata.obs[condition].isin(src_vals), "transport"] = "source"
    else:
        tgt_vals = target if isinstance(target, list) else [target]
        mapper = {v: "source" for v in src_vals}
        mapper.update({v: "target" for v in tgt_vals})
        adata.obs["transport"] = adata.obs[condition].map(mapper)

    # 2) create base mask: drop rows with NaN transport
    mask = adata.obs["transport"].notna()

    # 3) preserve spatial alignment explicitly (like your original)
    spatial_data = None
    if "spatial" in adata.obsm:
        spatial_data = adata.obsm["spatial"][mask.values].copy()

    # 4) optionally write split column
    if groupby is not None and split is not None:
        # IMPORTANT: compute split on the unmasked adata (as in your original),
        # then we will apply mask for the returned object.
        adata.obs["split"] = split_cell_data(
            adata,
            name=split,
            groupby=groupby,
            test_size=test_size,
            random_state=random_state,
            eval_size=eval_size,
        )


    # 5) finalize filtering
    filtered = adata[mask.values].copy()
    if spatial_data is not None:
        filtered.obsm["spatial"] = spatial_data

    print(filtered.n_obs)
    return filtered


def build_cell_data(
    adata, 
    condition_key: str, 
    spatial: bool,
    source, 
    target,
    split, 
    model_name,
    device='cpu',
    groupby=None,
    test_size=0.2,
    random_state=0,
    eval_size=0.2,
    embedding_type: str | None = None, 
    embedding_model=None,                      # model object (preferred)
    embedding_path: str | Path | None = None,  # OR a path we can load from 
    # Graph / spatial knobs (used by GraphAE/VAE)
    dim = 50,
    graph_spatial_key: str = "spatial",
    graph_k: int = 10,
    graph_self_loop: bool = True,
    graph_normalize: bool = True,
    # ARI label
    ari_label_key: str = "annotation"
):
    if model_name in ["AE", 'GraphAE', "VAE", "GraphVAE"]:
        target = "all"
        source = None

    adata = prepare_data(
        adata, 
        condition_key,                 
        source,                    
        target,                    
        groupby,
        split,
        test_size,
        random_state,
        eval_size
        )

    adata = apply_embedding(
        adata,
        device=device,
        embedding_type=embedding_type,
        embedding_model=embedding_model,
        embedding_path=embedding_path,
        graph_group_key=condition_key,
        graph_spatial_key=graph_spatial_key,
        graph_k=graph_k, 
        graph_self_loop=graph_self_loop, 
        graph_normalize=graph_normalize,
        ari_label_key=ari_label_key
    )

    dataset_args = {}

    if model_name in ["neuralot", "neuralot_unb", "nubot", "nubot_spatial"]:
        split_on = ["split", "transport"]
    elif model_name in ["AE", 'GraphAE', "VAE", "GraphVAE"]:
        split_on = ["split"]
    else:
        raise ValueError

    
    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        print(key)
        assert key in adata.obs.columns

    prefer_graph = (model_name in ["GraphAE", "GraphVAE"])
    print("split_on: ", split_on)

    if len(split_on) > 0:
        # groups the data indices based on the unique combinations of attributes specified in split_on
        #splits = {
        #    (str(key) if isinstance(key, tuple) else key): data[index]
        #    for key, index in data.obs[split_on].groupby(split_on).groups.items()
        #}

        splits = {
            (key if isinstance(key, str) else ".".join(key)): adata[index]
            for key, index in adata.obs[split_on].groupby(split_on).groups.items()
        }
        
        # nested dictionary (splits) where keys are the unique combinations of attributes from split_on
        if prefer_graph:
            dataset_cls = AnnDataGraphDataset   
            dataset_args.update(
                dict(
                    spatial_key=graph_spatial_key,
                    k=graph_k,
                    normalize_weights=graph_normalize,
                    self_loop=graph_self_loop,
                    group_key=condition_key,
                )
            ) 
        else:
            dataset_cls = AnnDataDataset

        print(dataset_cls)
        print(splits.keys())
        if 'train_test.source' in splits.keys():
            val = splits['train_test.source']
            val2 = splits['train_test.target']
            dataset = nest_dict(
        {
            "train.source": dataset_cls(val, spatial=False, **dataset_args),
            "train.target": dataset_cls(val2, spatial=False, **dataset_args),
            "test.source": dataset_cls(val, spatial=spatial, **dataset_args),
            "test.target": dataset_cls(val2, spatial=spatial, **dataset_args),
        },
        as_dot_dict=True,)
            
        elif {'train.source', 'train.target'}.issubset(splits.keys()):
            dataset_items = {
                "train.source": dataset_cls(splits["train.source"], spatial=False, **dataset_args),
                "train.target": dataset_cls(splits["train.target"], spatial=False, **dataset_args),
            }

            if 'eval.source' in splits and 'eval.target' in splits:
                dataset_items["eval.source"] = dataset_cls(
                    splits["eval.source"], spatial=spatial, **dataset_args
                )
                dataset_items["eval.target"] = dataset_cls(
                    splits["eval.target"], spatial=spatial, **dataset_args
                )

            full_source = adata[adata.obs["transport"] == "source"]
            full_target = adata[adata.obs["transport"] == "target"]

            dataset_items["test.source"] = dataset_cls(full_source, spatial=spatial, **dataset_args)
            dataset_items["test.target"] = dataset_cls(full_target, spatial=spatial, **dataset_args)

            dataset = nest_dict(dataset_items, as_dot_dict=True)

        elif 'train_test' in splits.keys():
            val = splits['train_test']
            ds = dataset_cls(val, spatial=False, **dataset_args)
            dataset = nest_dict(
        {
            "train": ds,
            "test": ds,
        },
        as_dot_dict=True,)
        else:
            dataset = nest_dict(
                {
                    key: dataset_cls(val, spatial=False,**dataset_args)
                    for key, val in splits.items()
                },
                as_dot_dict=True,
            )

        # Loop through the nested dictionary
        """
        if config.model.name in ["neuralot", "neuralot_unb"]:
            for split_key, split_val in dataset.items():
                for data_key, adata_dataset in split_val.items():
                    #Assuming adata_dataset contains an AnnData object and can be saved using .adata
                    path = config.data.graph_ae_emb.path
                    base_path = Path(path).parent
                    if config.model.name == 'neuralot':
                        adata_dataset.adata.write(f"{base_path}/model-neuralot/{split_key}_{data_key}_dataset.h5ad")
                    else: 
                        adata_dataset.adata.write(f"{base_path}/model-neuralot_unb/{split_key}_{data_key}_dataset.h5ad")#"""

    else:
        dataset = AnnDataDataset(adata, spatial=spatial, **dataset_args)
        #dataset.adata.write("test.h5ad")

    return dataset


def build_dataloaders(
    dataset,
    model_name,
    batch_size=256,
    shuffle_train=True,
    #weighted_sampling: bool = False,
    #weight_key: str | None = None,
    #replacement: bool = True,
):
    """Create PyTorch DataLoaders for a dict of datasets."""
    loader = cast_dataset_to_loader(
        dataset,
        model_name=model_name,
        batch_size=batch_size,
        dataloader_shuffle=shuffle_train,
        # NEW ↓↓↓ forward them
        #weighted_sampling=weighted_sampling,
        #weight_key=weight_key,
        #replacement=replacement,
    )
    return loader
