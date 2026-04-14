from torch.utils.data import DataLoader
from utils.helpers import nest_dict, flat_dict
from torch.utils.data import DataLoader, Dataset
from itertools import groupby
from absl import logging
import torch

def cast_dataset_to_loader(dataset, model_name, dataloader_shuffle=True, batch_size=256, batch_size_target_factor=1.0, **kwargs):

    def sparse_collate_fn(batch):
        inputs = []
        indices = []
        adj = None

        for x, adj_row, idx in batch:
            adj = adj_row  # same adjacency object reused (as in your original)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(idx, torch.Tensor):
                idx = torch.tensor(idx, dtype=torch.int)
            inputs.append(x)
            indices.append(idx)

        inputs = torch.stack(inputs)  # [batch_size, feature_dim]
        indices = torch.stack(indices)
        return inputs, adj, indices

    # if it's a single Dataset, behave exactly like before
    if isinstance(dataset, Dataset):
        return DataLoader(
            dataset,
            shuffle=dataloader_shuffle,  # assume training if single
            collate_fn=sparse_collate_fn if model_name in ['GraphAE', 'GraphVAE'] else None,
            **kwargs
        )

    # flatten structure exactly like you had
    flat_dataset = flat_dict(dataset)

    # compute per-topgroup minimum batch size (unchanged)
    minimum_batch_size = {
        group: min(*map(lambda x: len(flat_dataset[x]), keys), batch_size)
        for group, keys in groupby(flat_dataset.keys(), key=lambda x: x.split(".")[0])
    }

    if model_name in ["neuralot", "neuralot_unb"]:
        scaling_factor = batch_size_target_factor
        if scaling_factor <= 1.0:
            factor = {"source": 1, "target": scaling_factor}
        else:
            factor = {"source": 1 / scaling_factor, "target": 1}

        final_bs = {
            key: int(max(1, minimum_batch_size[key.split(".")[0]] * factor.get(key.split(".")[1], 1)))
            for key in flat_dataset.keys()
        }
        print("Batch_sizes: ", final_bs)

        loader = nest_dict(
            {
                key: DataLoader(
                    val,
                    batch_size=final_bs[key],
                    shuffle=dataloader_shuffle if key.startswith("train") else False,
                    collate_fn=sparse_collate_fn if model_name in ['GraphAE', 'GraphVAE'] else None,
                    **kwargs
                )
                for key, val in flat_dataset.items()
            },
            as_dot_dict=True,
        )

    elif model_name in ['AE', 'GraphAE', "VAE", "GraphVAE"]:
        min_bs = min(minimum_batch_size.values())
        if 'batch_size' in kwargs and kwargs['batch_size'] != min_bs:
            logging.warn(f'Batch size adapted to {min_bs} due to dataset size.')

        loader = nest_dict({
            key: DataLoader(
                val,
                batch_size=minimum_batch_size[key.split('.')[0]],
                shuffle=dataloader_shuffle if key.startswith("train") else False,
                collate_fn=sparse_collate_fn if model_name in ['GraphAE', 'GraphVAE'] else None,
                **{k: v for k, v in kwargs.items() if k != "shuffle"}  # keep your original guard
            )
            for key, val in flat_dataset.items()
        }, as_dot_dict=True)

    return loader

"""
def cast_loader_to_iterator(loader, cycle_all=True):
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    # unchanged logic
    if isinstance(loader, DataLoader):
        print("LOADED")
        return cycle(loader) if cycle_all else loader

    iterator = {}
    flat = flat_dict(loader)

    for key, value in flat.items():
        # Cycle only if key starts with "train"
        if cycle_all and key.startswith("train"):
            iterator[key] = cycle(value)
        else:
            iterator[key] = value

    # Sanity check
    for value in flat.values():
        assert len(value) > 0

    # Re-nest back into dot-dict format
    iterator = nest_dict(iterator, as_dot_dict=True)
    return iterator"""

def cast_loader_to_iterator(loader, cycle_mode="all"):
    """
    Convert a dataloader or dict of dataloaders into iterators with optional cycling.

    Args:
        loader: DataLoader or nested dict of DataLoaders.
        cycle_mode (str): Determines cycling behavior.
            - "all": cycle all loaders (train, test, etc.)
            - "train": cycle only loaders whose keys start with 'train'
            - "none": do not cycle any loaders
    """
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    # Case 1: Single DataLoader
    if isinstance(loader, DataLoader):
        if cycle_mode == "none":
            return loader
        else:
            return cycle(loader)

    # Case 2: Dictionary of loaders
    iterator = {}
    flat = flat_dict(loader)

    for key, value in flat.items():
        if cycle_mode == "all":
            iterator[key] = cycle(value)
        elif cycle_mode == "train" and key.startswith("train"):
            iterator[key] = cycle(value)
        else:
            iterator[key] = value

    # Sanity check
    for k, value in flat.items():
        assert len(value) > 0, f"Loader '{k}' is empty!"

    # Re-nest dictionary back
    return nest_dict(iterator, as_dot_dict=True)


    """
    iterator = nest_dict(
        {key: (cycle(item) if cycle_all else item) for key, item in flat_dict(loader).items()}, as_dot_dict=True
    )

    for value in flat_dict(loader).values():
        assert len(value) > 0

    return iterator"""