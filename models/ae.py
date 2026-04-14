# Adapted from cellot

import torch
from torch import nn
from collections import namedtuple
from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv  
from torch_geometric.nn import GATConv


def build_optimizer(params, lr=1e-3, weight_decay=1.0e-5, optimizer="Adam", **kwargs):
    """
    Create an optimizer directly from arguments (no config).
    """
    opt_name = optimizer.lower()
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer}'")


def build_model(name, **model_kwargs):
    """
    Instantiate a model by name with direct kwargs.
    Example:
        model = build_model(
            "AE",
            input_dim=2630, latent_dim=50, hidden_units=[512, 512], beta=0.0, dropout=0.0
        )
    """
    name = str(name)
    if name == "AE":
        cls = AutoEncoder
    elif name == "GraphAE":
        cls = GraphAutoEncoder
    else:
        raise ValueError(f"Unknown model name '{name}'. Expected 'AE' or 'GraphAE'.")
    return cls(**model_kwargs)


def load_autoencoder_model(
    name,
    restore=None,
    lr=1e-3, 
    weight_decay=1e-5, 
    optimizer="Adam",
    map_location='cpu',
    **model_kwargs
):
    """
    Build model + optimizer without a config file. Optionally restore from a checkpoint
    that contains {'model_state', 'optim_state', 'code_means' (optional)}.

    Example:
        model, optim = load_autoencoder_model(
            name="AE",
            input_dim=2630, latent_dim=50, hidden_units=[512,512], beta=0.0, dropout=0.0,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-5},
            restore=None,
        )
    """
    model = build_model(name, **model_kwargs)
    optim = build_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, optimizer="Adam")

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore, map_location=map_location)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        # optional field used in your original code
        if hasattr(model, "code_means") and ("code_means" in ckpt):
            model.code_means = ckpt["code_means"]
        print("Model Restored!")

    return model, optim


def dnn(
    input_dim,
    output_dim,
    hidden_units=[512, 512],
    activation="ReLU",
    dropout=0.0,
    batch_norm=False,
    **kwargs,
):
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)

    # load activation
    Activation = getattr(nn, activation) if isinstance(activation, str) else activation

    layers = []
    # hidden stack
    for in_dim, out_dim in zip([input_dim] + hidden_units[:-1], hidden_units):
        layers.append(nn.Linear(in_dim, out_dim, **kwargs)) # layer
        if batch_norm:  # batch norm
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(Activation()) # activation
        if dropout and dropout > 0: # dropout
            layers.append(nn.Dropout(dropout))

    # output layer
    layers.append(nn.Linear(hidden_units[-1], output_dim))
    net = nn.Sequential(*layers)
    return net


def gcn_dnn(
    input_dim,
    output_dim,
    hidden_units=[512, 512],
    activation="ReLU",
    dropout=0.0,
    batch_norm=False,
    use_gat=True,
    **kwargs,
):
    """
    First layer: Graph layer (GATConv by default, or SAGEConv if use_gat=False)
    Followed by MLP layers to produce output_dim.
    Note: For PyG, the graph layer expects (x, edge_index) or (x, edge_index, edge_weight).
    """
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)

    Activation = getattr(nn, activation) if isinstance(activation, str) else activation

    layers = []

    # Graph layer (expects edge_index / SparseTensor at call time)
    if use_gat:
        layers.append(GATConv(input_dim, hidden_units[0], heads=1, dropout=dropout, **kwargs))    # GAT Convolution
    else:
        layers.append(SAGEConv(input_dim, hidden_units[0], **kwargs))    #Graph SAGE Convolution
        #layers.append(GCNConv(input_dim, hidden_units[0], **kwargs)) 
    layers.append(Activation()) 

    # Dense stack
    for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
        layers.append(nn.Linear(in_dim, out_dim, **kwargs))  # Layer
        if batch_norm:  # batch norm
            layers.append(nn.BatchNorm1d(out_dim)) 
        layers.append(Activation()) # activation
        if dropout and dropout > 0: # dropout 
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_units[-1], output_dim))
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    LossComps = namedtuple("AELoss", "mse reg")
    Outputs = namedtuple("AEOutputs", "recon z")

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units=[512, 512],
        beta=0.0,
        dropout=0.0,
        mse=None,
        **kwargs,
    ):
        super(AutoEncoder, self).__init__(**kwargs)

        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(input_dim, latent_dim, hidden_units, dropout=dropout)

        if decoder_net is None:
            assert hidden_units is not None
            decoder_net = self.build_decoder(input_dim, latent_dim, hidden_units, dropout=dropout)

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.mse = mse if mse is not None else nn.MSELoss(reduction="none")


    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(input_dim=input_dim, output_dim=latent_dim, hidden_units=hidden_units, **kwargs)
        return net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_units=hidden_units[::-1],    # reversed 
            **kwargs,
        )
        return net

    def encode(self, inputs, **kwargs):
        return self.encoder_net(inputs, **kwargs)

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def outputs(self, inputs, **kwargs):
        code = self.encode(inputs, **kwargs)
        recon = self.decode(code, **kwargs)
        return self.Outputs(recon, code)

    def loss(self, inputs, outputs, rank_weights):
        #TODO: ADD GRN Weights
        mse_per_gene = self.mse(outputs.recon, inputs)                 # (B, G)
        if rank_weights is not None:
            mse = (mse_per_gene * rank_weights).mean(dim=-1)      # (B,)
        else:
            mse = mse_per_gene.mean(dim=-1)
        reg = torch.norm(outputs.z, dim=-1) ** 2
        total = mse + self.beta * reg
        return total, self.LossComps(mse, reg)

    def forward(self, inputs, rank_weights=None, **kwargs):
        outs = self.outputs(inputs, **kwargs)
        loss, comps = self.loss(inputs, outs, rank_weights)
        return loss, comps, outs


class GraphAutoEncoder(nn.Module):
    LossComps = namedtuple("AELoss", "mse reg")
    Outputs = namedtuple("AEOutputs", "recon z")

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units= [512, 512],
        beta=0.0,
        dropout=0.0,
        mse=None,
        use_gat=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(
                input_dim, latent_dim, hidden_units, use_gat=use_gat, dropout=dropout
            )

        if decoder_net is None:
            assert hidden_units is not None
            decoder_net = self.build_decoder(input_dim, latent_dim, hidden_units, dropout=dropout)

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.mse = mse if mse is not None else nn.MSELoss(reduction="mean")
        self.use_gat = use_gat


    def build_encoder(self, input_dim, latent_dim, hidden_units, use_gat=True, **kwargs):
        return gcn_dnn(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_units=hidden_units,
            use_gat=use_gat,
            **kwargs,
        )

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        return dnn(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_units=hidden_units[::-1],    # Reversed 
            **kwargs,
        )

    def encode(self, x, edge_index, **kwargs):
        """
        Pass x through the encoder. If the first layer is a PyG conv (GAT/SAGE),
        it expects (x, edge_index) or (x, SparseTensor). We keep your original logic:
        treat the first module specially, then call others normally.
        """
        for layer in self.encoder_net:
            if isinstance(layer, (GATConv, SAGEConv, GCNConv)):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def outputs(self, inputs, edge_index, **kwargs):
        code = self.encode(inputs, edge_index, **kwargs)
        recon = self.decode(code, **kwargs)
        return self.Outputs(recon, code)

    def loss(self, inputs, outputs, rank_weights):
        mse_per_gene = self.mse(outputs.recon, inputs)                 # (B, G)
        if rank_weights is not None:
            #rank_w = 1.0 + 0.3 * (rank_weights - 1.0)
            mse = (mse_per_gene * rank_weights).mean(dim=-1)      # (B,)
            #mse = (mse_per_gene * 1).mean(dim=-1) 
        else:
            mse = mse_per_gene.mean(dim=-1)
        reg = torch.norm(outputs.z, dim=-1) ** 2
        total = mse + self.beta * reg
        return total, self.LossComps(mse, reg)

    def forward(self, inputs, edge_index, rank_weights=None, **kwargs):
        inputs = inputs.squeeze(1)
        outs = self.outputs(inputs, edge_index, **kwargs)
        loss, comps = self.loss(inputs, outs, rank_weights)
        return loss, comps, outs