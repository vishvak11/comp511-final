import torch
from torch import nn
from collections import namedtuple
from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv  
from torch_geometric.nn import GATConv
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch_geometric.utils import negative_sampling


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
            "VAE",
            input_dim=2630, latent_dim=50, hidden_units=[512, 512], beta=0.01, dropout=0.0
        )
    """
    name = str(name)
    if name == "VAE":
        cls = VariationalAutoEncoder
    elif name == "GraphVAE":
        cls = GraphVariationalAutoEncoder
    else:
        raise ValueError(f"Unknown model name '{name}'. Expected 'VAE' or 'GraphVAE'.")
    return cls(**model_kwargs)


def load_vae_model(
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
            name="VAE",
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


class VariationalAutoEncoder(nn.Module):
    # named tuple with fields mse and regularization
    LossComps = namedtuple("VAELoss", "recon kl total")

    # named tuple with fields reconstruction data and encoded representations 
    Outputs = namedtuple("VAEOutputs", "recon mu logvar z")
    
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units=[512, 512],
        loss_type="gaussian",
        beta=1.0,
        dropout=0,
        mse=None,
        **kwargs
    ):
        
        super(VariationalAutoEncoder, self).__init__(**kwargs)

        # Encoder outputs 2*latent_dim (mean and logvar)
        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        # Decoder: Gaussian or ZILN 
        if decoder_net is None:
            assert hidden_units is not None
            if loss_type == "gaussian":
                self.decoder_net = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
            elif loss_type == "ziln":
                self.decoder_mu = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_log_sigma = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_dropout = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
            elif loss_type == "zinb":
                self.decoder_mu = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_dropout = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                # NB-specific parameter
                self.log_theta = nn.Parameter(torch.randn(input_dim))

            elif loss_type == "lognormal":
                self.decoder_mu = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_log_sigma = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")


        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        #self.decoder_net = decoder_net
        self.loss_type = loss_type


    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            input_dim=input_dim,output_dim=latent_dim*2, hidden_units=hidden_units, **kwargs
        )
        return net
    

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_units=hidden_units[::-1],  # reversed 
            **kwargs
        )
        return net


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encode(self, inputs, **kwargs):
        h = self.encoder_net(inputs, **kwargs)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar
    

    def decode(self, z, **kwargs):
        if self.loss_type == "gaussian":
            return self.decoder_net(z)
        elif self.loss_type == "ziln":
            mu = self.decoder_mu(z)
            log_sigma = self.decoder_log_sigma(z)
            dropout_logits = self.decoder_dropout(z)
            return mu, log_sigma, dropout_logits
        elif self.loss_type == "zinb":
            mu = self.decoder_mu(z)
            dropout_logits = self.decoder_dropout(z)
            return mu, dropout_logits
        elif self.loss_type == "lognormal":
            mu = self.decoder_mu(z)
            log_sigma = self.decoder_log_sigma(z)
            return mu, log_sigma


    def _apply_gene_weights(self, per_gene_loss, rank_weights):
        if rank_weights is not None:
            return (per_gene_loss * rank_weights).sum(dim=-1)
        else:
            return per_gene_loss.sum(dim=-1)

    
    def gaussian_recon_loss(self, x, recon_x):
        return ((x - recon_x) ** 2)   # (B, G) per-gene NLL


    def ziln_recon_loss(self, x, mu, log_sigma, dropout_logits):
        sigma = torch.exp(log_sigma)
        pi = torch.sigmoid(dropout_logits)
        lognorm = torch.distributions.LogNormal(mu, sigma)

        log_prob_nonzero = torch.log(1 - pi + 1e-8) + lognorm.log_prob(x + 1e-8)

        # FIX: use tensor for CDF input
        eps = torch.full_like(x, 1e-8)
        log_prob_zero = torch.log(pi + (1 - pi) * lognorm.cdf(eps) + 1e-8)

        is_zero = (x < 1e-8).float()
        log_prob = is_zero * log_prob_zero + (1 - is_zero) * log_prob_nonzero

        return -log_prob     # (B, G) per-gene NLL


    def zinb_recon_loss(self, x, mu, dropout_logits):
        eps = 1e-8

        # Ensure all parameters are valid
        x = torch.clamp(x, min=0.0)                                # counts must be ≥ 0
        mu = torch.clamp(mu, min=eps, max=1e6)

        theta = nn.functional.softplus(self.log_theta)
        nb_logits = (mu + 1e-5).log() - (theta + 1e-5).log()

        dist = ZeroInflatedNegativeBinomial(
            total_count=theta,
            logits=nb_logits,
            gate_logits=dropout_logits,
            validate_args=False
        )

        log_prob = dist.log_prob(x)
        return -log_prob  # (B, G) per-gene NLL


    def lognormal_recon_loss(self, x, mu, log_sigma):
        sigma = torch.exp(log_sigma)                   # convert log_sigma to sigma
        lognorm = torch.distributions.LogNormal(mu, sigma)
        log_prob = lognorm.log_prob(x + 1e-8)          # avoid log(0) issues
        return -log_prob                     # negative log-likelihood


    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


    def forward(self, inputs, rank_weights=None, **kwargs):
        mu, logvar = self.encode(inputs, **kwargs)
        z = self.reparameterize(mu, logvar)

        if self.loss_type == "gaussian":
            recon = self.decode(z)
            per_gene = self.gaussian_recon_loss(inputs, recon)
        elif self.loss_type == "ziln":
            mu_dec, log_sigma, dropout_logits = self.decode(z)
            per_gene = self.ziln_recon_loss(inputs, mu_dec, log_sigma, dropout_logits)
            recon = (mu_dec, log_sigma, dropout_logits)  # structured recon for ZILN
        elif self.loss_type == "zinb":
            mu_dec, dropout_logits = self.decode(z)
            per_gene = self.zinb_recon_loss(inputs, mu_dec, dropout_logits)
            recon = (mu_dec, dropout_logits)
        elif self.loss_type == "lognormal":
            mu_dec, log_sigma = self.decode(z)                # decode latent z
            per_gene = self.lognormal_recon_loss(inputs, mu_dec, log_sigma)  # compute loss
            recon = (mu_dec, log_sigma) 

        # apply GRN rank weights if they exist
        recon_loss = self._apply_gene_weights(per_gene, rank_weights)        # (B,)
                
        kl = self.kl_divergence(mu, logvar)
        total_loss = (recon_loss + self.beta * kl).mean()

        comps = self.LossComps(recon=recon_loss.mean(), kl=kl.mean(), total=total_loss)
        
        # Use VAEOutputs for clarity
        outputs = self.Outputs(recon=recon, mu=mu, logvar=logvar, z=z)  # ADDED
        
        return total_loss, comps, outputs


class GraphVariationalAutoEncoder(nn.Module):
    # named tuple with fields mse and regularization
    LossComps = namedtuple("VAELoss", "recon kl total") #new graph

    # named tuple with fields reconstruction data and encoded representations 
    Outputs = namedtuple("VAEOutputs", "recon mu logvar z")

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units=None,
        loss_type="gaussian",
        beta=1.0,
        dropout=0,
        use_gat=True,
        **kwargs
    ):
        super(GraphVariationalAutoEncoder, self).__init__(**kwargs)

    # Encoder outputs 2*latent_dim (mean and logvar)
        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(
                input_dim, latent_dim, hidden_units, use_gat=use_gat, dropout=dropout
            )

        # Decoder: Gaussian or ZILN 
        if decoder_net is None:
            assert hidden_units is not None
            if loss_type == "gaussian":
                self.decoder_net = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
            elif loss_type == "ziln":
                self.decoder_mu = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_log_sigma = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_dropout = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
            elif loss_type == "zinb":
                self.decoder_mu = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_dropout = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                # NB-specific parameter
                self.log_theta = nn.Parameter(torch.randn(input_dim))

            elif loss_type == "lognormal":
                self.decoder_mu = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )
                self.decoder_log_sigma = self.build_decoder(
                    input_dim, latent_dim, hidden_units, dropout=dropout
                )

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")


        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        #self.decoder_net = decoder_net
        self.loss_type = loss_type
        self.use_gat = use_gat


    def build_encoder(self, input_dim, latent_dim, hidden_units, use_gat, **kwargs):
        net = gcn_dnn(
            input_dim=input_dim, 
            output_dim=latent_dim*2, 
            hidden_units=hidden_units, 
            use_gat=use_gat, 
            **kwargs
        )
        return net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_units=hidden_units[::-1],
            **kwargs
        )
        return net

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # new graph
    def _edge_index_from_adj(self, adj, num_nodes):
        """Accepts edge_index or torch_sparse.SparseTensor; returns edge_index on device."""
        if isinstance(adj, torch.Tensor):
            # Assume already edge_index (2, E)
            if adj.dim() == 2 and adj.size(0) == 2:
                return adj
        # Try PyG SparseTensor
        row = col = None
        if hasattr(adj, "coo"):
            row, col, _ = adj.coo()
        elif isinstance(adj, tuple) and len(adj) == 2:
            # (row, col)
            row, col = adj
        if row is None or col is None:
            raise ValueError("adj must be edge_index (2,E) or a SparseTensor with .coo()")
        return torch.stack([row, col], dim=0)


    def encode(self, inputs, adj, **kwargs):
        x = inputs
        for layer in self.encoder_net:
            #if isinstance(layer, SAGEConv):
            if isinstance(layer, (GATConv, SAGEConv, GCNConv)):  # <<< CHANGED FOR GAT >>>
                x = layer(x, adj)
            else:
                x = layer(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar


    def decode(self, z, **kwargs):
        if self.loss_type == "gaussian":
            return self.decoder_net(z)
        elif self.loss_type == "ziln":
            mu = self.decoder_mu(z)
            log_sigma = self.decoder_log_sigma(z)
            dropout_logits = self.decoder_dropout(z)
            return mu, log_sigma, dropout_logits
        elif self.loss_type == "zinb":
            mu = self.decoder_mu(z)
            dropout_logits = self.decoder_dropout(z)
            return mu, dropout_logits
        elif self.loss_type == "lognormal":
            mu = self.decoder_mu(z)
            log_sigma = self.decoder_log_sigma(z)
            return mu, log_sigma


    def _apply_gene_weights(self, per_gene_loss, rank_weights):
        if rank_weights is not None:
            return (per_gene_loss * rank_weights).sum(dim=-1)
        else:
            return per_gene_loss.sum(dim=-1)

    def gaussian_recon_loss(self, x, recon_x):
        return ((x - recon_x) ** 2)#.sum(dim=1)


    def ziln_recon_loss(self, x, mu, log_sigma, dropout_logits):
        sigma = torch.exp(log_sigma)
        pi = torch.sigmoid(dropout_logits)
        lognorm = torch.distributions.LogNormal(mu, sigma)

        log_prob_nonzero = torch.log(1 - pi + 1e-8) + lognorm.log_prob(x + 1e-8)

        # FIX: use tensor for CDF input
        eps = torch.full_like(x, 1e-8)
        log_prob_zero = torch.log(pi + (1 - pi) * lognorm.cdf(eps) + 1e-8)

        is_zero = (x < 1e-8).float()
        log_prob = is_zero * log_prob_zero + (1 - is_zero) * log_prob_nonzero

        return -log_prob#.sum(dim=1)


    def zinb_recon_loss(self, x, mu, dropout_logits):
        eps = 1e-8

        # Ensure all parameters are valid
        x = torch.clamp(x, min=0.0)                                # counts must be ≥ 0
        mu = torch.clamp(mu, min=eps, max=1e6)

        theta = nn.functional.softplus(self.log_theta)
        nb_logits = (mu + 1e-5).log() - (theta + 1e-5).log()

        dist = ZeroInflatedNegativeBinomial(
            total_count=theta,
            logits=nb_logits,
            gate_logits=dropout_logits,
            validate_args=False
        )

        log_prob = dist.log_prob(x)
        return -log_prob#.sum(dim=1)


    def lognormal_recon_loss(self, x, mu, log_sigma):
        sigma = torch.exp(log_sigma)                   # convert log_sigma to sigma
        lognorm = torch.distributions.LogNormal(mu, sigma)
        log_prob = lognorm.log_prob(x + 1e-8)          # avoid log(0) issues
        return -log_prob#.sum(dim=1)                    # negative log-likelihood


    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


    def forward(self, inputs, adj, rank_weights=None, **kwargs):
        mu, logvar = self.encode(inputs, adj, **kwargs)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)

        if self.loss_type == "gaussian":
            recon = self.decode(z)
            per_gene = self.gaussian_recon_loss(inputs, recon)
        elif self.loss_type == "ziln":
            mu_dec, log_sigma, dropout_logits = self.decode(z)
            log_sigma = torch.clamp(log_sigma, min=-10, max=10)
            per_gene = self.ziln_recon_loss(inputs, mu_dec, log_sigma, dropout_logits)
            recon = (mu_dec, log_sigma, dropout_logits)  # structured recon for ZILN
        elif self.loss_type == "zinb":
            mu_dec, dropout_logits = self.decode(z)
            per_gene = self.zinb_recon_loss(inputs, mu_dec, dropout_logits)
            recon = (mu_dec, dropout_logits)
        elif self.loss_type == "lognormal":
            mu_dec, log_sigma = self.decode(z)                # decode latent z
            per_gene = self.lognormal_recon_loss(inputs, mu_dec, log_sigma)  # compute loss
            recon = (mu_dec, log_sigma) 
                
        # apply GRN rank weights if they exist
        recon_loss = self._apply_gene_weights(per_gene, rank_weights)        # (B,)

        kl = self.kl_divergence(mu, logvar)

        total_loss = (recon_loss + self.beta * kl).mean()

        comps = self.LossComps(recon=recon_loss.mean(), kl=kl.mean(), total=total_loss)
        
        # Use VAEOutputs for clarity
        outputs = self.Outputs(recon=recon, mu=mu, logvar=logvar, z=z)  # ADDED
        
        return total_loss, comps, outputs    