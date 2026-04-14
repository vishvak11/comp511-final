from .train_neuralot import train_neuralot
from .train_encoder import train_encoder
from .train_graph_encoder import train_graph_encoder
#from .train_vae import train_vae
#from .train_graph_vae import train_graph_vae

FUNS = {
    "neuralot": train_neuralot,
    "neuralot_unb": train_neuralot,
    "AE": train_encoder,
    "GraphAE": train_graph_encoder,
    "VAE": train_encoder,
    "GraphVAE": train_graph_encoder,
}