import torch
from torch.nn import Module
from .QuantUtils import Quantize

class QMLP(Module):
    def __init__(self, n_input_dims: int, n_output_dims: int, network_config: dict, Weight_Bits = 8, Feature_Bits = 8):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_neurons = network_config.get("n_neurons", 64)
        self.n_hidden_layers = network_config.get("n_hidden_layers", 1)
        
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.n_input_dims, self.n_neurons, bias = False)] + 
            [torch.nn.Linear(self.n_neurons, self.n_neurons, bias = False) for i in range(self.n_hidden_layers - 1)] + 
            [torch.nn.Linear(self.n_neurons, self.n_output_dims, bias = False)]
        )
        
        self.WeightBits = Weight_Bits
        self.FeatureBits = Feature_Bits
    def load_states(self, states: torch.Tensor):
        state_dict = {}
        for i, layer in enumerate(self.layers):
            if(i == 0):
                weight = states[:self.n_input_dims * self.n_neurons].reshape(
                    [self.n_neurons, self.n_input_dims]
                )
                state_dict[f"layers.{i}.weight"] = Quantize(weight, 4, 8)
                states = states[self.n_input_dims * self.n_neurons:]
            elif (i == self.n_hidden_layers):
                weight = states[:self.n_neurons * self.n_output_dims].reshape(
                    [self.n_output_dims, self.n_neurons]
                )
                state_dict[f"layers.{i}.weight"] = Quantize(weight, 4, 8)
                states = states[self.n_neurons * self.n_output_dims:]
            else:
                weight = states[:self.n_neurons * self.n_neurons].reshape(
                    [self.n_neurons, self.n_neurons]
                )
                state_dict[f"layers.{i}.weight"] = Quantize(weight, 4, 8)
                states = states[self.n_neurons * self.n_neurons:]
        self.load_state_dict(state_dict)
        
    def forward(self, inputs: torch.Tensor):
        x = Quantize(inputs, 4, 8)
        for i, layer in enumerate(self.layers):
            x = Quantize(layer(Quantize(x, 4, 8)), 4, 8)
            if(i < len(self.layers) - 1):
                x = torch.nn.functional.relu(x)
        return x