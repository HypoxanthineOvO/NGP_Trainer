import numpy as np
import torch
import tinycudann as tcnn
import msgpack
import nerfacc
import math
from utils import inv_morton_naive, morton_naive


class InstantNGP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Consts
        self.near = 0.6
        self.far = 2.0
        self.steps = 1024
        self.step_length = math.sqrt(3) / self.steps
        
        # Initialize models
        self.hash: torch.nn.Module = tcnn.NetworkWithInputEncoding(
            n_input_dims = 3,
            n_output_dims = 16,
            encoding_config = config["encoding"],
            network_config = config["network"]
        ).to("cuda")
        self.sh: torch.nn.Module = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = config["dir_encoding"],
            dtype = torch.float32
        ).to("cuda")
        self.mlp: torch.nn.Module = tcnn.Network(
            n_input_dims = 32,
            n_output_dims = 3,
            network_config = config["rgb_network"]
        ).to("cuda")
        self.grid: torch.nn.Module = nerfacc.OccGridEstimator(
            roi_aabb = [0, 0, 0, 1, 1, 1],
            resolution = 128, levels = 1
        ).to("cuda")
        self.snapshot = None
        
        # Initialize Density Grid
        init_params_grid = {
            "resolution": torch.tensor([128, 128, 128], dtype = torch.int32),
            "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "occs": torch.rand(128 ** 3, dtype = torch.float32) / 250,
            "binaries": torch.zeros([1, 128, 128, 128], dtype = torch.bool)
        }
        self.grid.load_state_dict(init_params_grid)

    def load_snapshot(self, path: str):
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
            snapshot = next(unpacker)
        self.snapshot = snapshot
        params_binary = torch.tensor(
            np.frombuffer(snapshot["snapshot"]["params_binary"], 
                        dtype = np.float16, offset = 0)#.astype(np.float32)
            , dtype = torch.float32)
        # Params for Hash Encoding Network
        ## Network Params Size: 32 * 64 + 64 * 16 = 3072
        params_hashnet = params_binary[:(32 * 64 + 64 * 16)]
        params_binary = params_binary[(32 * 64 + 64 * 16):]
        # Params for RGB Network
        ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
        params_rgbnet = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
        params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
        # Params for Hash Encoding Grid
        params_hashgrid = params_binary
        # Params of Density Grid
        grid_raw = torch.tensor(
            np.frombuffer(
                snapshot["snapshot"]["density_grid_binary"], dtype=np.float16).astype(np.float32),
            dtype = torch.float32
            )
        grid = torch.zeros([128 * 128 * 128], dtype = torch.float32)

        x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
        grid[x * 128 * 128 + y * 128 + z] = grid_raw
        grid_3d = torch.reshape(grid > 0.01, [1, 128, 128, 128]).type(torch.bool)
        
        params_hash = torch.cat([params_hashnet, params_hashgrid])
        self.hash.load_state_dict({"params": params_hash})
        self.mlp.load_state_dict({"params": params_rgbnet})
        self.grid.load_state_dict({
            "resolution": torch.tensor([128, 128, 128], dtype = torch.int32),
            "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "occs": grid,
            "binaries": grid_3d
        })
    
    def save_snapshot(self, path: str, load_path: str | None = None):
        if load_path is not None:
            with open(load_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
                snapshot = next(unpacker)
        # Parameters
        params_hash = self.hash.state_dict()["params"].clone().cpu().detach()
        params_rgbnet = self.mlp.state_dict()["params"].clone().cpu().detach()
        params_binary = torch.cat([
            params_hash[:(32 * 64 + 64 * 16)],
            params_rgbnet,
            params_hash[(32 * 64 + 64 * 16):]
        ]).numpy()
        snapshot["snapshot"]["params_binary"] = np.float16(params_binary).tobytes()
        # Density Grids
        density_grid: torch.Tensor = self.grid.state_dict()["occs"].clone().cpu().detach().type(torch.float16)
        grid_morton = torch.zeros(128 ** 3, dtype = torch.float16)
        indexs = torch.arange(0, 128**3, 1)
        grid_morton[morton_naive(indexs // (128 * 128), (indexs % (128 * 128)) // 128, indexs % 128)] = density_grid
        snapshot["snapshot"]["density_grid_binary"] = grid_morton.detach().numpy().tobytes()
        
        # HyperParameters
        snapshot['encoding']['log2_hashmap_size'] = self.config["encoding"]["log2_hashmap_size"]
        snapshot['encoding']['n_levels'] = self.config["encoding"]["n_levels"]
        snapshot['encoding']['n_features_per_level'] = self.config["encoding"]["n_features_per_level"]
        snapshot['encoding']['base_resolution'] = self.config["encoding"]["base_resolution"]
        with open(path, 'wb') as f:
            f.write(msgpack.packb(snapshot))
    
    def get_alpha(self, x: torch.Tensor):
        hash_features = self.hash(x)
        alphas_raw = hash_features[..., 0]
        alphas = (1. - torch.exp(-torch.exp(alphas_raw.type(torch.float32)) * self.step_length))
        return alphas

    def get_density(self, x: torch.Tensor):
        hash_features = self.hash(x)
        alphas_raw = hash_features[..., 0]
        density = torch.exp(alphas_raw - 1)
        return density

    def get_rgb(self, x: torch.Tensor, dir: torch.Tensor):
        hash_features = self.hash(x)
        sh_features = self.sh((dir + 1) / 2)        
        features = torch.concat([hash_features, sh_features], dim = -1)
        rgbs_raw = self.mlp(features)
        rgbs = torch.sigmoid(rgbs_raw)
        return rgbs

    def forward(self, position, direction):
        alphas = self.get_alpha(position)
        rgbs = self.get_rgb(position, direction)
        return rgbs, alphas