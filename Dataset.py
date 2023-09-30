import os
import json
import cv2 as cv
import numpy as np
from tqdm import trange
import torch
import nerfacc
from utils import nerf_matrix_to_ngp, inv_morton_naive

class NeRFSynthetic:
    def __init__(self, data_dir):
        self.scale = 0.33
        self.offset = [0.5, 0.5, 0.5]
        images = []
        transform_matrixs = []
        split = {}
        for data_type in ["train", "test", "val"]:
            print(f"Loading {data_type.capitalize()} Data...")
            with open(os.path.join(data_dir, f"transforms_{data_type}.json"), "r") as f:
                meta = json.load(f)
            
            split[data_type] = len(meta["frames"])
            for i in trange(len(meta["frames"])):
                frame = meta["frames"][i]
                file_name = os.path.join(data_dir, frame["file_path"] + ".png")
                image_raw = cv.imread(file_name, cv.IMREAD_UNCHANGED)
                image_raw = cv.cvtColor(image_raw, cv.COLOR_BGRA2RGBA)
                image = image_raw / 255.
                image = image[..., :3] * image[..., 3:]
                transform_matrix = nerf_matrix_to_ngp(
                    torch.tensor(frame["transform_matrix"]), self.scale, self.offset
                )
                images.append(image)
                transform_matrixs.append(transform_matrix)
        self.split = split
        images_np = np.array(images)
        self.images = torch.tensor(images_np, dtype = torch.float32)
        
        self.transform_matrixs = torch.tensor(np.array(transform_matrixs))[:, :3, :]
        
        H, W = self.images.shape[1:3]
        camera_angle = float(meta["camera_angle_x"])
        self.focal = torch.tensor([
            0.5 * W / np.tan(0.5 * camera_angle),
            0.5 * H / np.tan(0.5 * camera_angle)
        ])
        self.principal = torch.tensor([0.5, 0.5])
        self.HW = torch.tensor([H, W])
    def test(self, test_id = 0):
        id = self.split["train"] + test_id
        
        ref_images = self.images[id]
        c2w = self.transform_matrixs[id]
        w_ids, h_ids = np.meshgrid(
            np.linspace(0, self.HW[1]-1, self.HW[1]), 
            np.linspace(0, self.HW[0]-1, self.HW[0]), 
            indexing='xy'
        )
        w_ids = w_ids.reshape([-1,])
        h_ids = h_ids.reshape([-1,])
        hw_ids = np.random.randint(0, 800 * 800, (800 * 800, 1))
        w_id = torch.tensor(w_ids[hw_ids], dtype = torch.float32)
        h_id = torch.tensor(h_ids[hw_ids], dtype = torch.float32)
        ray_o, ray_d = torch.vmap(
            self.generate_ray,
            in_dims = (None, 0, 0, None, None, None)
        )(c2w, w_id, h_id, self.focal, self.principal, self.HW)
        return ref_images, ray_o, ray_d
    
    def sample(self, batch_size):
        image_id = torch.tensor(np.random.randint(0, self.images.shape[0]))
        transform_matrixs = self.transform_matrixs[image_id]#.expand([batch_size] + list(self.transform_matrixs[0].shape))
        image = self.images[image_id].reshape([-1, 3])
        
        w_ids, h_ids = np.meshgrid(
            np.linspace(0, self.HW[1]-1, self.HW[1]), 
            np.linspace(0, self.HW[0]-1, self.HW[0]), 
            indexing='xy'
        )
        w_ids = w_ids.reshape([-1,])
        h_ids = h_ids.reshape([-1,])
        hw_ids = np.random.randint(0, 800 * 800, (batch_size, 1))
        w_id = torch.tensor(w_ids[hw_ids], dtype = torch.float32)
        h_id = torch.tensor(h_ids[hw_ids], dtype = torch.float32)
        
        pixels = image[hw_ids].reshape([batch_size, 3])
        
        ray_o, ray_d = torch.vmap(
            self.generate_ray,
            in_dims = (None, 0, 0, None, None, None)
        )(transform_matrixs, w_id, h_id, self.focal, self.principal, self.HW)
        
        return pixels, ray_o, ray_d
    


    @staticmethod
    def generate_ray(transform_matrix, x, y, focal, principal, hw):
        x = (x + 0.5) / hw[0]
        y = (y + 0.5) / hw[1]
        ray_o = transform_matrix[:, 3]
        ray_d_x: torch.Tensor = (x - principal[0]) * hw[0] / focal[0]
        ray_d_y: torch.Tensor = (y - principal[1]) * hw[1] / focal[1]
        ray_d_z: torch.Tensor = torch.ones_like(ray_d_x)
        ray_d = torch.cat([ray_d_x, ray_d_y, ray_d_z], dim = -1)
        ray_d = torch.matmul(transform_matrix[:3, :3], ray_d)
        ray_d = ray_d / torch.norm(ray_d)
        return (ray_o, ray_d)