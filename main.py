import numpy as np
import torch
import json
import math
import cv2 as cv
from tqdm import trange, tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import argparse

from Dataset import NeRFSynthetic
from NGP import InstantNGP
from utils import Camera, render_image

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type = str, default = "lego")
parser.add_argument("--config", type = str, default = "base")
parser.add_argument("--max_steps", type = int, default = 25000)


# HyperParameters
config_path = "./configs/base.json"
scene_name = "lego"
max_steps = 5000
batch_size = 2048

# Consts
near = 0.6
far = 2.0
ngp_steps = 1024
step_length = math.sqrt(3) / ngp_steps

if __name__ == "__main__":
    with open(config_path, "r") as f:
        config = json.load(f)
    # Camera Parameters
    with open(f"./data/nerf_synthetic/{scene_name}/transforms_test.json", "r") as f:
        meta = json.load(f)
    m_Camera_Angle_X = float(meta["camera_angle_x"])
    m_C2W = np.array(meta["frames"][0]["transform_matrix"]).reshape(4, 4)
    camera = Camera((800, 800), m_Camera_Angle_X, m_C2W)
    ref_raw = cv.imread(f"./data/nerf_synthetic/{scene_name}/test/r_0.png", cv.IMREAD_UNCHANGED) / 255.
    ref_raw = ref_raw[..., :3] * ref_raw[..., 3:]
    ref = np.array(ref_raw, dtype=np.float32)
    # Datasets
    dataset = NeRFSynthetic(f"./data/nerf_synthetic/{scene_name}")
    
    # Initialize models
    ngp = InstantNGP(config).to("cuda")
    #ngp.load_snapshot("./snapshots/lego.msgpack")
    weight_decay = (
        1e-5 if scene_name in ["materials", "ficus", "drums"] else 1e-6
    )
    optimizer = torch.optim.Adam(
        ngp.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
    )
    # Train Utils
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[ max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10,],
            gamma=0.33,
    ),])
    
    # Training
    ngp.train()
    ngp.grid.train()
    for step in trange(max_steps + 1):
        def occ_eval_fn(x):
            density = ngp.get_density(x)
            return density * step_length
        ngp.grid.update_every_n_steps(step = step, occ_eval_fn = occ_eval_fn, occ_thre = 1e-2)
        
        pixels, rays_o, rays_d = dataset.sample(batch_size)
        pixels = pixels.cuda()
        color = render_image(
            ngp, ngp.grid, rays_o, rays_d
        )
        loss = torch.nn.functional.smooth_l1_loss(color, pixels)
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()
        
        # Eval
        if step % 1000 == 0:
            total_color = np.zeros([800 * 800, 3], dtype = np.float32)
            val_batch = 100 * 100
            for i in range(0, 800*800, val_batch):
                rays_o_total = torch.tensor(camera.rays_o[i: i+val_batch], dtype = torch.float32)
                rays_d_total = torch.tensor(camera.rays_d[i: i+val_batch], dtype = torch.float32)
                color = render_image(
                    ngp, ngp.grid, rays_o_total, rays_d_total,
                ).cpu().detach().numpy()
                total_color[i: i+val_batch] = color
                torch.cuda.empty_cache()
            image = np.clip(total_color[..., [2, 1, 0]].reshape(800, 800, 3), 0, 1)

            cv.imwrite(f"{scene_name}.png", image * 255.)
            psnr = compute_psnr(image, ref)
            tqdm.write(f"Step {step}, PSNR = {round(psnr.item(), 4)}")
        torch.cuda.empty_cache()
    ngp.save_snapshot(path = f"./{scene_name}.msgpack", load_path = "./snapshots/lego.msgpack")