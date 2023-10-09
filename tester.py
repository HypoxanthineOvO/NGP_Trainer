import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
import cv2 as cv

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def PSNR(path1, name):
    path2 = f"./data/nerf_synthetic/{name}/test/r_0.png"
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    return compute_psnr(cv.resize(img1, (800, 800)), img2)
def Show_Diff(name:str):
    path1 = f"./ExampleOutputs/Test_{name}.png"
    path2 = f"./data/nerf_synthetic/{name}/test/r_0.png"
    img1 = cv.resize(np.array(cv.imread(path1) / 255., dtype=np.float32), (800, 800))
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    diff = np.abs(img1 - img2)
    plt.imsave(f"Diff_{name}.png",diff)


if __name__ == "__main__":
    psnr = PSNR("lego.png", "lego")
    print(psnr)
    exit()
    psnrs = []
    for scene in scenes:
        psnr = PSNR(f"./Test_{scene}_0_ACC.png", scene)
        print(psnr)
        psnrs.append(psnr)
    print(round(np.mean(psnrs), 4))
