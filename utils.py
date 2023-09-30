import numpy as np
import torch
import nerfacc
import math

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0.5, 0.5, 0.5]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def render_image(
    ngp_model: torch.nn.Module,
    grid: nerfacc.OccGridEstimator,
    rays_o_total: torch.Tensor, rays_d_total: torch.Tensor,
    near: float = 0.6, far: float = 2.0, step_size: float = math.sqrt(3) / 1024,
    batch_size = 10000
):
    num_pixels = rays_o_total.shape[0]
    rays_o = rays_o_total.cuda()
    rays_d = rays_d_total.cuda()
    def alpha_fn(t_starts, t_ends, ray_indices):
        origins = rays_o[ray_indices]
        directions = rays_d[ray_indices]
        ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
        positions = origins + directions * ts
        alphas = ngp_model.get_alpha(positions)
        return alphas
    
    def rgb_alpha_fn(t_starts, t_ends, ray_indices):
        origins = rays_o[ray_indices]
        directions = rays_d[ray_indices]
        ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
        positions = origins + directions * ts
        
        rgbs, alphas = ngp_model(positions, directions)
        return rgbs, alphas    

    
    ray_indices, t_starts, t_ends = grid.sampling(
        rays_o, rays_d, near_plane = near, far_plane = far, 
        #alpha_fn = alpha_fn,
        render_step_size = step_size
    )
    if(ray_indices.shape[0] <= 0):
        return torch.zeros([num_pixels, 3]).cuda()
        #continue

    color, opacity, depth, extras = nerfacc.rendering(
        t_starts, t_ends, ray_indices, 
        n_rays = num_pixels, rgb_alpha_fn = rgb_alpha_fn
    )
    return color

def Part_1_By_2(x: torch.tensor):
    x &= 0x000003ff;                 # x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x

def morton_naive(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    return Part_1_By_2(x) + (Part_1_By_2(y) << 1) + (Part_1_By_2(z) << 2)

def morton(input):
    return morton_naive(input[..., 0], input[..., 1], input[..., 2])

def inv_Part_1_By_2(x: torch.tensor):
    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >>16) | x) & 0x000003FF
    return x

def inv_morton_naive(input: torch.tensor):
    x = input &        0x09249249
    y = (input >> 1) & 0x09249249
    z = (input >> 2) & 0x09249249
    
    return inv_Part_1_By_2(x), inv_Part_1_By_2(y), inv_Part_1_By_2(z)

def inv_morton(input:torch.tensor):
    x,y,z = inv_morton_naive(input)
    return torch.stack([x,y,z], dim = -1)



def get_ray(x, y, hw, transform_matrix, focal, principal = [0.5, 0.5]):
    x = (x + 0.5) / hw[0]
    y = (y + 0.5) / hw[1]
    ray_o = transform_matrix[:3, 3]
    ray_d = np.array([
        (x - principal[0]) * hw[0] / focal,
        (y - principal[1]) * hw[1] / focal,
        1.0,
    ])
    ray_d = np.matmul(transform_matrix[:3, :3], ray_d)
    ray_d = ray_d / np.linalg.norm(ray_d)
    return ray_o, ray_d

class Camera:
    def __init__(self, resolution, camera_angle, camera_matrix):
        # Resolution: For Generate Image
        self.resolution = resolution
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.image = np.zeros((resolution[0] * resolution[1], 3)) # RGB Image
        # Parameters
        self.position = np.array([0.0, 0.0, 0.0])
        self.camera_to_world = np.zeros((3, 3))
        self.focal_length = 1.0

        # Camera Coordinate Directions
        self.directions = None
        # Rays Origin and Direction
        self.rays_o = np.zeros((resolution[0], resolution[1], 3))
        self.rays_d = np.zeros((resolution[0], resolution[1], 3))

        assert camera_matrix.shape == (3, 4) or camera_matrix.shape == (4, 4)
        if(camera_matrix.shape == (4, 4)):
            camera_matrix = camera_matrix[:3]

        self.position = camera_matrix[:3, -1]
        self.camera_to_world = camera_matrix[:3, :3]
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.focal_length = .5 * self.w / np.tan(.5 * camera_angle)
        # Generate Directions
        i, j = np.meshgrid(
            np.linspace(0, self.w-1, self.w), 
            np.linspace(0, self.h-1, self.h), 
            indexing='xy'
        )
        ngp_mat = nerf_matrix_to_ngp(camera_matrix)

        rays_o, rays_d = [], []
        for i in range(self.h):
            for j in range(self.w):
                ro, rd = get_ray(j, i, [self.h, self.w], ngp_mat, self.focal_length)
                rays_o.append(ro)
                rays_d.append(rd)
        
        self.rays_o = np.array(rays_o).reshape((-1, 3))
        self.rays_d = np.array(rays_d).reshape((-1, 3))
