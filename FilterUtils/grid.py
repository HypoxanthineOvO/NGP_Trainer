import numpy as np
import matplotlib.pyplot as plt
import os, msgpack
from tqdm import tqdm, trange
from morton import *

class DensityGrid:
    def __init__(self, path: str):
        '''
        Initialize the Density Grid
        '''
        assert (os.path.isfile(path))
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw = False)
            self.__config = next(unpacker)
        grid_raw = torch.tensor(np.clip(
            np.frombuffer(self.__config["snapshot"]["density_grid_binary"],dtype=np.float16).astype(np.float32),
            0, 1), dtype = torch.float32)
        grid = torch.zeros([128 * 128 * 128], dtype = torch.float32)
        x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
        grid[x * 128 * 128 + y * 128 + z] = grid_raw
        self.grid3D = grid.reshape([128, 128, 128])
        
        
    def get_indexs(self, THRESHOLD = 0.01):
        '''
        Get indexs whose density larger than THRESHOLD
        return it in form ([x0,x1,...].[y0,y1,...],[z0,z1,...])
        '''
        xyz = torch.where(self.grid3D > THRESHOLD)
        return xyz
    
    def conv_filterate(self, THRESHOLD = 0.01):
        assert(len(self.grid3D.shape) == 3)
        size_x, size_y, size_z = self.grid3D.shape
        res_grid = torch.zeros(self.grid3D.shape)
        for i in trange(size_x):
            for j in range(size_y):
                for k in range(size_z):
                    if (i <= 1 or j <= 1 or k <= 1 or i >= size_x-1 or j >= size_x-2 or k >= size_x-2):
                        res_grid[i,j,k] = 0
                    else:
                        if(torch.mean(self.grid3D[i-1:i+2, j-1:j+2, k-1:k+2]) >= THRESHOLD):
                            res_grid[i,j,k] = np.clip(self.grid3D[i,j,k], 0, 1)
                        else:
                            res_grid[i,j,k] = 0
        self.grid3D = res_grid
        #torch.save(res_grid, "FilteredGrid.pt")

    
    def high_pass_filterate(self, THRESHOLD = 0.01):
        assert(len(self.grid3D.shape) == 3)
        size_x, size_y, size_z = self.grid3D.shape
        res_grid = torch.zeros(self.grid3D.shape)
        for i in range(size_x):
            for j in range(size_y):
                for k in range(size_z):
                    if (i <= 1 or j <= 1 or k <= 1 or i >= size_x-1 or j >= size_x-2 or k >= size_x-2):
                        res_grid[i,j,k] = 0
                    else:
                        if(self.grid3D[i,j,k] >= THRESHOLD):
                            res_grid[i,j,k] = np.clip(self.grid3D[i,j,k], 0, 1)
                        else:
                            res_grid[i,j,k] = 0
        self.grid3D = res_grid

    
        
    def display(self, title = "Density Grid", figure_size = [10, 10], dpi = 80, save_fig = False):
        fig = plt.figure(figsize = figure_size, dpi = dpi)
        x,y,z = self.get_indexs()
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        density = self.grid3D[x,y,z]
        ax = fig.add_subplot(projection= '3d')
        ax.set_xlim(0,127)
        ax.set_xlabel("Z")
        ax.set_ylim(0,127)
        ax.set_ylabel("X")
        ax.set_zlim(0,127)
        ax.set_zlabel("Y")
        pts = ax.scatter(z,x,y, c = density, cmap = "rainbow", s = 10)
        newx,newy,newz = [],[],[]
        for i in range(0, 128):
            for j in range(0, 128):
                for k in range(0, 128):
                    # Only Draw Edge lines
                    if (
                        (i == 0 and j == 0) or (j == 0 and k == 0) or (i == 0 and k == 0) or
                        (i == 0 and j == 127) or (j == 0 and k == 127) or (i == 0 and k == 127) or
                        (i == 127 and j == 0) or (j == 127 and k == 0) or (i == 127 and k == 0) or
                        (i == 127 and j == 127) or (j == 127 and k == 127) or (i == 127 and k == 127)
                    ):
                        newx.append(i)
                        newy.append(j)
                        newz.append(k)
        ax.scatter(newx, newy, newz, c = "black", s = 5)
        ax.axis("off")
        #fig.colorbar(pts)
        #ax.set_title(title, fontdict = {"size": 20, "color": "black"})
        if save_fig:
            plt.savefig(title + ".png")
        else:
            plt.show()
        

    def write_msgpack(self, path:str):
        density_grid = (self.grid3D > 0.01).type(torch.float16).reshape([128**3])
        grid_morton = torch.zeros([128**3], dtype = torch.float16)
        indexs = torch.arange(0, 128**3, 1)
        grid_morton[morton_naive(indexs // (128 * 128), (indexs % (128 * 128)) // 128, indexs % 128)] = density_grid
        self.__config["snapshot"]["density_grid_binary"] = grid_morton.detach().numpy().tobytes()
        with open(path,'wb') as f:
            f.write(msgpack.packb(self.__config, use_bin_type = True))
        print("Save Config in {}".format(path))
