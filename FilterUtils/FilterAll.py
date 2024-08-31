import numpy as np
import os
from grid import DensityGrid

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


for hashsize in range(14, 20):
    for scene in scenes:
        path = f"./Inputs/Hash{hashsize}/{scene}.msgpack"
        grid = DensityGrid(path)
        if (scene in ["ficus", "materials", "mic"]):
            THREDHOLD = 0.025
        else:
            THREDHOLD = 0.01
        
        grid.conv_filterate(THREDHOLD)
        grid.display(title = f"{scene}", save_fig=True)
        
        os.makedirs(f"./Outputs/Hash{hashsize}", exist_ok = True)
        grid.write_msgpack(f"./Outputs/Hash{hashsize}/{scene}.msgpack")
