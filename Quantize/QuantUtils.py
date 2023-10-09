import torch
from torch.autograd import Variable
import math
import warnings


warnings.filterwarnings("ignore")

def Get_int_Part(input, overflow_rate):
    """
    input: tensor that need to compute
    overflow_rate: overflow_rate after quantize
    """
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def Get_ScaleFactor_From_int_Part(bits, sf_int):
    return bits - 1. - sf_int

def Compute_Scale_Factor(input, bits, ov=0.0):
    sfind = Get_int_Part(input, overflow_rate=ov)
    sfd = Get_ScaleFactor_From_int_Part(bits=bits, sf_int=sfind)
    return sfd

def Quantize_with_ScaleFactor(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    
    original_value = input
    output = (clipped_value - original_value).detach() + original_value
    
    return output

def Linear_Quantize(input: torch.Tensor, bits, ov=0.0):
    sf = Compute_Scale_Factor(input, bits, ov)
    quant_t = Quantize_with_ScaleFactor(input, sf, bits)
    torch.cuda.empty_cache()
    return quant_t