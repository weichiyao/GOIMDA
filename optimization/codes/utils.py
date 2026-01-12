import torch
from torch import Tensor
import numpy as np 
import gc
import os, sys

import datetime


class SuppressPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self._original_stdout

def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def eta_func(i, emin, emax, T):
    if i < T:
        return emax-i*(emax-emin)/T
    else:
        return emin

def print_x(x:np.ndarray, decimal:int=3):
    format_s = '{:.'+str(decimal)+'f}'
    d=x.shape[-1]
    x = x.reshape(-1, d)
    return ','.join(('['+','.join([format_s]*d)+']').format(*k) for k in x)

def print_progress(nsample:int,
                   tol:float, 
                   train_tol:float,
                   xmin:np.ndarray,
                   auxname:int=None, 
                   aux:float=None, 
                   train_aux:float=None):
    if aux is None:
        print("Status: nsamples={:d}, curr_tol={:.3f} (training curr_tol={:.3f}), xmin={:s}".format(
            nsample, 
            tol,
            train_tol, 
            print_x(xmin)
        ))
    else:
        print("Status: nsamples={:d}, curr_tol={:.3f}, curr_{:s}={:.3f} (training curr_tol={:.3f}, curr_{:s}={:.3f}), xmin={:s}".format(
            nsample, 
            tol, auxname, aux, 
            train_tol, auxname, train_aux,
            print_x(xmin)
        ))

def vectorize(x:Tensor, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def generate_seed_according_to_time(n:int=1):    
    # Getting todays date and time using now() of datetime class
    current_date = datetime.datetime.now()

    # Using the strftime() of datetime class
    # which takes the components of date as parameter
    # %Y - year
    # %m - month
    # %d - day
    # %H - Hours
    # %M - Minutes
    # %S - Seconds 
    addon = np.random.choice(50000, size=(n,), replace=False)
    out = int(current_date.strftime("%m%d%H%M%S")) + addon 
    return out.tolist()

def zero_one_denormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized

def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean