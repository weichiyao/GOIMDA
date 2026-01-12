# https://github.com/xwshen51/engression/blob/main/engression-python/engression/models.py
from typing import Callable, Tuple 
import time
 

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader 
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar

import numpy as np
from jax import jit, numpy as jnp
from jax.tree_util import Partial, tree_map
from flax.core.frozen_dict import FrozenDict, freeze
 
from utils import print_x, gc_cuda, vectorize
from influence_max.active_optimization.opt_data_module import OptDataModule
from influence_max.active_optimization.opt_model_module import preprocess, sto_parameter_reconstruct 
from influence_max.global_optimizer import global_optimization

import os 
import math
 

def get_xmin_from_net(
        model_fn        : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
        model_params    : FrozenDict, 
        search_domain   : jnp.ndarray, 
        nstart          : int=100,
        method          : str="grid-search",
        optimize_method : str='trust-constr',
        use_double      : bool=False,
        **kwargs
    ) -> jnp.ndarray: 
    fun_to_opt = Partial(model_fn, model_params, **kwargs)
     
    xmin, _ = global_optimization(
        fun_to_opt      = fun_to_opt,
        method          = method,
        search_domain   = search_domain,  
        nstart          = nstart, 
        optimize_method = optimize_method,   
        use_double      = use_double
    )
    return xmin

def get_train_y_stats(n, y_savedir, do_normalize_y:bool=False):
    ymean = 0.
    ystd  = 1.

    if do_normalize_y:
        ymean = []
        yvar  = [] 
         
        for ii in range(n):
            output_filename = 'i{:04d}_output-train.pt'.format(ii)
            output_filepath = os.path.join(y_savedir, output_filename)
            ymean.append(torch.load(output_filepath)['ymean'])
            yvar.append(torch.load(output_filepath)['yvar'])
    
        ymean = torch.stack(ymean).sum()/n
        ystd  = torch.sqrt(torch.stack(yvar).sum()/n)
    else:
        ymean = 0.
        ystd = 1.
    return ymean, ystd

def get_train_y_for_all_x(x:Tensor, selected_base_indices:LongTensor, y_savedir:str, ymean=0., ystd=1.):
    selected_y = []
    n = x.shape[0]
    for ii in range(n):
        output_filename = 'i{:04d}_output-train.pt'.format(ii)
        output_filepath = os.path.join(y_savedir, output_filename)
        output = torch.load(output_filepath)
        yy = output['y'][selected_base_indices]
        selected_y.append(yy)
    selected_y = torch.stack(selected_y, dim=0) # (n, n_select)
    # do_normalization_y
    selected_y = (selected_y - ymean) / ystd
    return selected_y

def get_precomputed_base_x_embedding(y_savedir, selected_base_indices:torch.LongTensor=None, data:str="train"):
    ## Obtain the corresponding precomputed base_x_embedding values 
    # if (self.y_savedir is not None) and (selected_base_indices is not None):
    #     output_filepath = os.path.join(self.y_savedir, f'precomputed_featemb-{data}.pt')
    #     assert os.path.isfile(output_filepath), f"Pretrained feature embedding cannot be found in path {output_filepath}"
    #     selected_base_x_embedding = torch.load(output_filepath)[selected_base_indices]
    # else: 
    #     raise ValueError(f"When precomputed_base_x_embedding = True, cannot find it in y_savedir={self.y_savedir}")
    output_filepath = os.path.join(y_savedir, f'precomputed_featemb-{data}.pt')
    selected_base_x_embedding = torch.load(output_filepath)[selected_base_indices]
    return selected_base_x_embedding # (n_select, d_base)

def train_pl_model( 
    x:Tensor,
    search_domain:Tensor,
    base_dm:pl.LightningDataModule, 
    base_x_embedding_fn:nn.Module=None, 
    base_x_embedding_dim:int=512,
    train_y_fn:Callable[[np.ndarray], np.ndarray]=None,
    train_y_savedir:str=None,
    do_normalize_y:bool=False, 
    output_ensemble_xmin:bool=False,
    noiseed:int=101,
    **kwargs
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:  
    """
    x: (n,d)
    base_x: (n_base, ...)
    base_y: (n_base, ...) 
    n_select_base: int
        total number of base data samples. For each set of x, 
        it selects the same subset of (base_x, base_y) of size 
        int(n_select_base/n) to obtain the corresponding y.
    y_fn: 
        callable function to give y that takes (x, base_x) as inputs.
        If y_savedir is None, y_fn has to be given.
    y_savedir:
        directory where the precomputed y are saved for all pairs of (x, base_x).
        It checks whether y_savedir is None first. If is None, y_fn has to be given.
    y_mchoise:
        model choice to compute y. Can be either 'last' or 'best'. 
    do_normalize_y: bool
    """
    if kwargs['use_double']:
        dtype = jnp.float64
    else:
        dtype = jnp.float32
     
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    n_model=kwargs['n_candidate_model']+kwargs['n_ensemble_model']
    
    if kwargs.get('use_residual', False):
        from influence_max.active_optimization.opt_model_module_pytorch import StoModelNonPL as ModelNonPL
        from influence_max.active_optimization.opt_model_module import preprocess, rnt_parameter_reconstruct as parameter_reconstruct, RntJMLPBatch as JMLPBatch, RntJMLPSingle as JMLPSingle

    else:
        from influence_max.active_optimization.opt_model_module_pytorch import RntModelNonPL as ModelNonPL
        from influence_max.active_optimization.opt_model_module import preprocess, sto_parameter_reconstruct as parameter_reconstruct, StoJMLPBatch as JMLPBatch, StoJMLPSingle as JMLPSingle

    infmax_model = ModelNonPL(
        *kwargs['n_hidden'],  
        base_x_embedding_dim=base_x_embedding_dim,
        n_model=n_model, 
        no_batch_norm= kwargs.get('no_batch_norm',False) , 
        n_noise= kwargs['sto_n_noise'] ,
        noise_std=kwargs['sto_noise_std'] ,
        search_domain=search_domain ,
        trans_method=kwargs['trans_method'] , 
        trans_rbf_nrad=kwargs['trans_rbf_nrad'] , 
        use_double=kwargs['use_double'])
    optimizer = torch.optim.Adam(infmax_model.parameters(), lr=kwargs['learning_rate'], weight_decay=kwargs.get('weight_decay', 0.01))

    
    train_base_x_embedding = torch.load(os.path.join(kwargs['path_logs'], f'precomputed_featemb-train.pt'))
    n_base = train_base_x_embedding.shape[0]
    print(f"In total we have training base data n={n_base}.")


    infmax_model.to(device)
    infmax_model.train()
    

    ymean, ystd = get_train_y_stats(
        x.shape[0], 
        y_savedir=train_y_savedir, 
        do_normalize_y=do_normalize_y)
    
    for epoch in range(kwargs['max_epochs']):
        running_loss = 0.

        # selected_base_indices = torch.from_numpy(np.random.choice(n_base, size=kwargs['n_select_base'], replace=False))
        # selected_base_x_embedding = train_base_x_embedding[selected_base_indices]         # (n_selected_base, ...)
        # selected_y = get_train_y_for_all_x(
        #     x,                                   # (n, d)
        #     selected_base_indices,               # (n_selected_base, )
        #     y_savedir=kwargs['path_logs'], 
        #     ymean=ymean, 
        #     ystd=ystd
        # )                                        # (n, n_selected_base)

        train_y = get_train_y_for_all_x(
            x,                                   # (n, d)
            torch.arange(n_base),                # (n_base, )
            y_savedir=kwargs['path_logs'], 
            ymean=ymean, 
            ystd=ystd
        )                                        # (n, n_base)
        
        # j = 0 
        n_batch = math.ceil(kwargs['n_select_base']/kwargs['batch_size'])
        for b in range(n_batch):
            lower = b*kwargs['batch_size']
            upper = min((b+1)*kwargs['batch_size'], kwargs['n_select_base'])
            minibatch_base_x_embedding = train_base_x_embedding[lower:upper]     # (base_batch_size , ...)
            minibatch_y = train_y[:,lower:upper]                                 # (n, base_batch_size )
            
            for i in range(x.shape[0]): 
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                
                # Compute the loss and its gradients
                loss = infmax_model.loss_fn(
                    minibatch_base_x_embedding.to(device), 
                    x[i].to(device),
                    torch.tile(minibatch_y[i],(n_model,1)).to(device))
                loss.backward()
                # Adjust learning weights
                optimizer.step()

                running_loss += loss.item()

        running_loss = running_loss/ x.shape[0] / n_batch
        
        print('At epoch {:d}, loss: {:.4f}'.format(epoch, running_loss))
        running_loss = 0.

    
    # for b in range(n_batch):
    #     lower = b*kwargs['batch_size']
    #     upper = min((b+1)*kwargs['batch_size'], kwargs['n_select_base'])
    #     minibatch_base_x_embedding = train_base_x_embedding[lower:upper]     # (base_batch_size , ...)
    #     minibatch_y = train_y[:,lower:upper]                                 # (n, base_batch_size )
        
    #     for i in range(x.shape[0]): 
    #         # Compute the loss
    #         loss = infmax_model.loss_fn(
    #             minibatch_base_x_embedding.to(device), 
    #             x[i].to(device),
    #             torch.tile(minibatch_y[i],(n_model,1)).to(device))
    #         running_loss += loss.item()


    ## Creating the preprocess function...
    latent_embedding_fn = preprocess(
        mu     = jnp.array(infmax_model.latent_embedding_fn.mu.cpu().numpy()), 
        gamma  = jnp.array(infmax_model.latent_embedding_fn.gamma.cpu().numpy()),
        method = kwargs.get('trans_method', 'rbf'))
   
    small_base_x_embedding_train = jnp.array(get_precomputed_base_x_embedding(
        kwargs['path_logs'], base_dm.small_base_train_i, 'train')
    )
    small_base_x_embedding_targt = jnp.array(get_precomputed_base_x_embedding(
        kwargs['path_logs'], base_dm.small_base_targt_i, 'targt')
    )
    print("use_pretrained_featemb", small_base_x_embedding_train.shape, small_base_x_embedding_targt.shape)

    ## Reconstruct the model in JAX ...
    model_fn_BATCH = JMLPBatch( 
        n_hidden            = kwargs['n_hidden'], 
        latent_embedding_fn = latent_embedding_fn,  
        ymean               = jnp.array(ymean),
        ystd                = jnp.array(ystd),
        no_batch_norm       = kwargs.get('no_batch_norm', False),
        n_noise             = kwargs.get('sto_n_noise', 500),
        noise_std           = kwargs.get('sto_noise_std', 1.),
        resample_size       = kwargs.get('sto_n_resample', 200),
        dtype               = dtype,
        key                 = noiseed
    ).apply

    model_fn_SINGLE = JMLPSingle( 
        n_hidden            = kwargs['n_hidden'], 
        latent_embedding_fn = latent_embedding_fn,  
        ymean               = jnp.array(ymean),
        ystd                = jnp.array(ystd),
        no_batch_norm       = kwargs.get('no_batch_norm', False),
        n_noise             = kwargs.get('sto_n_noise', 500),
        noise_std           = kwargs.get('sto_noise_std', 1.),
        resample_size       = kwargs.get('sto_n_resample', 200),
        dtype               = dtype,
        key                 = noiseed
    ).apply
    
    small_y_train  = jnp.array(get_train_y_for_all_x(selected_base_indices=base_dm.small_base_train_i))
    model_vars_all = sto_parameter_reconstruct(infmax_model.nets)
    
    del infmax_model, base_dm
    gc_cuda()     
    
    ## Obtaining xmin of test loss for each individual candidate model...
    t0 = time.time()
    xmins = jnp.vstack(tree_map(
        lambda j: global_optimization(
            fun_to_opt = Partial(
                    jit(model_fn_BATCH), 
                    freeze(
                        {'params'     : model_vars_all['params']['MLP_'+str(j)],
                        'batch_stats' : model_vars_all['batch_stats']['MLP_'+str(j)]}), 
                    small_base_x_embedding_targt
                ),
                # ###################### TOO SLOW ######################
                # ## If we put process_in_batches outside the StoJMLP  
                # ## then it takes around 3 mins 
                # ## Instead, if we put process_in_batches inside the StoJMLP  
                # ## then it only takes around 30 seconds
                # lambda x: process_in_batches(
                #             jit(Partial(model_fn, 
                #             freeze(
                #                 {'params'      : model_vars_all['params']['MLP_'+str(j)],
                #                 'batch_stats' : model_vars_all['batch_stats']['MLP_'+str(j)]}), 
                #             x=x)), 
                # small_base_x_embedding_targt,
                # 1,
                # "mean"),
            method          = kwargs.get('search_xmin_method', 'grid-search'),
            search_domain   = jnp.array(search_domain), 
            nstart          = kwargs.get('search_xmin_nstart', 100), 
            optimize_method = kwargs.get('search_xmin_opt_method', 'trust-constr'),  
            use_double      = kwargs.get('use_double', False))[0], 
        list(range(kwargs.get('n_candidate_model', 5)))
    ))
    t1 = time.time() - t0 
    print("Obtained xmin from {:d} models. It takes {:.3f}s.".format(kwargs.get('n_candidate_model',5), t1))
    
    for ii in range(kwargs.get('n_candidate_model', 5)):
        print("xmin(M{:d})=({:s})".format(ii, print_x(xmins[ii])))
    
    del model_fn_BATCH
    gc_cuda()

    xmin_star = None
    if output_ensemble_xmin:
        t0 = time.time()
        """
        Random choose one to obtain next 
        """
        choosen_idx = np.random.choice(kwargs['n_candidate_model'])
        xmin_star = xmins[choosen_idx]
        """
        Obtaining xmin for the ensemble model (excluding jackknife ones)
        """
        # xmin_star = get_xmin_from_net(
        #     model_fn        = jit(ens_test.apply),
        #     model_params    = variables, 
        #     search_domain   = jnp.array(search_domain), 
        #     nstart          = kwargs.get('search_xmin_nstart', 100), 
        #     method          = kwargs.get('search_xmin_method', 'grid-search'),
        #     optimize_method = kwargs.get('search_xmin_opt_method', 'trust-constr'),   
        #     use_double      = kwargs.get('use_double', False)
        # )
         
        t1 = time.time()-t0
        print("Obtained xmin_star=({:s}). It takes {:.3f}s.".format(print_x(xmin_star), t1))    
    # if xmin_pre is not None and kwargs['eta'] < 1:
    #     xmin = xmin*kwargs['eta'] + xmin_pre * (1-kwargs['eta'])
    

    return (   
        model_fn_SINGLE,   
        model_vars_all, 
        small_base_x_embedding_train,
        small_base_x_embedding_targt,
        small_y_train,                          # (n, n_base)
        xmins, 
        xmin_star,   
        jnp.array(list(train_metrics.values()))
    ) 
            
 
