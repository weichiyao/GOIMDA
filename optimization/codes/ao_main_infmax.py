# SOMEHOW V2 LIMIT THE EXPLORATION: 
# IT TRAINS NN AND ACQUIRES THE XMIN(NN)
# IT THEN RETRAINS THE NN WITH XMIN(NN), FXMIN(NN) INCLUDED
# ALGORIGHM ACQUIRS XNEW AFTERWARDS
# XNEW VERY CLOSE TO XMIN(NN)
# V1 ACQUIRES XMIN(NN) AFTER XNEW FROM ALGORIGHM, 
# IT ONLY RETRAIN NETWORKS ONCE 
# DOES THAT MEAN THE ALGORITHM IS MOSTLY EXPLOITATION?

import argparse 
import copy
import os

from scipy.stats import gamma

import numpy as np
import jax.numpy as jnp 
import numpy.random as npr
import jax.random as jr
jr.PRNGKey(42)

import torch
import functools
import blackhc.laaos as laaos 
from blackhc.laaos import create_file_store

from utils import print_x, print_progress, gc_cuda
from context_stopwatch import ContextStopwatch
import data_modules.data_generator as OptGenData 
from influence_max.influence_max import InfluenceMax  
from influence_max.active_optimization.opt_train_pl import train_pl_model

def create_experiment_config_argparser(parser):
    parser.add_argument(
        "--use_double", 
        action="store_true", 
        default=False, 
        help="whether to use double precision; if not, float"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10, 
        help="input batch size for training"
    )
    
    parser.add_argument(
        "--scoring_batch_size", 
        type=int, 
        default=4096, 
        help="input batch size for scoring"
    )

    parser.add_argument(
        "--test_batch_size", 
        type=int, 
        default=16384, 
        help="input batch size for testing"
    )

    parser.add_argument(
        "--validation_set_size",
        type=int,
        default=1024,
        help="validation set size (0 for len(test_dataset) or whatever we got from the dataset)",
    )

    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=1000, 
        help="maximum number of epochs to train"
    )

    parser.add_argument(
        "--min_epochs", 
        type=int, 
        default=1, 
        help="minimum number of epochs to train"
    )
    
    parser.add_argument(
        "--sampler",
        type=str,
        default="none",
        help="choice of 'none' or 'random_fixed_length'"
    )

    parser.add_argument(
        "--epoch_samples", 
        type=int, 
        default=1024,
        help="number of epochs to train"
    )
    
    parser.add_argument(
        "--available_sample_k",
        type=int,
        default=1,
        help="number of active samples to add per active learning iteration",
    )

    parser.add_argument(
        "--n_initial_samples",
        type=int,
        default=20,
        help="number of initial active samples before the active optimization starts",
    )

    parser.add_argument(
        "--target_num_acquired_samples", 
        type=int, 
        default=100, 
        help="max number of samples to acquire"
    )

    parser.add_argument(
        "--target_accuracy", 
        type=float, 
        default=0.80, 
        help="max accuracy to train to"
    )

    parser.add_argument(
        "--target_tolerance", 
        type=float, 
        default=1e-4, 
        help="min tolerance to train to"
    )

    parser.add_argument(
        "--acquisition_method", 
        type=str, 
        default="infmax", 
        help="acquisition method can be 'infmax' or 'random'."
    )

    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        default=False, 
        help="disables CUDA training"
    )

    parser.add_argument(
        "--n_devices", 
        type=int, 
        default=1, 
        help="Number of devices used in trainer"
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=5, 
        help="Number of workers used in dataloader"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        help="random seed"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help=f"Branin,Rosenbrock,Hartmann3,Levy4,Hartmann6,Ackley",
    )
    
    parser.add_argument(
        "--sample_var",
        type=float,
        default=0.,
        help="var of the gaussian noise",
    )
    
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.,
        help="if sample_var = -1.0, var of the gaussian noise is noise_level * var(y) \
              where y are randomly generated in the noise-free setting",
    )

    parser.add_argument(
        "--low_dim",
        type=int,
        default=0,
        help="dimensions to generate dataset using PPCA",
    )

    parser.add_argument(
        "--high_dim",
        type=int,
        default=0,
        help="dimensions of the inputs (default value is 0, meaning not in high-dimensional setting)",
    )

    parser.add_argument(
        "--path_laaos_prefix",
        type=str,
        default='/scratch/wy635/active_learning/ActiveOptimization/singularity/data-out/',
        # default='/home/',
        help="path of the prefix for laaos",
    )

    parser.add_argument(
        "--path_logs",
        type=str,
        default='/scratch/wy635/active_learning/ActiveOptimization/singularity/data-out/',
        help="path of the lightning logs",
    )
    
    parser.add_argument(
        "--path_data",
        type=str,
        default='/scratch/wy635/active_learning/ActiveOptimization/data/',
        help="path of the downloaded datasets, such as MNIST, CIFAR10, and etc.",
    )

    parser.add_argument(
        "--n_candidate_model",
        type=int,
        default=3,
        help="number of candidate models to estimate y",
    )

    parser.add_argument(
        "--n_ensemble_model",
        type=int,
        default=5,
        help="number of iterations/models to compute (jacknife/ensemble) estimator of Ey",
    )

    parser.add_argument(
        "--leave_one_out",
        action="store_true", 
        default=False, 
        help="whether we want to perform leave-one-out in estimation of Ey",
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true", 
        default=False, 
        help="whether perform early_stopping",
    )

    parser.add_argument(
        "--use_validation_set",
        action="store_true",
        default=False,
        help="whether use validation set",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3, 
        help="whether perform early_stopping",
    )

    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=100, 
        help="One check happens after every _ numbers of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate used in optim.AdamW for training",
    )
 
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight_decay in the optim.AdamW for training",
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="gamma used in lr_scheduler.MultiStepLR for training",
    ) 
     
    parser.add_argument(
        "--eta",
        type=float,
        default=1.,
        help="weight for the new calculated xmin",
    ) 

    parser.add_argument(
        "--n_hidden",
        type=int,
        nargs='+',
        default=[256, 64, 64, 16],
        help="Number of hidden units in each layer of the neural networks",
    )

    parser.add_argument(
        "--sto_n_resample",
        type=int,
        default=50,
        help="number of resamples to average over for stochastic MLP",
    )

    parser.add_argument(
        "--sto_n_noise",
        type=int,
        default=100,
        help="number of noise neurons in the stochastic layer",
    )

    parser.add_argument(
        "--sto_noise_std",
        type=float,
        default=1.,
        help="std of noise neurons in the stochastic layer",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Probability of an element to be zeroed",
    ) 
    
    parser.add_argument(
        "--ihvp_method",
        type=str,
        default='cg',
        help="Method to compute the inverse hessian vector product, conjugate gradient 'cg' or LiSSA 'lissa'",
    )

    parser.add_argument(
        "--ihvp_batch_size", 
        type=int, 
        default=5000, 
        help="input batch size for hessian in inverse hessian product estimation"
    )
    
    parser.add_argument(
        "--cg_method",
        type=str,
        default='trust-ncg',
        help="optimization method for conjugate gradient; choice: 'Newton-CG', 'trust-ncg', 'trust-krylov'.",
    )

    parser.add_argument(
        "--cg_lambda",
        type=float,
        default=1e-7,
        help="The damping term to stabilize the solution for conjugate gradient",
    )

    parser.add_argument(
        "--lissa_T",
        type=int,
        default=1,
        help="number of independent runs whose results to be averaged to compute inverse hvp",
    )

    parser.add_argument(
        "--lissa_J",
        type=int,
        default=50,
        help="number of recursive computation so that approximation converges in inverse hvp computation",
    )

    parser.add_argument(
        "--lissa_scaling",
        type=float,
        default=1e-4,
        help="scaling factor in lissa inverse hessian product estimation",
    )

    parser.add_argument(
        "--lissa_damping",
        type=float,
        default=0,
        help="damping factor in lissa inverse hessian product estimation",
    )
    
    parser.add_argument(
        "--scaling_task",
        type=float,
        default=1e-4,
        help="scaling factor in hessian computation for task gradients",
    )
    
    parser.add_argument(
        "--search_xmin_nstart",
        type=int,
        default=100,
        help="jow many times we initialize the starting points for optimization in xmin search",
    )

    parser.add_argument(
        "--search_xmin_opt_tol",
        type=float,
        default=1e-5,
        help="it determines the optimization algorithm will stop when either the gradient norm or the step size falls below tol, depending on which of these conditions is relevant for the chosen method",
    )
    
    parser.add_argument(
        "--search_xmin_method",
        type=str,
        default='grid-search',
        help="method in xmin search; choice includes 'multi-start' and 'grid-search'",
    )

    parser.add_argument(
        "--search_xmin_opt_method",
        type=str,
        default='trust-constr',
        help="optimization method in xmin search; choice includes 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell' and 'trust-constr'",
    )

    parser.add_argument(
        "--disp",
        action="store_true", 
        default=True, 
        help="set to True to print convergence messages",
    )
    
    parser.add_argument(
        "--trans_method",
        type=str,
        default='rbf',
        help="transform method",
    )
    
    parser.add_argument(
        "--trans_rbf_nrad",
        type=int,
        default=5,
        help="number of dimensions to expand in radial basis functions",
    ) 
    
    parser.add_argument(
        "--do_normalize_output",
        action="store_true", 
        default=False, 
        help="whether to normalize the output",
    )
   
    parser.add_argument(
        "--n_small_data",
        type=int,
        default=5000, 
        help="number of base data to make a small training base dataset",
    )

    parser.add_argument(
        "--n_select_base",
        type=int,
        default=10000, 
        help="number of base data to train the infmax model",
    )

    parser.add_argument(
        "--eps_greedy",
        type=float,
        default=0.05,
        help="percentage of the time we use random acquisition for infMax",
    )
    
    parser.add_argument(
        "--progress_bar",
        action="store_true", 
        default=False, 
        help="whether print the progress",
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="_-th trial",
    ) 
    
    parser.add_argument(
        "--resume",
        action="store_true", 
        default=False, 
        help="whether to resume from the previous results",
    ) 

    parser.add_argument(
        "--acquire_fxmin_step",
        type=int,
        default=1, 
        help="every # number of steps (>=1) to acquire fxmin; if -1 then not at all ", 
    )

    parser.add_argument(
        "--m_kmeanspp",
        type=float,
        default=1.,
        help="multiples to oversample and apply kmeans++",
    ) 

    parser.add_argument(
        "--mu_true",
        action="store_true", 
        default=False, 
        help="whether to use mu_true for E_p0(y|x)[y] in influenceMax computation", 
    )

    parser.add_argument(
        "--net_type",
        type=str,
        default="syn",
        help=f"syn,vgg,lenet,cnn,resnet,preactrn,wrn",
    )

    parser.add_argument(
        "--no_batch_norm",
        action="store_true", 
        default=False, 
        help="set to True to deactivate batch_norm",
    )

    parser.add_argument(
        "--covariate_shift",
        action="store_true", 
        default=False, 
        help="set to True to activate covariate shift",
    )

    parser.add_argument(
        "--shift_in_distribution",
        action="store_true", 
        default=False, 
        help="set to True to generate in distribution shifted test data",
    )

    parser.add_argument(
        "--ignore_threshold",
        type=float,
        default=0.,
        help="if any of the acquired samples has all dimensions with \
              its absolute difference of the candidate's corresponding \
              dimension smaller than the threshold, then we should ignore. \
              The default is we should not ignore at all.",
    ) 
    
    parser.add_argument(
        "--replace_fxmin_if_reocurr",
        action="store_true", 
        default=False, 
        help="set to True to acquire perturbed xmin when xmin satisfies the ignore_criterion; \
              default ignore/not acquire ('duplicated') xmin.",
    )   

    parser.add_argument(
        "--use_pretrained_featemb",
        action="store_true", 
        default=False, 
        help="use the last step feature embedding model or train to obtain the feature embedding (paired with disable_base_x_embedding_training)...."
    )

    parser.add_argument(
        "--pretrained_featemb_model_path",
        type=str,
        default="", 
        help="full path to get pretrained_featemb file, e.g. '/usr/pretrained_model/resnet18.pt'"
    )

    parser.add_argument(
        "--pretrained_featemb_use_logits",
        action="store_true", 
        default=False, 
        help="use the logits from the pretrained model or use the feature embedding from the pretrained model as the embedding."
    )

    parser.add_argument(
        "--disable_base_x_embedding_training",
        action="store_false", 
        default=True, 
        help="disable the base_x_embedding training. If True then use_pretrained_featemb has to be True. If False and paired with use_pretrained_featemb, then train a linear layer on top of feature embedding from the pretrained model, otherwise train from scratch."
    )

    parser.add_argument(
        "--differ_sample_for_each_model",
        action="store_true", 
        default=False, 
        help="differ samples to train for each model in the ensembles (train_pl)"
    )

    parser.add_argument(
        "--use_residual",
        action="store_true", 
        default=False, 
        help="whether to use residual neural networks"
    )

    return parser
 
def ignore_criterion(samples_x, new_x=None, threshold:float=0.05, search_domain=None):
    """To decide whether we should ignore/not acquire new_x 
    
    If new_x is not None and there exists (or exist more than) one sample from samples_x, 
    the distance between the two points in every dimension is smaller than 
    the threshold (scaled w.r.t. corresponding search domain width), 
    then new_x should be ignored.

    Arguments
    ################################
    samples_x      : of shape (n_sample, n_dim)
    new_x         : of shape (..., n_dim)
    threshold     : similarity defined w.r.t. search domain being [0,1]
        Threshold values should be defined w.r.t. each i, i=1,...,n_dim. 
        For features that have a wider search domain, 
        the similarity threshold should be larger. 
    search_domain : of shape (n_dim, 2)
    
    Returns
    #################################
    bool: True -- too similar to one of the samples_x, should ignore; 
          else not similar to any one of the samples_x, should Not ignore.
    """
    if new_x is not None:
        scaled_threshold = threshold * (search_domain[:,1] - search_domain[:,0]) # (n_dim,)
        # Whether for all dimensions, each falls into the corresponding threshold
        res = np.all(np.abs(new_x - samples_x) < scaled_threshold, axis=-1) # (n_sample,)
        # Decide if it is True for any of the samples_x
        return np.any(res)
    else:
        return True

def perturb(x:np.ndarray, threshold:float=0.05, search_domain=None, offset:float=0.1, only_one=True):
    """To perturb x on one randomly chosen dimension

    Given every dimension of x fall within the threshold (scaled w.r.t. corresponding 
    search domain width), randomly pick one dimension to perturb, using a right-skewed 
    gamma distribution with density peaking around threshold.

    Arguments
    ################################
    x             : of shape (..., n_dim)
    threshold     : similarity defined w.r.t. search domain being [0,1]
    search_domain : of shape (n_dim, 2)
    offset        : sample values starting at threshold*(1-offset)
    
    Returns
    #################################
    perturbed x on one randomly chosen dimension 
    """
    n_dim = x.shape[-1]
    dtype = x.dtype
     
    gamma_shape = 3 # A smaller shape parameter will increase the right skewness
    gamma_scale = threshold * 0.4 / gamma_shape
    
    if only_one:
        # to perturb one randomly chosen dimension 
        to_perturb = npr.choice(n_dim, (1,)) 
        
    else: 
        
        # every index can be perturbed with 0.5 probability; n_perturb = sum(ind)
        indicator = npr.choice(2, n_dim).astype(bool)
        # make sure at least one dimension is perturbed
        while sum(indicator) < 1:
            indicator = npr.choice(2, n_dim).astype(bool)

        to_perturb = np.where(indicator)[0]

    for i in to_perturb:
        # perturb the target dimensions 
        val = gamma.rvs(a=gamma_shape, scale=gamma_scale) + (1 - offset) * threshold
        # compute the absolute scaled values
        scaled_val = val * (search_domain[i,1] - search_domain[i,0])
        # randomized the sign
        sign = npr.choice(2) * 2 - 1 
        x_perturb_i = x[...,i] + sign * scaled_val 
        while (x_perturb_i < search_domain[i,0]) or (x_perturb_i > search_domain[i,1]): 
            # sample a different perturbed values
            val = gamma.rvs(a=gamma_shape, scale=gamma_scale) + (1 - offset) * threshold
            # compute the absolute scaled values
            scaled_val = val * (search_domain[i,1] - search_domain[i,0])
            # randomized the sign
            sign = npr.choice(2) * 2 - 1 
            x_perturb_i = x[...,i] + sign * scaled_val
        # finally we have the perturbed x falls into the search domain    
        x[...,i] = x_perturb_i
    return x.astype(dtype)

def main():
    parser = argparse.ArgumentParser(
        description="InfMax", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
    )
    
    parser.add_argument(
        "--experiment_description", 
        type=str, 
        default="Active optimization", 
        help="Description of the experiment"
    )
    parser = create_experiment_config_argparser(parser)
    args, _ = parser.parse_known_args()
     
    ## Setting compute 
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.pin_memory = True if args.use_cuda else False

    if args.sample_var < 0:
        store_name = '_'.join([
            '-'.join([args.dataset, args.net_type]) if args.net_type != 'syn' else args.dataset,
            'h' + str(args.high_dim) +'l' + str(args.low_dim),
            'var' + str(args.noise_level),  
            'bsize' + str(args.available_sample_k),  
            'seed' + str(args.seed),
            'trial'+ str(args.trial),
            args.acquisition_method
        ])
    else:
        store_name = '_'.join([
            '-'.join([args.dataset, args.net_type]) if args.net_type != 'syn' else args.dataset,
            'h' + str(args.high_dim) +'l' + str(args.low_dim),
            'var' + str(args.sample_var),  
            'bsize' + str(args.available_sample_k),  
            'seed' + str(args.seed),
            'trial'+ str(args.trial),
            args.acquisition_method
        ])

    resume_filename = os.path.join(args.path_laaos_prefix, store_name+'.py')
    
    # Initial samples are assumed yet-to-be-generated
    initial_to_generate = True
    # Acquisition step is initialized to be 0
    initial_iterations = []
    it = 0
    
    if args.resume and os.path.isfile(resume_filename):
        print(f"Loading previous file {resume_filename}... ")

        resume_file = laaos.safe_load(resume_filename)
        resume_args = resume_file['args']

        path_laaos_prefix = args.path_laaos_prefix
        target_num_acquired_samples = args.target_num_acquired_samples
        use_cuda    = args.use_cuda
        pin_memory  = args.pin_memory
        num_workers = args.num_workers
        n_devices   = args.n_devices
        
        args = resume_args
        args['target_num_acquired_samples'] = target_num_acquired_samples
        args['path_laaos_prefix'] = path_laaos_prefix
        args['num_workers'] = num_workers
        args['pin_memory']  = pin_memory
        args['use_cuda']    = use_cuda
        args['n_devices']   = n_devices
        dtype = np.float64 if args['use_double'] else np.float32

        if 'initial_samples' in resume_file and len(resume_file['initial_samples']) > 0:
            initial_to_generate = False 
            initial_samples = resume_file['initial_samples']
            samples_x       = np.array(initial_samples['init_x'],       dtype=dtype)
            samples_y_train = np.array(initial_samples['init_y_train'], dtype=dtype)
            samples_y_targt = np.array(initial_samples['init_y_targt'], dtype=dtype) 
            
            gendata = getattr(OptGenData, args['dataset'])(
                low_dim=args['low_dim'],
                high_dim=args['high_dim'],
                sample_var=args['sample_var'],
                noise_level=args['noise_level'],   
                use_double=args['use_double'],
                net_type=args['net_type'],
                n_small_data=args['n_small_data'],
                data_dir=args['path_data'],
                log_dir=args['path_logs'],
                seed=args['seed'],
                shift=args.get('covariate_shift', False),
                in_distribution=args.get('shift_in_distribution', False),
                num_workers=args['num_workers'],
                pin_memory=args['pin_memory'],
                n_devices=args['n_devices']
            )
            if 'iterations' in resume_file and len(resume_file['iterations']) > 0:
                initial_iterations = resume_file['iterations']
                # iteration recording at step 0, so needs to exclude the first one
                for item in initial_iterations[1:]:
                    chosen_samples  = np.array(item['chosen_samples'], dtype=dtype)
                    samples_x       = np.vstack([samples_x, chosen_samples]) 
                    samples_y_train = np.hstack([samples_y_train, item['y_train_chosen_samples']]).astype(dtype)
                    samples_y_targt = np.hstack([samples_y_targt, item['y_targt_chosen_samples']]).astype(dtype)
                                        
                lastitem = initial_iterations[-1]
                it = lastitem['step'] 

                if lastitem['step'] >= target_num_acquired_samples:
                    print("DONE.")
                    return        
        
    else:
        print(f"Creating new file {store_name}... ")
        args = args.__dict__
        dtype = np.float64 if args['use_double'] else np.float32
    print(args)

    ## Setting compute  
    device = "cuda" if args['use_cuda'] else "cpu"
    print(f"Using {device} for computations.")

    if initial_to_generate:
        gendata = getattr(OptGenData, args['dataset'])(
            low_dim=args['low_dim'],
            high_dim=args['high_dim'],
            sample_var=args['sample_var'],
            noise_level=args['noise_level'],  
            seed=args['seed'], 
            use_double=args['use_double'],
            n_small_data=args['n_small_data'],
            net_type=args['net_type'],
            data_dir=args['path_data'],
            log_dir=args['path_logs'],
            shift=args.get('covariate_shift', False),
            in_distribution=args.get('shift_in_distribution', False),
            num_workers=args['num_workers'],
            pin_memory=args['pin_memory']
        )
        
        # generate initial samples
        samples_x = gendata.gen_samples(args['n_initial_samples'])
        samples_y_train = []
        samples_y_targt = [] 
        for ii in range(args['n_initial_samples']):
            # call the base model 
            out_train, out_targt = gendata.evaluate_all_and_save_individual(samples_x[ii], ii)
            samples_y_train.append(out_train)  
            samples_y_targt.append(out_targt) 
        
        samples_y_train = np.hstack(samples_y_train).astype(samples_x.dtype)
        samples_y_targt = np.hstack(samples_y_targt).astype(samples_x.dtype) 
            
        initial_samples = {'init_x'  : samples_x.tolist(), 
                           'init_y_train': samples_y_train.tolist(), 
                           'init_y_targt': samples_y_targt.tolist()}  

    ## Current status
    opt_target = gendata._f_opt
    # the i-th sample that gives the best loss value on dataset A 
    idx_obsmin = np.argmin(samples_y_train)
    # the corresponding hyperparameters
    x_obsmin   = samples_x[idx_obsmin]
    
    if args['net_type'] != 'syn':
        # the best loss on dataset A   
        tol_train_obsmin = samples_y_train[idx_obsmin] - opt_target
        # the corresponding loss on dataset B
        tol_targt_obsmin  = samples_y_targt[idx_obsmin] - opt_target

    tol = copy.deepcopy(tol_targt_obsmin)
           
    print_progress(
        nsample   = samples_x.shape[0],
        tol       = tol_targt_obsmin,  
        train_tol = tol_train_obsmin, 
        xmin      = x_obsmin
    )

    store = create_file_store(
        store_name,
        suffix="",
        prefix=args['path_laaos_prefix'],
        truncate=False,
        type_handlers=(laaos.StrEnumHandler(), laaos.ToReprHandler()),
    )
    store["args"] = args
    store["initial_samples"] = initial_samples
    # store wraps the empty list in a storable list, so we need to fetch it separately.
    store["iterations"] = []
    ## The best results from the initial samples 
    iterations = store["iterations"]

    if it == 0:
        iterations.append(
            dict(
                step=0, 
                tol=tol_targt_obsmin,  
                train_tol=tol_train_obsmin, 
                xmin=x_obsmin.tolist(), 
                scores=['random'],
                chosen_samples         = samples_x[-args['available_sample_k']:].reshape(args['available_sample_k'], -1).tolist(), 
                y_train_chosen_samples = samples_y_train[-args['available_sample_k']:].tolist(), 
                y_targt_chosen_samples = samples_y_targt[-args['available_sample_k']:].tolist(), 
                train_model_elapsed_time=0,
                batch_acquisition_elapsed_time=0,
            )
        ) 
    else:
        ## The best results from the following acquisitions, if there is any
        for item in initial_iterations:
            iterations.append(item)
    
    # Updating the number of initial sample points
    n_init = samples_x.shape[0]
    print(f"Starting from {n_init} number of samples, intend to acquire another {args['target_num_acquired_samples']}.")
    
    # if ever acquire, then acquire the xmin after the very first fit
    acquire_fxmin_current_step = args['acquire_fxmin_step']-1 if args['acquire_fxmin_step'] < 1 else args['acquire_fxmin_step']
    ignore_flag, ignore_flag_fxmin = True, True
    while (samples_x.shape[0] - args['n_initial_samples'] < args['target_num_acquired_samples']): 
        if tol <= args['target_tolerance']:
            print("Reached target tolerance {:.3f} with current tolerance {:.3f}".format(
                args['target_tolerance'], tol))
            break
        
        
        with ContextStopwatch() as train_model_stopwatch: 
            if it != 1: 
                base_x_embedding_fn, base_x_embedding_dim = gendata.get_base_x_embedding(
                    args['pretrained_featemb_model_path'] if args['use_pretrained_featemb'] else "", 
                    args.get('pretrained_featemb_use_logits', False))

            
                (model_fn, model_vars, model_vars_truehat, 
                small_base_x_embedding_train, small_base_x_embedding_targt, 
                small_y_train, xmins, new_x_nn, train_metrics) = train_pl_model( 
                    x=torch.from_numpy(samples_x), 
                    base_dm=gendata.dm,
                    search_domain=torch.from_numpy(gendata.search_domain), 
                    base_x_embedding_fn=base_x_embedding_fn, 
                    base_x_embedding_dim=base_x_embedding_dim,
                    train_y_savedir=gendata.log_dir,
                    do_normalize_y=args['do_normalize_output'],
                    output_ensemble_xmin=True,
                    noiseed=it,
                    **args
                )
                del base_x_embedding_fn, base_x_embedding_dim
                gc_cuda()
        if it == 1:
            ignore_flag_fxmin = False
        else:

            ignore_flag_fxmin = (acquire_fxmin_current_step < args['acquire_fxmin_step'])   

            ## Update ignore_flag_fxmin     
            if (not ignore_flag_fxmin):
                duplicate_nn = ignore_criterion(samples_x, 
                                                new_x_nn, 
                                                args.get('ignore_threshold', 0.01), 
                                                gendata.search_domain)
                if duplicate_nn:
                    if args.get('replace_fxmin_if_reocurr', False):
                        # perturb xmin
                        new_x_nn = perturb(np.array(new_x_nn).astype(dtype), 
                                        threshold=args.get('ignore_threshold', 0.01)*1.5,
                                        search_domain=gendata.search_domain,
                                        only_one=False)
                        duplicate_nn = ignore_criterion(samples_x=samples_x, 
                                                        new_x=new_x_nn, 
                                                        threshold=args.get('ignore_threshold', 0.01),
                                                        search_domain=gendata.search_domain)
                        while duplicate_nn:
                            new_x_nn = perturb(np.array(new_x_nn).astype(dtype), 
                                            threshold=args.get('ignore_threshold', 0.01)*1.5,
                                            search_domain=gendata.search_domain,
                                            only_one=False)
                            duplicate_nn = ignore_criterion(samples_x=samples_x, 
                                                            new_x=new_x_nn, 
                                                            threshold=args.get('ignore_threshold', 0.01),
                                                            search_domain=gendata.search_domain)
                    
                        scores_nn = np.full((1,), 'fxmin-perturbed')
                    else:
                        # do not acquire 
                        ignore_flag_fxmin = True 
                else: 
                    scores_nn = np.full((1,), 'fxmin')

        ## Acquiring xmin and adding to the observed samples   
        if (not ignore_flag_fxmin):
            ## Acquiring xmin 
            with ContextStopwatch() as retrain_fxmin_stopwatch:
                if it != 1:
                    # new_x_nn = copy.deepcopy(xmins[jnp.argmin(train_metrics[:args['n_candidate_model']])])
                    new_y_nn_train, new_y_nn_targt = gendata.evaluate_all_and_save_individual(
                        np.array(new_x_nn).astype(dtype), args['n_initial_samples']+it)
                    print(f"it={it+1}, acquired new_y_nn_train={new_y_nn_train}")
                    print(f"it={it+1}, acquired new_y_nn_targt={new_y_nn_targt}")
                    ## Adding the latest acquired data points to the training set
                    samples_x = np.vstack((samples_x, new_x_nn)).astype(dtype) 
                    samples_y_train = np.hstack((samples_y_train, new_y_nn_train)).astype(dtype)
                    samples_y_targt = np.hstack((samples_y_targt, new_y_nn_targt))
                    
                    it += 1
                ## Retraining the model 
                base_x_embedding_fn, base_x_embedding_dim = gendata.get_base_x_embedding(
                    args['pretrained_featemb_model_path'] if args['use_pretrained_featemb'] else "", 
                    args.get('pretrained_featemb_use_logits', False))
                (model_fn, model_vars, model_vars_truehat, 
                 small_base_x_embedding_train, small_base_x_embedding_targt, 
                 small_y_train, xmins, _, train_metrics) = train_pl_model( 
                    x=torch.from_numpy(samples_x), 
                    base_dm=gendata.dm,
                    search_domain=torch.from_numpy(gendata.search_domain), 
                    base_x_embedding_fn=base_x_embedding_fn, 
                    base_x_embedding_dim=base_x_embedding_dim,
                    train_y_savedir=gendata.log_dir,
                    do_normalize_y=args['do_normalize_output'],
                    output_ensemble_xmin=True,
                    noiseed=it,
                    **args
                )
                del base_x_embedding_fn, base_x_embedding_dim
                gc_cuda()
            
            if it != 1:
                ## Updating min values
                # the i-th sample that gives the best loss value on dataset A 
                idx_obsmin = np.argmin(samples_y_train) # if (not ignore_flag) else np.argmin(samples_y)
                # the corresponding hyperparameters
                x_obsmin = samples_x[idx_obsmin]
                if args['net_type'] != 'syn':
                    # the best loss on dataset A   
                    tol_train_obsmin = samples_y_train[idx_obsmin] - opt_target
                    # the corresponding loss on dataset B
                    tol_targt_obsmin = samples_y_targt[idx_obsmin] - opt_target
                else:
                    raise ValueError(f"not yet implemented for net_type={args['net_type']}")

                print("Round {:d} acquires xmin({}): (x,y)=({:s},{:.3f}) (after {:d} steps). Total training takes {:.3f}s.".format(
                    it, 
                    scores_nn[0],
                    print_x(new_x_nn), 
                    float(new_y_nn_train), 
                    acquire_fxmin_current_step,
                    train_model_stopwatch.elapsed_time+retrain_fxmin_stopwatch.elapsed_time
                )) 
                
                print_progress(
                    nsample   = samples_x.shape[0],
                    tol       = tol_targt_obsmin,  
                    train_tol = tol_train_obsmin, 
                    xmin      = x_obsmin
                ) 

                iterations.append(
                    dict(
                        step=it, 
                        tol=tol_targt_obsmin,  
                        train_tol=tol_train_obsmin, 
                        xmin=x_obsmin.tolist(), 
                        scores=['fxmin'],
                        chosen_samples         = new_x_nn.tolist(), 
                        y_train_chosen_samples = new_y_nn_train.tolist(), 
                        y_targt_chosen_samples = new_y_nn_targt.tolist(), 
                        train_model_elapsed_time=0,
                        batch_acquisition_elapsed_time=0,
                    )
                )  

                tol = copy.deepcopy(tol_targt_obsmin)
            ## Reset after we acquire fxmin
            acquire_fxmin_current_step = 0
        
        ## Acquiring new data points
        with ContextStopwatch() as batch_acquisition_stopwatch: 
            acquire_influence_max = InfluenceMax(
                available_x       = jnp.array(samples_x), 
                available_y       = small_y_train, 
                train_loss        = train_metrics[:args['n_candidate_model']], 
                xmins             = xmins,
                search_domain     = jnp.array(gendata.search_domain), 
                model_fn          = model_fn,
                base_x_embedding_train = small_base_x_embedding_train,
                base_x_embedding_targt = small_base_x_embedding_targt,
                model_vars        = model_vars, 
                model_vars_true   = model_vars_truehat, 
                acquire_k         = args['available_sample_k'], 
                **args
            )
            batch  = acquire_influence_max.compute_optima()
            
            new_x  = batch.samples
            scores = batch.scores
            
            ignore_flag = ignore_criterion(samples_x=samples_x, 
                                           new_x=new_x, 
                                           threshold=args.get('ignore_threshold', 0.01),
                                           search_domain=gendata.search_domain)
            if ignore_flag: 
                # scores = np.full((args['available_sample_k'],), 'random')
                scores = np.full((args['available_sample_k'],), 'infmax-perturbed')
                while ignore_flag: 
                    new_x       = perturb(np.array(new_x), 
                                          threshold=args.get('ignore_threshold', 0.01),
                                          search_domain=gendata.search_domain)
                    ignore_flag = ignore_criterion(samples_x=samples_x, 
                                                   new_x=new_x, 
                                                   threshold=args.get('ignore_threshold', 0.01),
                                                   search_domain=gendata.search_domain)
            
            new_y_train, new_y_targt = gendata.evaluate_all_and_save_individual(
                np.array(new_x).astype(dtype), 
                args['n_initial_samples']+it
            )
            
            ## Adding the latest acquired data points to the training set
            samples_x       = np.vstack((samples_x, new_x)).astype(dtype) 
            samples_y_train = np.hstack((samples_y_train, new_y_train)).astype(dtype)
            samples_y_targt = np.hstack((samples_y_targt, new_y_targt))
            
            scores_to_print = 'infmax-perturbed' if isinstance(scores[0], str) else '{:.3f}'.format(scores.mean())

        it += 1
        ## Update the current step that we count to acquire_fxmin
        acquire_fxmin_current_step = acquire_fxmin_current_step + 1 if args['acquire_fxmin_step'] > 0 else args['acquire_fxmin_step']-1 
        ## Update min values
        # the i-th sample that gives the best loss value on dataset A 
        idx_obsmin = np.argmin(samples_y_train) # if (not ignore_flag) else np.argmin(samples_y)
        # the corresponding hyperparameters
        x_obsmin   = samples_x[idx_obsmin]
        if args['net_type'] != 'syn':
            # the best loss on dataset A   
            tol_train_obsmin = samples_y_train[idx_obsmin] - opt_target
            # the corresponding loss on dataset B
            tol_targt_obsmin = samples_y_targt[idx_obsmin] - opt_target
        else:
            raise ValueError(f"not yet implemented for net_type={args['net_type']}") 
        
        print("Round {:d} acquired: xnext: (x,y)=({:s},{:.3f}), scores={}. It takes {:.3f}s.".format(
            it, print_x(new_x), float(new_y_train), scores_to_print,
            batch_acquisition_stopwatch.elapsed_time
        ))
        
        print_progress(
            nsample   = samples_x.shape[0],
            tol       = tol_targt_obsmin,  
            train_tol = tol_train_obsmin, 
            xmin      = x_obsmin
        ) 
        
        iterations.append(
            dict(
                step=it, 
                tol=tol_targt_obsmin,  
                train_tol=tol_train_obsmin, 
                xmin=x_obsmin.tolist(), 
                scores=scores_to_print,
                chosen_samples           = new_x.tolist(), 
                y_train_chosen_samples   = new_y_train.tolist(), 
                y_targt_chosen_samples   = new_y_targt.tolist(), 
                train_model_elapsed_time = (train_model_stopwatch.elapsed_time
                                            +(retrain_fxmin_stopwatch.elapsed_time if not ignore_flag_fxmin else 0)),
                batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time
            )
        )  

        tol = copy.deepcopy(tol_targt_obsmin)
    print("DONE")
        
if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


