from __future__ import division 
import time 
from abc import ABC, abstractmethod
from typing import List, Callable, Union

import numpy as np
import numpy.random as npr
 
import torch 
 
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import generate_seed_according_to_time, gc_cuda
from networks.pretrained_resnet import PretrainedFeatEmbModule
from data_modules.image_model import *
from data_modules.model_utils import *
from data_modules.image_dataset import *
from data_modules.data_utils import lowrank_perb 

MAXEPOCHS = {
    'cnn'     : 50,
    'wrn'     : 50,
    'resnet'  : 50,
    'preactrn': 50,
    'lenet'   : 50,
    'vgg'     : 50,
}

def generate_samples(
        n:int, 
        search_domain:np.ndarray, 
        method='random', 
        low_dim=0, 
        seed=generate_seed_according_to_time(1)[0]
    ) -> np.ndarray: 
    """
    Inputs
        n: Number of samples to generate.
        search_domain: (ndim, 2)
        method: 'grid' or 'random' or 'lhs'
    Returns:
        (n**ndim, ndim) if method == 'grid' else (n, ndim)

    """

    d = search_domain.shape[0]
    rng = npr.default_rng(seed)

    if method == 'lowrank':
        assert low_dim > 0, f"using 'lowrank' in generate_samples, low_dim has to be larger than 0! Received low_dim={low_dim}."
        samples = lowrank_perb(n, low_dim, d, search_domain, rng=rng)
    if method == 'grid':
        ls = [np.linspace(lo, hi, n, endpoint=True) for (lo, hi) in search_domain]
        mesh_ls = np.meshgrid(*ls)
        all_mesh = [np.reshape(x, [-1]) for x in mesh_ls]
        samples = np.stack(all_mesh, axis=1) 
    elif method == 'random':
        samples = np.zeros((n, d))
        samples = rng.uniform(0, 1, size=(n, d)) 
        samples = samples*(search_domain[:,1]-search_domain[:,0])+search_domain[:,0]
    elif method == 'lhs':
        """Latin hypercube sampling (LHS).
        It generates n points in [0,1)^d. Each univariate marginal distribution is 
        stratified, placing exactly one point in [j/n, (j+1)/n] for j = 0,1,...,n-1.

        When LHS is used for integrating a function f over n, 
        LHS is extremely effective on integrands that are nearly additive. 
        With a LHS of n points, the variance of the integral is always lower than 
        plain MC on n-1 points. There is a central limit theorem for LHS 
        on the mean and variance of the integral, but not necessarily for 
        optimized LHS due to the randomization.
        """
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        samples= qmc.scale(sampler.random(n), search_domain[:,0], search_domain[:,1])
    else: 
        raise ValueError(f"Received method={method}; can only be 'grid' or 'random' or 'lhs'.")
    
    return samples.astype(search_domain.dtype) 

class OptimizationDataGenerator(ABC):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101, 
        **kwargs
    ):   
        if use_double:
            self.npdtype = np.float64
            self.dtype = torch.float64
        else:
            self.npdtype = np.float32
            self.dtype = torch.float32
        
        ## Set the seed
        # create the RNG that you want to pass around
        rng = npr.default_rng(101)
        # get the SeedSequence of the passed RNG
        ss = rng.bit_generator._seed_seq
        # create 10 initial independent states for 10 total experiments 
        self.child_states = ss.spawn(5000) 
        # seed to generate samples
        self.seed   = seed
        # optimum
        self._f_opt = 0
    
    @abstractmethod
    def gen_samples(self, n) -> np.ndarray:
        pass

    @abstractmethod
    def init_attributes(self) -> None: 
        pass 

    @abstractmethod
    def post_init_steps(self) -> None:
        pass 
    
    @abstractmethod
    def evaluate(self, x:np.ndarray, seed:int=None):
        pass
        
class ImageDataGenerator(OptimizationDataGenerator):
    """
    Arguments: 
    ================
    noise_level: float positive 
        sample var = noise_level * var(y) 
    high_dim: int 
        Number of total inputs. 
        See _high_dim and _active_dim in Attributes  
    low_dim: int
        Number of variables to generate 
        the lowrank high-dimensional inputs
    seed: int 
        To generate random state
    sample_var: float (None by default) 
        If sample_var = -1, noise_level * var(y),
        where y of size 10000 are randomly sampled.  

    Attributes:
    ================
    _high_dim: int
        Number of total inputs. 
    _active_dim: int 
        Number of active variables in Branin function
    _sample_var: float
        Noise variance  
    """
    def __init__(
        self, 
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10, 
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False, 
        **kwargs
    ):   
        super().__init__(use_double, seed, **kwargs)
        self.search_domain = np.array([[0.,1.],[-4,-1.],[-4.,-1.],[0.,1.]]).astype(self.npdtype)
        ## Image model  
        self.n_small_data = n_small_data
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.use_validation = use_validation
        self.net_type = net_type
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.max_epochs = MAXEPOCHS[net_type]
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        ## Creating the trainer...
        self.kwargs_trainer = {
            'max_epochs': self.max_epochs, 
            'min_epochs': min(50, self.max_epochs),
            'accelerator': self.device,
            'enable_model_summary': True,
            'check_val_every_n_epoch': log_interval if use_validation else 1,
            'default_root_dir': log_dir,
            'enable_checkpointing': True,
            'enable_progress_bar': progress_bar,
            'logger': True,
            'deterministic': False,
            'devices': n_devices,
        }

    def init_attributes(self):
        self.num_classes = None
        self.rgb = None
        self.dm = None
        # base model 
        self.base_model = None

    def post_init_steps(self):
        if self.dm is None:
            raise AttributeError("Attribute 'dm' must be set in the child class.")
        if self.num_classes is None:
            raise AttributeError("Attribute 'num_classes' must be set in the child class.")
        if self.rgb is None:
            raise AttributeError("Attribute 'rgb' must be set in the child class.")
        
    def gen_samples(self, n, seed=None) -> np.ndarray:
        if seed is None: 
            seed = self.child_states[self.seed]
        return generate_samples(n, self.search_domain, seed=seed)
    
    def load_model(self, x:np.ndarray, save:bool=False, log_filepath:str=None, minfo_filepath:str=""):
        """
        x: (d,)
        """
        if os.path.isfile(minfo_filepath):
            image_model = ImageModelModule.load_from_checkpoint(minfo_filepath, map_location="cpu") 
            print("Found and loaded model from path {}".format(minfo_filepath))
        else:
            momentum, lr, wd, dropout_rate = x 
            lr = pow(10, lr)
            wd = pow(10, wd)  

            dropout_rate = np.clip(dropout_rate, None, 0.99)

            image_model = ImageModelModule(
                self.net_type,
                self.num_classes,
                self.rgb,
                dropout_rate,
                lr, 
                wd,
                momentum,
                self.max_epochs, 
                save_logpath=log_filepath
            ) 
            
            if save:
                assert minfo_filepath is not None, "When save=True, minfo_filepath cannot be None!"
                print("Could not find model path {}. Creating model and save the best...".format(minfo_filepath))
                # Save the top 1 checkpoint that give the smallest values of training or validation loss
                monitor_value = 'val_loss' if self.use_validation else 'loss'
                checkpoint_callback = [
                    ModelCheckpoint(
                            monitor=monitor_value,
                            dirpath=self.log_dir,
                            filename=os.path.splitext(minfo_filepath)[0], # '-'.join([minfo_filepath.split('.')[0],'best']),
                            auto_insert_metric_name=False,
                            # save_last=True,
                            save_top_k=1,
                            mode='min',
                        )
                ]
            else: 
                print("Creating model ... (will not save)".format(minfo_filepath))
                checkpoint_callback = None

            # Fitting the model...
            t0 = time.time()
            trainer = pl.Trainer(**self.kwargs_trainer, callbacks=checkpoint_callback)
            trainer.fit(model=image_model, datamodule=self.dm) 
            t1 = time.time()-t0
            print("Training takes {:.3f}s.".format(t1)) 
            
            # Saving the model
            # if save:
            #     # Modify the last checkoint saved name
            #     last_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, 'last.ckpt')
                
            #     if os.path.exists(last_checkpoint_path):
            #         os.rename(last_checkpoint_path, minfo_filepath)
            #     else:
            #         raise ValueError(f"The 'last' checkpoint cannot be found in {trainer.checkpoint_callback.dirpath}!")
            #     print("Saved model in {}...".format(minfo_filepath))
            #     # if model_choice == "best":
            #     #     best_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, model_best_filename)
            #     #     if os.path.exists(best_checkpoint_path):
            #     #         image_model.load_from_checkpoint(best_checkpoint_path)
            #     #     else:
            #     #         raise ValueError(f"The '{model_choice}' checkpoint cannot be found in {trainer.checkpoint_callback.dirpath}!")
        return image_model.eval()
    
    def get_base_x_embedding(self, pretrained_featemb_model_path:str="", return_logits:bool=False) -> nn.Module:
        if os.path.isfile(pretrained_featemb_model_path):  
            print(f"Generating base x embedding from pretrained feature embedding model from {pretrained_featemb_model_path} (use_logits={str(return_logits)[0]})...")
            featemb_model = PretrainedFeatEmbModule(
                filepath=pretrained_featemb_model_path,
                num_classes=self.num_classes,
                num_input_channels=3 if self.rgb else 1,
                return_logits=return_logits)
                
            if not os.path.isfile(os.path.join(self.log_dir, f'precomputed_featemb-train.pt')):
                trainer = pl.Trainer(**self.kwargs_trainer)
                all_outputs = trainer.predict(model=featemb_model, datamodule=self.dm)
                
                ## SAVE FEATURE EMBEDDING FOR BOTH TRAIN AND TEST 
                for dd in range(len(all_outputs)):
                    output_filename = 'precomputed_featemb-{}.pt'.format(self.dm._data[dd])
                    output_filepath = os.path.join(self.log_dir, output_filename) 
                    torch.save(torch.concatenate(all_outputs[dd]), output_filepath)  
            return lambda x:x, featemb_model.n_features
        else:
            print(f"Cannot load pretrained feature embedding model from {pretrained_featemb_model_path}.")
            print("Initiate a feature embedding model from the target image model...")
            featemb_model = ImageModelModule(
                self.net_type,
                self.num_classes,
                self.rgb,
                0,
                0.001, 
                0.001,
                0.9,
                1, 
                save_logpath=None
            ).featemb
            return featemb_model, featemb_model.n_features
        # minfo_filepath = os.path.join(self.log_dir, 'i{:04d}_model-best.ckpt'.format(use_i)) 
        # image_model = ImageModelModule.load_from_checkpoint(minfo_filepath, map_location="cpu").featemb  
        # ## LOAD TRAINER
        # trainer = pl.Trainer(**self.kwargs_trainer)

        # ## MAKE PREDICTIONS    
        # _ = trainer.test(model=image_model, datamodule=self.dm)
        # for dd in range(len(self.dm.get_embedding_data)): 
        #     output_filename = 'i{:04d}_emb-{}.pt'.format(use_i, self.dm.get_embedding_data[dd])
        #     output_filepath = os.path.join(self.log_dir, output_filename)
        #     torch.save(image_model.base_x_embedding[dd], output_filepath)
        # del image_model
        # return None
    
    def evaluate_all_and_save_individual(self, x:np.ndarray, i:int) -> List[np.ndarray]:
        """
        x: (d,) or (1,d)
        """
        x = x.reshape(-1)
        log_filepath   = os.path.join(self.log_dir, 'i{:04d}_log.pt'.format(i))
        minfo_filepath = os.path.join(self.log_dir, 'i{:04d}_model-best.ckpt'.format(i)) 
        image_model = self.load_model(x, save=False, log_filepath=log_filepath, minfo_filepath=minfo_filepath)
        
        ## LOAD TRAINER
        trainer = pl.Trainer(**self.kwargs_trainer)
        
        ## MAKE PREDICTIONS    
        all_outputs = trainer.predict(model=image_model, datamodule=self.dm)
        
        del image_model, trainer
        gc_cuda()

        ## SAVE EACH LOSS FOR BOTH TRAIN AND TEST, AND RETURN AVERAGE LOSS
        ret = []
        for dd in range(len(all_outputs)):
            output = torch.concatenate(all_outputs[dd]) 
            # in case there is nan values
            output[torch.isnan(output)]=9.

            ret.append(output.mean().numpy())

            output_filename = 'i{:04d}_output-{}.pt'.format(i, self.dm._data[dd])
            output_filepath = os.path.join(self.log_dir, output_filename)
            output_dict = {"y": output, "ymean": output.mean(), "yvar": output.var()}
            torch.save(output_dict, output_filepath)
            
        del all_outputs
        gc_cuda()
        return ret
        
    def evaluate(self, x:np.ndarray) -> np.ndarray:
        """
        x: (d,) or (1,d)
        """
        image_model = self.load_model(x.reshape(-1))
 
        ## LOAD TRAINER
        trainer = pl.Trainer(**self.kwargs_trainer)

        ## MAKE PREDICTIONS    
        all_outputs = trainer.predict(model=image_model, datamodule=self.dm)

        del image_model, trainer
        gc_cuda()

        ret = []
        for dd in range(len(all_outputs)):
            output = torch.concatenate(all_outputs[dd]) 
            # in case there is nan values
            output[torch.isnan(output)]=9.
            
            ret.append(output.mean().numpy())

        del all_outputs
        gc_cuda()
        return ret 
         
class CIFAR10(ImageDataGenerator):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10,
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False,
        shift                : bool=False,
        targets_to_shrink    : list=[1,2,7],
        shrink_to_proportion : float=0.1,
        in_distribution      : bool=False,
        **kwargs
    ):   
        super().__init__(use_double, seed, net_type, n_small_data,
                         data_dir, log_dir, log_interval, use_validation, 
                         num_workers, pin_memory, n_devices, progress_bar, **kwargs)
        self.shift= shift
        self.targets_to_shrink = targets_to_shrink
        self.shrink_to_proportion = shrink_to_proportion
        self.in_distribution = in_distribution
        
        self.init_attributes()
        self.post_init_steps()

    def init_attributes(self):
        ## Creating the data module...
        self.dm = CIFAR10DataModule(
            batch_size=128, 
            n_small_data=self.n_small_data,
            dtype=self.dtype,
            data_dir=self.data_dir, 
            seed=self.seed,
            use_validation=self.use_validation, 
            ntrain=None,
            shift=self.shift, 
            targets_to_shift=self.targets_to_shrink,
            shrink_to_proportion=self.shrink_to_proportion,
            in_distribution=self.in_distribution,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        self.num_classes = 10
        self.rgb = True
         
class MNISTFRS(ImageDataGenerator):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10,
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False,
        **kwargs
    ):
        super().__init__(use_double, seed, net_type, n_small_data,
                         data_dir, log_dir, log_interval, use_validation, 
                         num_workers, pin_memory, n_devices, progress_bar,
                         **kwargs)
        self.init_attributes()
        self.post_init_steps()

    def init_attributes(self):
        self.rgb = False
        self.num_classes = 10
        ## Creating the data module...
        self.dm =  MNISTFRSDataModule(
            batch_size=128, 
            n_small_data=self.n_small_data,
            dtype=self.dtype,
            data_dir=self.data_dir, 
            seed=self.seed,
            use_validation=self.use_validation, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

class SVHNFRM(ImageDataGenerator):
    def __init__(
        self,  
        use_double           : bool=False,
        seed                 : int=101,
        net_type             : str='cnn',
        n_small_data         : int=0,
        data_dir             : str='/home/data/',
        log_dir              : str='/home/log/',
        log_interval         : int=10,
        use_validation       : bool=False, 
        num_workers          : int=5,
        pin_memory           : bool=False,
        n_devices            : int=1,
        progress_bar         : bool=False,
        **kwargs
    ):
        super().__init__(use_double, seed, net_type, n_small_data,
                         data_dir, log_dir, log_interval, use_validation, 
                         num_workers, pin_memory, n_devices, progress_bar,
                         **kwargs)
        self.init_attributes()
        self.post_init_steps()

    def init_attributes(self):
        self.rgb = False
        self.num_classes = 10
        ## Creating the data module...
        self.dm =  SVHNFRMDataModule(
            batch_size=128, 
            n_small_data=self.n_small_data,
            dtype=self.dtype,
            data_dir=self.data_dir, 
            seed=self.seed,
            use_validation=self.use_validation, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
