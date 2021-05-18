import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from models.networks import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        # if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.optimizer_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        self.save_dir = os.path.join(opt.checkpoints_dir,opt.project_name, opt.exp_name, opt.run_name)  # save all the checkpoints to save_dir
        self.net_dict = { name: getattr(self, 'net' + name) for name in self.model_names}
        self.optimizer_dict = {name: getattr(self, 'optimizer_' + name) for name in self.model_names} if opt.isTrain else {}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self,vis=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers] #TODO: why need this?

        # get the latest checkpoints
        import glob

        checkpoint_folder = os.path.join(opt.checkpoints_dir, opt.project_name, opt.exp_name, opt.run_name)

        search_pattern = '*.pth'

        checkpoint_list = glob.glob(os.path.join(checkpoint_folder, search_pattern))
        print(checkpoint_folder)
        iter_start = 0
        if len(checkpoint_list) > 0:
            load_suffix = self.opt.load_suffix
            self.load_networks(load_suffix)
            if self.isTrain:
                self.load_optimizers(load_suffix)
                iter_start = self.load_states(load_suffix)

        self.print_networks(opt.verbose)

        return iter_start

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
                
    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_videos(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.video_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save(self, suffix,iter):
        """ Save all the networks, optimizers and states to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        self.save_optimizers(suffix)
        self.save_networks(suffix)
        self.save_states(suffix,iter)

    def save_networks(self, suffix):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_name = 'model_%s.pth' % suffix
        save_path = os.path.join(self.save_dir, save_name)
        outdict = {}
        for name in self.model_names:
            if isinstance(name, str):
                net_name = 'net' + name
                outdict[net_name] = getattr(self, net_name).state_dict()

        torch.save(outdict, save_path)

    def save_optimizers(self,suffix):
        """Save all the optimizers to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_name = 'optimizer_%s.pth' % suffix
        save_path = os.path.join(self.save_dir, save_name)
        output_dict = {}
        for name in self.optimizer_names:
            if isinstance(name, str):
                optimizer_name = 'optimizer_' + name
                output_dict[optimizer_name] = getattr(self, optimizer_name).state_dict()
        torch.save(output_dict, save_path)

    def save_states(self, suffix, iter):
        """Save all the states (epoch, iter) to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_name = 'states_%s.txt' % suffix
        save_path = os.path.join(self.save_dir, save_name)
        import numpy as np
        states = np.array([iter])
        np.savetxt(save_path, states)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, suffix):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_path = os.path.join(self.save_dir, 'model_%s.pth' % (suffix))
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        try:
            out_dict = torch.load(load_path,map_location=str(self.device))
            for name in self.model_names:
                if isinstance(name, str):
                    net_name = 'net' + name
                    net = getattr(self, net_name)


                    state_dict = out_dict[net_name]
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                    net.load_state_dict(state_dict, strict=True)
                    print('[%s] loaded from [%s]' % (net_name,load_path))


        except Exception:
            print('no checkpoints for the network found, parameters will be initialized')



    def load_optimizers(self, suffix):
        """Load all the optimizers from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_path = os.path.join(self.save_dir, 'optimizer_%s.pth' % (suffix))
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        try:
            out_dict = torch.load(load_path,map_location=str(self.device))
            for name in self.optimizer_names:
                if isinstance(name, str):
                    optimizer_name = 'optimizer_' + name
                    optimizer = getattr(self, optimizer_name)
                    optimizer.load_state_dict(out_dict[optimizer_name])
            print('optimizer loaded from [%s]' % load_path)
        except Exception:
            print('no checkpoints for the optimizer found, parameters will be initialized')


    def load_states(self, suffix):
        """Load all the states (epoch, iterations) from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_path = os.path.join(self.save_dir, 'states_%s.txt' % (suffix))
        import numpy as np
        try:
            out_dict = np.loadtxt(load_path)
            print('states loaded from [%s]' % load_path)
            return out_dict
        except Exception:
            print('no states found, start from epoch 1, iter 0')
            return 0

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] has [%.3f M] parameters' % (name, num_params / 1e6))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad_(requires_grad)
