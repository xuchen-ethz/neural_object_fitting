import os
import sys

import argparse
import torch

import data
import models
from util import util
from util.visualizer.base_visualizer import BaseVisualizer as Visualizer


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--config', type=str)
        # basic parameters
        basic_args = parser.add_argument_group('basic')
        basic_args.add_argument('--project_name', type=str, default='project template',help='project name, use project folder name by default')
        basic_args.add_argument('--dataroot', type=str,help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        basic_args.add_argument('--run_name', type=str, default='', help='id of the experiment run, specified as string format, e.g. lr={lr} or string. Using current datetime by default')
        basic_args.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        basic_args.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        model_args = parser.add_argument_group('model')
        model_args.add_argument('--model', type=str, default='latent_object', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        model_args.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        model_args.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        model_args.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        model_args.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        data_args = parser.add_argument_group('data')
        data_args.add_argument('--dataset_mode', type=str, default='nocs_real', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        data_args.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        data_args.add_argument('--batch_size', type=int, default=1, help='input batch size')
        data_args.add_argument('--load_size', type=int, default=64, help='scale images to this size')
        data_args.add_argument('--crop_size', type=int, default=64, help='then crop to this size')
        data_args.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        data_args.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        data_args.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        data_args.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        data_args.add_argument('--keep_last', action='store_true', help='drop the last batch of the dataset to keep batch size consistent.')
        # additional parameters
        misc_args = parser.add_argument_group('misc')
        misc_args.add_argument('--load_suffix', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        misc_args.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        misc_args.add_argument('--visualizers', nargs='+', type=str, default=['terminal', 'wandb'], help='visualizers to use. local | wandb')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
    

        # get the basic options
        opt, _ = parser.parse_known_args()
        if opt.config is not None: opt = self.load_options(opt)
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, args = parser.parse_known_args()  # parse again with new defaults
        if opt.config is not None: opt = self.load_options(opt)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # modify visualization-related parser options
        parser = Visualizer.modify_commandline_options(parser)

        # save and return the parser
        self.parser = parser
        opt = parser.parse_args()
        if opt.config is not None: opt = self.load_options(opt)

        opt.exp_name = opt.category

        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_options(self,opt):
        output_dict = {}
        for group in self.parser._action_groups:
            if group.title in ['positional arguments', 'optional arguments']: continue
            output_dict[group.title] = {a.dest: getattr(opt, a.dest, None) for a in group._group_actions}

        import yaml

        if self.isTrain:
            output_path = os.path.join(opt.checkpoints_dir,opt.project_name,opt.exp_name, opt.run_name,'config.yaml')
        else:
            output_path = os.path.join(opt.results_dir, opt.project_name, opt.test_name, 'config.yaml')

        util.mkdirs(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            yaml.dump(output_dict,f,default_flow_style=False, sort_keys=True)

    def load_options(self,opt):
        assert(opt.config is not None)
        from envyaml import EnvYAML
        
        args_usr = [ arg[2:] for arg in sys.argv if '--' in arg]
        config = EnvYAML(opt.config,include_environment=False)
        for name in config.keys():
            # make sure yaml won't overwrite cmd input and the arg is defined
            basename = name.split('.')[-1]
            if basename not in args_usr and hasattr(opt,basename):
                setattr(opt, basename, config[name])

        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        opt.isTrain = self.isTrain   # train or test

        # process opt.run_name
        if opt.run_name != '':
            opt.run_name = opt.run_name.format(**vars(opt))
        else:
            from datetime import datetime
            opt.run_name = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        self.save_options(opt)

        if opt.verbose:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt