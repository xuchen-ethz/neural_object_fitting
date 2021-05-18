import os

import numpy as np
import wandb

from util import util
from util.visualizer.base_visualizer import BaseVisualizer


class WandbVisualizer(BaseVisualizer):
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'wandb' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt):
        self.opt = opt  # cache the option
        config_file = os.path.join(opt.checkpoints_dir,opt.project_name, opt.exp_name, opt.run_name, 'config.yaml')
        import yaml
        with open(config_file, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
        wandb.init(project=opt.project_name, name=opt.run_name, group=opt.exp_name, config=config)

    def update_state(self,epochs,iters,times):
        self.epochs = epochs
        self.iters = iters
        self.times = times

    def display_current_results(self, visuals):
        from torchvision.utils import make_grid
        visual_wandb = {}
        for key, image_tensor in visuals.items():
            visual_wandb[key] = wandb.Image(util.tensor2im(make_grid(image_tensor).unsqueeze(0)))
        wandb.log(visual_wandb,step=self.iters)

    def display_current_videos(self, visuals):
        from torchvision.utils import make_grid
        video_wandb = {}
        for label, visual in visuals.items():
            frames = []
            for frame in visual:
                image = util.tensor2im(make_grid(frame).unsqueeze(0))
                frames.append(image)
            gif = np.stack(frames, axis=0)
            gif = np.transpose(gif, (0, 3, 1, 2))

            video_wandb[label] = wandb.Video(gif, fps=20)
            print('hello')

        wandb.log(video_wandb,step=self.iters)

    def plot_current_losses(self, losses):
        wandb.log(losses,step=self.iters)