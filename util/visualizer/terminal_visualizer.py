import os
import time

from util import util
from util.visualizer.base_visualizer import BaseVisualizer


class TerminalVisualizer(BaseVisualizer):
    """This class stores the training results, images in HTML and losses in text file.
    """

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt):
        """Initialize the Visualizer class
        """
        self.opt = opt  # cache the option
        self.name = opt.exp_name
        self.win_size = opt.crop_size
        self.epoch = -1
        self.web_dir = os.path.join(opt.checkpoints_dir,  opt.project_name, opt.exp_name,opt.run_name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.project_name, opt.exp_name,opt.run_name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def update_state(self,epochs,iters,times):
        self.epochs = epochs
        self.iters = iters
        self.times = times

    def display_current_results(self, visuals):

        # save images to the disk
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (self.epochs, label))
            util.save_image(image_numpy, img_path)

        # # update website
        # webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
        # for n in range(self.epochs, 0, -1):
        #     webpage.add_header('epoch [%d]' % n)
        #     ims, txts, links = [], [], []

        #     for label, image_numpy in visuals.items():
        #         image_numpy = util.tensor2im(image)
        #         img_path = 'epoch%.3d_%s.png' % (n, label)
        #         ims.append(img_path)
        #         txts.append(label)
        #         links.append(img_path)
        #     webpage.add_images(ims, txts, links, width=self.win_size)
        # webpage.save()

    def display_current_videos(self, visuals):
        import imageio
        from torchvision.utils import make_grid
        for label, visual in visuals.items():
            frames = []
            path = os.path.join(self.web_dir, 'epoch%.3d_%s.gif' % (self.epochs, label))
            for frame in visual:
                image = util.tensor2im(make_grid(frame).unsqueeze(0))
                frames.append(image)
            imageio.mimsave(path, frames)

    # losses: same format as |losses| of plot_current_losses
    def plot_current_losses(self, losses):

        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (self.epochs, self.iters, self.times['comp'], self.times['data'])
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message