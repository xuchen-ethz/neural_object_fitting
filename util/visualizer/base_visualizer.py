import importlib
import os

import numpy as np

from util import util

from cv2 import resize

class BaseVisualizer():
    @staticmethod
    def modify_commandline_options(parser):
        opt, _ = parser.parse_known_args()

        for vis_name in opt.visualizers:
            vis_filename = "util.visualizer." + vis_name + "_visualizer"
            vislib = importlib.import_module(vis_filename)
            vis = None
            target_vis_name = vis_name + 'visualizer'
            for name, cls in vislib.__dict__.items():
                if name.lower() == target_vis_name.lower() \
                        and issubclass(cls, BaseVisualizer):
                    vis = cls

            if vis is None:
                print(
                    "In %s.py, there should be a subclass of BaseVisualizer with class name that matches %s in lowercase." % (
                    vis_filename, target_vis_name))
                exit(0)

            parser = vis.modify_commandline_options(parser)

        return parser

    def __init__(self, opt):
        self.visualizer_list = []

        for vis_name in opt.visualizers:
            vis_filename = "util.visualizer." + vis_name + "_visualizer"
            vislib = importlib.import_module(vis_filename)
            vis = None
            target_vis_name = vis_name + 'visualizer'
            for name, cls in vislib.__dict__.items():
                if name.lower() == target_vis_name.lower() \
                        and issubclass(cls, BaseVisualizer):
                    vis = cls

            if vis is None:
                print(
                    "In %s.py, there should be a subclass of BaseVisualizer with class name that matches %s in lowercase." % (
                    vis_filename, target_vis_name))
                exit(0)

            self.visualizer_list.append(vis(opt))

    def update_state(self,epochs,iters,times):
        for visualizer in self.visualizer_list:
            visualizer.update_state(epochs,iters,times)

    def display_current_results(self, visuals):
        """Display current results on visdom; save current results to an HTML file.

            Parameters:
                visuals (OrderedDict) - - dictionary of images to display or save
                epoch (int) - - the current epoch
                save_result (bool) - - if save the current results to an HTML file
        """
        for visualizer in self.visualizer_list:
            visualizer.display_current_results(visuals)

    def display_current_videos(self, visuals):
        """Display current results on visdom; save current results to an HTML file.

            Parameters:
                visuals (OrderedDict) - - dictionary of images to display or save
                epoch (int) - - the current epoch
                save_result (bool) - - if save the current results to an HTML file
        """
        for visualizer in self.visualizer_list:
            visualizer.display_current_videos(visuals)

    def plot_current_losses(self, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        # losses: same format as |losses| of plot_current_losses
        for visualizer in self.visualizer_list:
            visualizer.plot_current_losses(losses)




# TODO merge image saver for test and training time
def save_images(webpage, visuals, name, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()

    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = resize(im, (h, int(w * aspect_ratio)), interpolation='bicubic')
        if aspect_ratio < 1.0:
            im = resize(im, (int(h / aspect_ratio), w), interpolation='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def get_img_from_fig(fig, dpi=180):
    import io
    import cv2
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
