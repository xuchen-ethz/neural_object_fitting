"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer.base_visualizer import BaseVisualizer as Visualizer
import copy


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    print('------------- Creating Dataset ----------------')
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.dataset_size = len(dataset) # get the number of images in the dataset.
    print('-----------------------------------------------\n')

    print('-------------- Creating Model -----------------')
    model = create_model(opt)      # create a model given opt.model and other options
    iter_start = model.setup(opt)               # regular setup: load and print networks; create schedulers
    print('train from [Iter %d]' % (iter_start))
    print('-----------------------------------------------\n')

    print('------------ Creating Visualizer --------------')
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    print('-----------------------------------------------\n')


    print('--------------- Start Training -----------------')
    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates at the end of every epoch. TODO: moved from end to the begining, check the consequence

        for i, data in enumerate(dataset):  # inner loop within one epoch

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if total_iters < iter_start: continue  # skip until the starting iteration

            times = {}                     # recording compuation time
            iter_start_time = time.time()  # timer for computation per iteration
            times['data'] = iter_start_time - iter_data_time

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            times['comp'] = time.time() - iter_start_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visualizer.update_state(epoch, total_iters,times=times)
                visualizer.display_current_results(model.get_current_visuals())
                visualizer.display_current_videos(model.get_current_videos())

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.update_state(epoch, total_iters,times=times)
                visualizer.plot_current_losses(losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save('latest', iter=total_iters)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save("%d" % total_iters, iter=total_iters)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))