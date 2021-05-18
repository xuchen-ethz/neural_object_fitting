import numpy as np
import itertools
import torch
from scipy.spatial.transform import Rotation as scipy_rot

from models.networks import networks

from .base_model import BaseModel

class LatentObjectModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        models_args = parser.add_argument_group('models')

        models_args.add_argument('--z_dim', type=int, default=16, help='dimension of z')
        models_args.add_argument('--batch_size_vis', type=int, default=8, help='number of visualization samples')
        models_args.add_argument('--use_VAE', action='store_true', default=True, help='use KL divergence')
        models_args.add_argument('--category', type=str, default='laptop', help='object category')
        if is_train:
            models_args.add_argument('--lambda_recon', type=float, default=10., help='weight for reconstruction loss')
            models_args.add_argument('--lambda_KL', type=float, default=0.01, help='weight for the KL divergence')
        else:
            fitting_args = parser.add_argument_group('fitting')
            fitting_args.set_defaults(dataset_mode='nocs_hdf5', batch_size=1, no_flip=True, preprocess=' ')
            fitting_args.add_argument('--n_iter', type=int, default=50, help='number of optimization iterations')
            fitting_args.add_argument('--n_init', type=int, default=32, help='number of initializations')
            fitting_args.add_argument('--lambda_reg', type=float, default=1, help='weight for the KL divergence')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.use_VAE = opt.use_VAE 

        self.loss_names = ['G_recon']
        if self.opt.use_VAE > 0: self.loss_names += ['KL']

        self.visual_names = ['real_A','real_B','fake_B']

        self.video_names = ['anim_azim','anim_elev']

        self.model_names = ['G','E']

        self.optimizer_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.Generator(opt.z_dim).to(self.device)
        networks.init_net(self.netG, init_type=self.opt.init_type, init_gain=self.opt.init_gain,gpu_ids=self.gpu_ids)

        output_dim = opt.z_dim *2 if self.use_VAE else opt.z_dim
        self.netE = networks.Encoder(3, opt.crop_size, output_dim).to(self.device)
        self.netE = networks.add_SN(self.netE)
        networks.init_net(self.netE, init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.criterion_recon = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(),self.netE.parameters()), lr=opt.lr, betas=(0.5,0.999))
            self.optimizers.append(self.optimizer_G)

        # define the prior distribution`
        mu = torch.zeros(opt.z_dim, device=self.device)
        scale = torch.ones(opt.z_dim, device=self.device)
        self.z_dist = torch.distributions.Normal(mu, scale)

        self.batch_size_vis = opt.batch_size_vis

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.theta  = input['B_pose'].to(self.device)

    def forward(self):
        if self.use_VAE  == 0:
            self.z = self.netE(self.real_A)
        else:
            b = self.real_A.shape[0]
            output = self.netE(self.real_A)
            self.mu, self.logvar = output[:,:self.opt.z_dim],output[:,self.opt.z_dim:]
            std = self.logvar.mul(0.5).exp_()
            self.z_sample = self.z_dist.sample((b,))
            eps = self.z_sample
            self.z = eps.mul(std).add_(self.mu)

        self.fake_B = self.netG(self.z,self.theta)
        return self.fake_B

    def backward(self):
        self.loss_KL = (1 + self.logvar - self.mu.pow(2) - self.logvar.exp()).mean() * (-0.5 * self.opt.lambda_KL)
        self.loss_G_recon = self.criterion_recon(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_recon * self.opt.lambda_recon
        self.loss_G.backward()

    def optimize_parameters(self):
        self.train()
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def compute_visuals(self):
        self.netG.eval()
        self.real_A = self.real_A[:self.batch_size_vis,...]
        self.real_B = self.real_B[:self.batch_size_vis,...]
        self.fake_B = self.fake_B[:self.batch_size_vis,...]

        self.z_vis = self.netE(self.real_A)[:,:self.opt.z_dim]

        with torch.no_grad():
            self.anim_azim = []
            elev = 0
            for azim in range(-180,180,3):
                theta = torch.zeros((self.batch_size_vis,3)).to(self.device)
                theta[:,0],theta[:,1] = elev,azim
                frame = self.netG(self.z_vis,theta).detach().data
                self.anim_azim.append(frame)
            self.anim_elev= []
            azim = 0
            for elev in range(-90,90,3):
                theta = torch.zeros((self.batch_size_vis, 3)).to(self.device)
                theta[:, 0], theta[:, 1] = elev, azim
                frame = self.netG(self.z_vis, theta).detach().data
                self.anim_elev.append(frame)

    def fitting(self, real_B):
        import tqdm
        import torch.optim as optim
        import torch.nn.functional as F

        from models.networks.utils import grid_sample, warping_grid, init_variable
        from models.networks.losses import PerceptualLoss

        real_B = real_B.to(self.device).repeat((self.opt.n_init, 1, 1, 1))
        real_B = real_B[:, [2, 1, 0], :, :]

        ay = init_variable(dim=1, n_init=self.opt.n_init, device=self.device, mode='linspace', range=[-1/2,1/2])
        ax = init_variable(dim=1, n_init=self.opt.n_init, device=self.device, mode='constant', value=1/4)
        az = init_variable(dim=1, n_init=self.opt.n_init, device=self.device, mode='constant', value=0)
        s  = init_variable(dim=1, n_init=self.opt.n_init, device=self.device, mode='constant', value=1)
        tx = init_variable(dim=1, n_init=self.opt.n_init, device=self.device, mode='constant', value=0)
        ty = init_variable(dim=1, n_init=self.opt.n_init, device=self.device, mode='constant', value=0)
        z  = init_variable(dim=self.opt.z_dim, n_init=self.opt.n_init, device=self.device, mode='constant', value=0)

        latent = self.netE(F.interpolate(real_B, size=self.opt.crop_size, mode='nearest'))
        if self.opt.use_VAE:
            mu, logvar = latent[:, :self.opt.z_dim], latent[:, self.opt.z_dim:]
            std = logvar.mul(0.5).exp_()
            eps = self.z_dist.sample((self.opt.n_init,))
            z.data = eps.mul(std).add_(mu)
        else:
            z.data = latent

        variable_dict = [
            {'params': z, 'lr': 3e-1},
            {'params': ax, 'lr': 1e-2},
            {'params': ay, 'lr': 3e-2},
            {'params': az, 'lr': 1e-2},
            {'params': tx, 'lr': 3e-2},
            {'params': ty, 'lr': 3e-2},
            {'params': s,  'lr': 3e-2},
        ]
        optimizer = optim.Adam(variable_dict,betas=(0.5,0.999))

        losses = [('VGG', 1, PerceptualLoss(reduce=False))]
        reg_creterion = torch.nn.MSELoss(reduce=False)

        loss_history = np.zeros( (self.opt.n_init,self.opt.n_iter,len(losses)+1))
        state_history = np.zeros( (self.opt.n_init,self.opt.n_iter,6 + self.opt.z_dim))
        image_history = []

        for iter in tqdm.tqdm(range(self.opt.n_iter)):

            optimizer.zero_grad()

            angle = 180 * torch.cat([ax, ay, torch.zeros_like(ay)], dim=1)
            fake_B = self.netG(z,angle)

            grid = warping_grid(az * np.pi, tx, ty, s, fake_B.shape)
            fake_B = grid_sample(fake_B, grid)

            fake_B_upsampled = F.interpolate(fake_B, size=real_B.shape[-1], mode='bilinear')
            
            error_all = 0
            for l, (name, weight, criterion)in enumerate(losses):
                error = weight * criterion(fake_B_upsampled, real_B).view(self.opt.n_init,-1).mean(1)
                loss_history[:,iter,l] = error.data.cpu().numpy()
                error_all = error_all + error

            error = self.opt.lambda_reg * reg_creterion(z,torch.zeros_like(z)).view(self.opt.n_init,-1).mean(1)
            loss_history[:, iter, l+1] = error.data.cpu().numpy()
            error_all = error_all + error

            error_all.mean().backward()

            optimizer.step()
            image_history.append(fake_B)

            state_history[:, iter, :3] = 180*torch.cat([-ay-0.5, ax+1, -az],dim=-1).data.cpu().numpy()
            state_history[:, iter, 3:] = torch.cat([tx, ty, s, z],dim=-1).data.cpu().numpy()

        return state_history, loss_history, image_history

    def visulize_fitting(self, real_B, RT_gt, state_history, loss_history, image_history):
        import matplotlib.pyplot as plt
        from util.util import tensor2im
        from models.networks.utils import set_axis
        import matplotlib
        matplotlib.use('TkAgg')

        RT_gt = RT_gt.numpy()[0]
        R_gt = RT_gt[:3, :3]
        real_B_img = tensor2im(real_B)

        n_init, n_iter, n_loss = loss_history.shape

        fig, axes = plt.subplots(nrows=loss_history.shape[2] + 2, ncols=n_init + 1, sharey='row')
        axes[0, -1].clear();axes[0, -1].axis('off')
        axes[0, -1].imshow(real_B_img)
        plt.ion()

        plots = axes.copy()
        for row in range(axes.shape[0]):
            for col in range(axes.shape[1]):
                if row == 0:
                    axes[row, col].axis('off')
                    plots[row, col] = axes[row, col].imshow(real_B_img)
                elif col < n_init:
                    if row < n_loss+1:
                        set_axis(axes[row, col])
                        plots[row, col] = axes[row, col].plot(np.arange(n_iter),loss_history[col, :,row-1])
                    else:
                        plots[row, col] = axes[row, col].plot(np.arange(n_iter),60*np.ones(n_iter))
                        axes[row, col].set_ylim([0,60])

        errors = np.zeros((n_init,n_iter))

        for iter in range(n_iter):
            for init in range(n_init):
                pose = state_history[init,iter,:3]
                R_pd = scipy_rot.from_euler('yxz', pose, degrees=True).as_dcm()[:3, :3]
                
                R_pd = R_pd[:3, :3]/np.cbrt(np.linalg.det(R_pd[:3, :3]))
                R_gt = R_gt[:3, :3]/np.cbrt(np.linalg.det(R_gt[:3, :3]))

                R = R_pd @ R_gt.transpose()
                errors[init,iter] = np.arccos((np.trace(R) - 1)/2) * 180/np.pi

            ranking = [r[0] for r in sorted(enumerate(loss_history[:, iter,:].mean(-1)), key=lambda r: r[1])]

            for r, b in enumerate(ranking[::-1]):
                plots[0, r].set_data(tensor2im(image_history[iter][b].unsqueeze(0)))
                for l  in range(loss_history.shape[2]):
                    plots[l + 1, r][0].set_data(np.arange(iter),loss_history[b, :iter,l])
                plots[-1, r][0].set_data(np.arange(iter),  errors[b, :iter])

            plt.draw()

            plt.pause(0.01)
        plt.close(fig)