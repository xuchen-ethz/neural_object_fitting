import numpy as np
import torch
from torch.nn import init
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from models.networks import spectral_norm

class Generator(nn.Module):
    def __init__(self, z_dim, euler_seq='zyx', **kwargs):
        super().__init__()
        self.shape_code = nn.Parameter(0.02*torch.randn(1,512,4,4,4),requires_grad=True)
        # Upsampling 3D
        self.enc_1 = nn.Sequential(*[nn.ConvTranspose3d(512,128,kernel_size=4,stride=2, padding=1)])
        self.enc_2 = nn.Sequential(*[nn.ConvTranspose3d(128,64, kernel_size=4, stride=2, padding=1)])
        # Projection
        self.proj = nn.Sequential(*[nn.ConvTranspose2d(64*16,64*16, kernel_size=1,stride=1)])
        # Upsampling 2D
        self.enc_3 = nn.Sequential(*[nn.ConvTranspose2d(64*16,64*4,kernel_size=4,stride=2, padding=1)])
        self.enc_4 = nn.Sequential(*[nn.ConvTranspose2d(64*4,64,kernel_size=4,stride=2, padding=1)])
        self.enc_5 = nn.Sequential(*[nn.ConvTranspose2d(64,3,kernel_size=3,stride=1,padding=1)])
        # MLP for AdaIN
        self.mlp0 = LinearBlock(z_dim,512*2,activation='relu')
        self.mlp1 = LinearBlock(z_dim,128*2,activation='relu')
        self.mlp2 = LinearBlock(z_dim,64*2,activation='relu')
        self.mlp3 = LinearBlock(z_dim,256*2,activation='relu')
        self.mlp4 = LinearBlock(z_dim,64*2,activation='relu')

        self.euler_seq = euler_seq

    def forward(self, z,  angle, a=None, debug=False):
        b,_ = z.size()
        angle = angle / 180. * np.pi
        # Upsampling 3D
        h0 = self.shape_code.expand(b, 512, 4, 4, 4).clone()
        a0 = self.mlp0(z)
        h0 = actvn( adaIN(h0,a0) )

        h1 = self.enc_1(h0)
        a1 = self.mlp1(z)
        h1 = actvn( adaIN(h1,a1) )

        h2 = self.enc_2(h1)
        a2 = self.mlp2(z)
        h2 = actvn(adaIN(h2, a2))

        # Rotation
        h2_rot = rot(h2,angle,euler_seq=self.euler_seq,padding="border")
        b,c,d,h,w = h2_rot.size()
        h2_2d = h2_rot.contiguous().view(b,c*d,h,w)
        h2_2d = actvn(self.proj(h2_2d))
        # Upsampling 2D
        h3 = self.enc_3(h2_2d)
        a3 = self.mlp3(z)
        h3 = actvn(adaIN(h3, a3))

        h4 = self.enc_4(h3)
        a4 = self.mlp4(z)
        h4 = actvn(adaIN(h4, a4))

        h5 = self.enc_5(h4)
        return F.tanh(h5)
def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

class Encoder(nn.Module):
    def __init__(self,in_dim=3, in_size=64, z_dim=128):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(  in_dim,  64, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d( 64, 128, 3, 2, 1), nn.LeakyReLU(0.2), nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(0.2), nn.InstanceNorm2d(256),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU(0.2), nn.InstanceNorm2d(512),
        ])
        self.enc_out = nn.Sequential(*[
            nn.Linear((in_size//16)**2*512,128), nn.LeakyReLU(0.2),
            nn.Linear(128,z_dim), nn.Tanh()
        ])
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.model.forward(x).view(b,(h//16)**2*512)
        enc = self.enc_out(x)
        return enc

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

def rot(x,angle,euler_seq='xyz',padding='zeros'):
    b,c,d,h,w = x.shape
    grid = set_id_grid(x)
    grid_flat = grid.reshape(b, 3, -1)
    grid_rot_flat = euler2mat(angle,euler_seq=euler_seq).bmm(grid_flat)
    grid_rot = grid_rot_flat.reshape(b,3,d,h,w)
    x_rot = F.grid_sample(x,grid_rot.permute(0,2,3,4,1),padding_mode=padding,mode='bilinear')
    return x_rot

def euler2mat(angle, euler_seq='xyz' ):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    zeros = z.detach()*0
    ones = zeros.detach()+1

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    if euler_seq == 'xyz':
        rotMat = xmat.bmm(ymat).bmm(zmat)
    elif euler_seq == 'zyx':
        rotMat = zmat.bmm(ymat).bmm(xmat)
    return rotMat


def set_id_grid(x):
    b, c, d, h, w = x.shape
    z_range = (torch.linspace(-1,1,steps=d)).view(1, d, 1, 1).expand(1, d, h, w).type_as(x)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h)).view(1, 1, h, 1).expand(1, d, h, w).type_as(x)  # [1, H, W, D]
    x_range = (torch.linspace(-1,1,steps=w)).view(1, 1, 1, w).expand(1, d, h, w).type_as(x)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0)[None,...]  # x,y,z
    grid = grid.expand(b,3,d,h,w)
    return grid

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4 or len(size) == 5)
    N, C = size[:2]

    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    if len(size)==5:
        feat_std = feat_std.unsqueeze(-1)
        feat_mean = feat_mean.unsqueeze(-1)

    return feat_mean, feat_std


def adaIN(content_feat, style_mean_std):
    assert(content_feat.size(1) == style_mean_std.size(1)/2)
    size = content_feat.size()
    b,c = style_mean_std.size()
    style_mean, style_std = style_mean_std[:,:c//2],style_mean_std[:,c//2:]

    style_mean = style_mean.unsqueeze(-1).unsqueeze(-1)
    style_std = style_std.unsqueeze(-1).unsqueeze(-1)
    if len(size)==5:
        style_mean = style_mean.unsqueeze(-1)
        style_std = style_std.unsqueeze(-1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def add_SN(m):
    for name, c in m.named_children():
        m.add_module(name, add_SN(c))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return spectral_norm.spectral_norm(m)#nn.utils.spectral_norm(m)
    else:
        return m

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler