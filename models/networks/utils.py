import torch
import torch.nn.functional as F

def init_variable(dim, n_init, device, mode='random',range=[0,1],value=1):

    shape = (n_init,dim)
    var = torch.ones( shape,requires_grad=True,device=device,dtype=torch.float)
    if mode == 'random':
        var.data = torch.rand(shape,device=device) * (range[1]-range[0]) + range[0]
    elif mode == 'linspace':
        var.data = torch.linspace(range[0],range[1],steps=n_init,device=device).unsqueeze(-1)
    elif mode == 'constant':
        var.data =value*var.data
    else:
        raise NotImplementedError
    return var

def grid_sample(image, grid, mode='bilinear',padding_mode='constant',padding_value=1):
    image_out = F.grid_sample(image, grid, mode=mode, padding_mode='border')
    if padding_mode == 'constant':
        out_of_bound  = grid[:, :, :, 0] > 1
        out_of_bound += grid[:, :, :, 0] < -1 
        out_of_bound += grid[:, :, :, 1] > 1 
        out_of_bound += grid[:, :, :, 1] < -1
        out_of_bound = out_of_bound.unsqueeze(1).expand(image_out.shape)
        image_out[out_of_bound] = padding_value
    return image_out

def warping_grid(angle, transx, transy, scale, image_shape):
    cosz = torch.cos(angle)
    sinz = torch.sin(angle)
    affine_mat = torch.cat(  [cosz, -sinz, transx,
                              sinz,  cosz, transy], dim=1).view(image_shape[0], 2, 3)
    scale = scale.view(-1,1,1).expand(affine_mat.shape)
    return F.affine_grid(size=image_shape, theta=scale*affine_mat)

def set_axis(ax):
    ax.clear()
    ax.xaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y')