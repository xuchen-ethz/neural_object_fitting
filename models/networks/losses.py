
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from math import exp

class PerceptualLoss(nn.Module):

    def __init__(self,type='l2',reduce=True,final_layer=14):
        super(PerceptualLoss, self).__init__()
        self.model = self.contentFunc(final_layer=final_layer)
        self.model.eval()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
        self.type = type
        if type == 'l1':
            self.criterion = torch.nn.L1Loss(reduce=reduce)
        elif type == 'l2':
            self.criterion = torch.nn.MSELoss(reduce=reduce)
        elif type == 'both':
            self.criterion1 = torch.nn.L1Loss(reduce=reduce)
            self.criterion2 = torch.nn.MSELoss(reduce=reduce)
        else:
            raise NotImplementedError
        
    def normalize(self, tensor):
        tensor = (tensor+1)*0.5
        tensor_norm = (tensor-self.mean.expand(tensor.shape))/self.std.expand(tensor.shape)
        return tensor_norm

    def contentFunc(self,final_layer=14):
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == final_layer:
                break
        return model

    def forward(self, fakeIm, realIm):
        f_fake = self.model.forward(self.normalize(fakeIm))
        f_real = self.model.forward(self.normalize(realIm))
        if self.type == 'both':
            loss = self.criterion1(f_fake, f_real.detach())+self.criterion2(f_fake, f_real.detach())
        else:
            loss = self.criterion(f_fake, f_real.detach())
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, reduce=True,negative=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.reduce = reduce
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.negative = negative

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        if self.negative:
            return -_ssim(img1, img2, window, self.window_size, channel, self.reduce)
        else:
            return _ssim(img1, img2, window, self.window_size, channel, self.reduce)


def ssim(img1, img2, window_size=11, reduce=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, reduce)