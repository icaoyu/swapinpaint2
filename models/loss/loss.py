import numpy as np

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)






class SP2_Loss(nn.Module):
    def __init__(self, hp ):
        super(SP2_Loss, self).__init__()

        self.vgg_weight = hp.model.lambda_vgg
        self.id_weight = hp.model.lambda_id
        self.rec_weight = hp.model.lambda_rec
        self.mask_rec_weight = 0
        self.vgg_div = hp.model.vgg_div
#         self.vgg_weight = 1
#         self.id_weight = 5
#         self.rec_weight = 5

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.criterionVGG = VGGLoss(vgg_div=self.vgg_div)

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def rec_loss_mask(self,X,Y,mask):
        return self.l1(X*(1-mask),Y*(1-mask))
    
    def att_loss(self, z_att_X, z_att_Y):
        loss = 0
        for i in range(8):
            loss += self.l2(z_att_X[i], z_att_Y[i])
        return loss
        

    def rec_loss(self, X, Y, same):
        # lossm = self.l1(X,Y)
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return self.l1(X, Y)

    def forward(self, X, Y, z_id_X, z_id_Y,mask,same):
        loss_dict = {}
        vgg_loss = self.criterionVGG(X, Y) # for attr loss
        id_loss = self.id_loss(z_id_X, z_id_Y) #for id loss
        rec_loss = self.rec_loss(X, Y, same) #for 
        rec_mask_loss = self.rec_loss_mask(X,Y,mask)
        loss_dict['vgg_loss'] = self.vgg_weight * float(vgg_loss)
        loss_dict['id_loss'] = self.id_weight *float(id_loss)
        loss_dict['rec_loss'] = self.rec_weight *float(rec_loss)
        loss_dict['rec_m_loss'] = self.mask_rec_weight *float(rec_mask_loss)
        loss = self.vgg_weight * vgg_loss + self.id_weight * id_loss + self.rec_weight * rec_loss + self.mask_rec_weight * rec_mask_loss
        loss_dict['loss'] = float(loss)
        return loss, loss_dict


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1,h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self,vgg_div=1.0):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.MSELoss()
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#         self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0]
        # self.weights = [1.0/4, 1.0/2, 1.0, 1.0, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(0, len(self.weights)):
            w = self.weights[i]
            w = w / vgg_div
            if i > 0:
                for j in range(1, i + 1):
                    if w < 1.0:
                        w = w * 2
            self.weights[i] = w
        print(self.weights)


    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
