# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model_wbg import BbStyleGenerator as Generator, Discriminator
from models.discriminator import LatentCodesDiscriminator

'''
# 在stylegan2生成器的基础上，训练一个encoder 和 bgencoder，bgencoder作为noise（+）到 stylegan2中，使psp能够完成结构inpainting
maskedimg-------->F&W-pspencoder（training Ranger）---->w------>stylegan2_wbg（pre-trained）------->inpainted result
   |                                                               | noise
    ---------------------------------------------------------------

'''

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts, istrain=True):
        super(pSp, self).__init__()
        self.opts = opts
        self.latent_avg = None
        # self.bgencoder = bgencoder(opts.stylegan_size)
        self.encoder = self.set_encoder()

        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.latent_discriminator = LatentCodesDiscriminator(512, 4)
        self.gan_discriminator = Discriminator(opts.stylegan_size, channel_multiplier=2)
        #         self.face_pool = torch.nn.AdaptiveAvgPool2d((128, 128))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((opts.stylegan_size, opts.stylegan_size))

        # Load weights if needed
        if istrain:
            self.load_weights()

    def set_encoder(self):
        self.opts.encoder_type = 'Encoder4Editing'
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            # self.bgencoder.load_state_dict(get_keys(ckpt, 'bgencoder'), strict=True)
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.gan_discriminator.load_state_dict(get_keys(ckpt, 'gan_discriminator'), strict=True)
            self.latent_discriminator.load_state_dict(get_keys(ckpt, 'latent_discriminator'), strict=True)

            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load('chkpt/model_ir_se50.pth')
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.gan_discriminator.load_state_dict(ckpt['d'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)



    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, return_features=False, alpha=None):
        if input_code:
            codes = x
        else:
            # print(x.size())
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    # print(codes, self.latent_avg,self.opts.device)
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code

        images, result_latent = self.decoder([codes],x,
                                             input_is_latent=input_is_latent,
                                             return_latents=return_latents,return_features=return_features)

        if resize:
            images = self.face_pool(images)

        if return_latents or return_features:
            #         if return_latents:

            return images, result_latent
        else:
            return images

    def get_ef_and_defeature(self,x):
        codes,en_features = self.encoder(x,fout=True)
        # print('codes',codes.size())
        if self.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        images, de_features = self.decoder([codes], x,
                     input_is_latent=True,
                     return_latents=False,
                     return_features = True
                     )


        if len(en_features)<len(de_features):
            tempde = de_features[::-1]
            for i in range(len(en_features),len(de_features)):
                en_features.append(tempde[i])

        return images, en_features[::-1],de_features

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                self.latent_avg = self.decoder.generator.mean_latent(10000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
