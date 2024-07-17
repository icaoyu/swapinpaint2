import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torch.nn.functional as F
from .psp3 import pSp
from .fusor.CBAM import CBAMResNet
from .fusor.FusorNet256_biaad_ef import BIADDGenerator
from .fusor.MultiScaleDiscriminator import MultiscaleDiscriminator
from argparse import Namespace

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

'''
id_sampler: resnet50
'''
class ID_Inpaintor(nn.Module):
    def __init__(self, hp):
        super(ID_Inpaintor, self).__init__()
        self.hp = hp
        self.G = BIADDGenerator(hp.arcface.vector_size)
        self.D = MultiscaleDiscriminator(input_nc=3,getIntermFeat=self.hp.model.getIntermFeat)
        self.Z = CBAMResNet(50, feature_dim=hp.arcface.vector_size, mode='ir')  # resnet50
        self.A = None
        self.load_weights()


    def forward(self, occ_img, id_img,mask=None):
        z_id = self.Z(F.interpolate(id_img[:,:,23:233,23:233], size=112, mode='bilinear',align_corners=True))
        z_id = F.normalize(z_id)
        imgout,en_map,attribute_map = self.A.get_ef_and_defeature(occ_img)
        output = self.G(z_id.detach(),attribute_map,en_map,mask)
        output_z_id = self.Z(F.interpolate(output[:,:,23:233,23:233], size=112, mode='bilinear',align_corners=True))
        output_z_id = F.normalize(output_z_id)
        return output,imgout, z_id, output_z_id

    def load_weights(self):
        if self.hp.log.resume_training_from_ckpt is not None:
            print('Loading inpaintor from checkpoint: {}'.format(self.hp.log.resume_training_from_ckpt))
            ckpt = torch.load(self.hp.log.resume_training_from_ckpt, map_location='cpu')
            opts = ckpt['opts']
            opts = Namespace(**opts)
            opts.device = self.hp.model.device
            self.A = pSp(opts, istrain=False)
            self.A.load_state_dict(get_keys(ckpt, 'A'), strict=True)
            self.G.load_state_dict(get_keys(ckpt, 'G'), strict=True)
            self.Z.load_state_dict(get_keys(ckpt, 'Z'), strict=True)
            self.D.load_state_dict(get_keys(ckpt, 'D'), strict=True)
            self.A.latent_avg = ckpt['latent_avg'].to(self.hp.model.device)
            # self.__load_latent_avg(ckpt)
        else:
            self.Z.load_state_dict(torch.load(self.hp.arcface.chkpt_path, map_location='cpu'))
            print('Loading z weights from pretrained!')
            ckpt = torch.load(self.hp.psp.chkpt_path, map_location='cpu')
            opts = ckpt['opts']
            opts = Namespace(**opts)
            opts.device = self.hp.model.device
            opts.checkpoint_path = self.hp.psp.chkpt_path
            self.A = pSp(opts, istrain=True)
#             self.A.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
#             self.A.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
#             self.A.__load_latent_avg(ckpt)
            # self.__load_latent_avg(ckpt, repeat=self.A.encoder.style_count)
            # print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.hp.psp.chkpt_path))

#     def __load_latent_avg(self, ckpt, repeat=None):
#         if 'latent_avg' in ckpt:
#             self.A.latent_avg = ckpt['latent_avg'].to(self.pspopts.device)
#         elif self.pspopts.start_from_latent_avg:
#             # Compute mean code based on a large number of latents (10,000 here)
#             with torch.no_grad():
#                 self.A.latent_avg = self.A.decoder.mean_latent(10000).to(self.pspopts.device)
#         else:
#             self.A.latent_avg = None
#         if repeat is not None and self.A.latent_avg is not None:
#             self.A.latent_avg = self.A.latent_avg.repeat(repeat, 1)


# if __name__=='__main__':
#     z_id = torch.rand(size=(1, 256))
#     z_att = []
#     z_att.append(torch.rand((1,512,2,2)))
#     z_att.append(torch.rand((1,1024,4,4)))
#     z_att.append(torch.rand((1,512,8,8)))
#     z_att.append(torch.rand((1,256,16,16)))
#     z_att.append(torch.rand((1,128,32,32)))
#     z_att.append(torch.rand((1,64,64,64)))
#     z_att.append(torch.rand((1,64,112,112)))
#     x =torch.rand((1,3,128,128))
#     net = ID_Inpaintor()
#     attrnet = MultilevelAttributesEncoder()
#     netout = net(z_id,attrnet(x))
#     print(netout.size())

