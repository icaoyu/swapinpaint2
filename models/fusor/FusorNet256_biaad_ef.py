import torch
import torch.nn as nn
import torch.nn.functional as F

# 256 z_id . outputsize:(b,3,128,128)
class MultilevelAttributesEncoder(nn.Module):
    def __init__(self):
        super(MultilevelAttributesEncoder, self).__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512, 512]
        self.Encoder = nn.ModuleDict()

        self.Encoder.add_module(name=f'layer_0', module=nn.Sequential(
            nn.UpsamplingBilinear2d(size=(128, 128)),
            nn.Conv2d(self.Encoder_channel[0], self.Encoder_channel[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Encoder_channel[1]),
            nn.LeakyReLU(0.1)
        ))
        for i in range(1, 6):
            self.Encoder.add_module(name=f'layer_{i}', module=nn.Sequential(
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Encoder_channel[i + 1]),
                nn.LeakyReLU(0.1)
            ))

        self.Decoder_inchannel = [512, 1024, 512, 256, 128]
        self.Decoder_outchannel = [512, 256, 128, 64, 32]

        self.Decoder = nn.ModuleDict({f'layer_{i}': nn.Sequential(
            nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(self.Decoder_outchannel[i]),
            nn.LeakyReLU(0.1)
        ) for i in range(5)})

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Upsample112 = nn.UpsamplingBilinear2d(size=(112, 112))

    def forward(self, x):
        arr_x = []
        for i in range(6):
            x = self.Encoder[f'layer_{i}'](x)
            arr_x.append(x)
        arr_y = []
        arr_y.append(arr_x[5])
        y = arr_x[5]
        for i in range(5):
            y = self.Decoder[f'layer_{i}'](y)
            y = torch.cat((y, arr_x[4 - i]), 1)
            arr_y.append(y)
        y = self.Upsample(y)
        arr_y.append(y)

        return arr_y


class BIADD(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, z_id_size=256):
        super(BIADD, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(3, z_inchannel, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.oconv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.oconv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id, occ, mask, use_align=False):
        h_bar = self.BNorm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))
        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        i = r_id * h_bar + beta_id

        actv = self.mlp_shared(occ)
        r_occ = self.oconv1(actv)
        beta_occ = self.oconv2(actv)

        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)

        mask = mask.expand_as(r_occ)
        # print(mask.size())

        r_att = r_att * mask + r_occ * (1 - mask)
        beta_att = beta_att * mask + beta_occ * (1 - mask)

        a = r_att * h_bar + beta_att
        h_out = (1 - m) * a + m * i
        return h_out


class BIADD3(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, en_inchannel, z_id_size=256):
        super(BIADD3, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.oconv1 = nn.Conv2d(en_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.oconv2 = nn.Conv2d(en_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id, z_en, mask):
        h_bar = self.BNorm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))
        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        i = r_id * h_bar + beta_id

        # actv = self.mlp_shared(occ)
        actv = z_en
        r_occ = self.oconv1(actv)
        beta_occ = self.oconv2(actv)

        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        # print(mask.size(),r_occ.size())
        mask = mask.expand_as(r_occ)
       

        r_att = r_att * mask + r_occ * (1 - mask)
        beta_att = beta_att * mask + beta_occ * (1 - mask)

        a = r_att * h_bar + beta_att
        h_out = (1 - m) * a + m * i
        return h_out


class SPADE(nn.Module):
    def __init__(self, h_inchannel, z_inchannel):
        super(SPADE, self).__init__()

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(3, z_inchannel, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att):
        h_bar = self.BNorm(h_in)
        x = z_att
        x = self.mlp_shared(x)
        r_att = self.conv1(x)
        beta_att = self.conv2(x)
        h_out = r_att * h_bar + beta_att
        return h_out


class ADD(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, z_id_size=256):
        super(ADD, self).__init__()
        # print(z_inchannel)
        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id):
        h_bar = self.BNorm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))
        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        i = r_id * h_bar + beta_id
        # print(z_att.size())
        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        a = r_att * h_bar + beta_att
        h_out = (1 - m) * a + m * i

        return h_out


class BIADDResBlock3(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, en_inchannel, h_outchannel):
        super(BIADDResBlock3, self).__init__()

        self.h_inchannel = h_inchannel
        self.z_inchannel = z_inchannel
        self.h_outchannel = h_outchannel
        self.add1 = ADD(h_inchannel, z_inchannel)
        self.add2 = ADD(h_inchannel, z_inchannel)

        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.add3 = BIADD3(h_inchannel, z_inchannel, en_inchannel, z_id_size=256)
        self.conv3 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, h_in, z_att, z_id, z_en,mask):
        mask = F.interpolate(mask,size=h_in.size()[2],mode='nearest')
        x1 = self.activation(self.add1(h_in, z_att, z_id))
        x1 = self.conv1(x1)
        x1 = self.activation(self.add2(x1, z_att, z_id))
        x1 = self.conv2(x1)
        x2 = h_in
        if not self.h_inchannel == self.h_outchannel:
            x2 = self.activation(self.add3(h_in, z_att, z_id, z_en,mask))
            x2 = self.conv3(x2)
        return x1 + x2


class BIADDGenerator(nn.Module):
    def __init__(self, z_id_size, normtype=True):
        super(BIADDGenerator, self).__init__()
        # self.sw, self.sh = 4, 4
        # self.fc = nn.Conv2d(3, 512, 3, padding=1)
        self.convt = nn.ConvTranspose2d(z_id_size, 512, kernel_size=4, stride=1, padding=0)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.h_inchannel = [512, 512, 512, 256, 128, 64, 32]
        self.z_inchannel = [512, 512, 512, 512, 512, 256, 128]
        self.h_outchannel = [512, 512, 256, 128, 64, 32, 16]
        self.e_inchannel = [512, 512, 512, 256, 128, 64, 64]
        self.model = nn.ModuleDict(
            {f"layer_{i}": BIADDResBlock3(self.h_inchannel[i], self.z_inchannel[i], self.e_inchannel[i],
                                         self.h_outchannel[i])
             for i in range(7)})
        self.convo = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        if normtype:

            self.out = nn.Tanh()
        else:
            self.out = nn.Sigmoid()

    def forward(self, z_id, z_att, z_en,mask):
        mask = mask[:,0:1,:,:]
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))

        for i in range(6):
            x = self.model[f"layer_{i}"](x, z_att[i], z_id, z_en[i],mask)
            x = self.Upsample(x)
        x = self.model["layer_6"](x, z_att[6], z_id, z_en[6],mask)
        x = self.convo(x)
        return self.out(x)


if __name__ == '__main__':
    z_id = torch.rand(size=(1, 256))
    occ = torch.rand(size=(1, 3, 256, 256))
    mask = torch.rand(size=(1, 1, 256, 256))
    z_att = []
    en_att = []

    z_att.append(torch.rand((1, 512, 4, 4)))
    z_att.append(torch.rand((1, 512, 8, 8)))
    z_att.append(torch.rand((1, 512, 16, 16)))
    z_att.append(torch.rand((1, 512, 32, 32)))
    z_att.append(torch.rand((1, 512, 64, 64)))
    z_att.append(torch.rand((1, 256, 128, 128)))
    z_att.append(torch.rand((1, 128, 256, 256)))

    en_att.append(torch.rand(1, 512, 4, 4))
    en_att.append(torch.rand(1, 512, 8, 8))
    en_att.append(torch.rand(1, 512, 16, 16))
    en_att.append(torch.rand(1, 256, 32, 32))
    en_att.append(torch.rand(1, 128, 64, 64))
    en_att.append(torch.rand(1, 64, 128, 128))
    en_att.append(torch.rand(1, 64, 256, 256))

    x = torch.rand((1, 3, 128, 128))
    net = BIADDGenerator(256)
    netout = net(z_id, z_att, en_att)
    print(netout.size())

