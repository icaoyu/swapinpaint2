import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from utils import common, train_utils
from models.loss.loss import SP2_Loss, GANLoss

from models.ID_inpaintor256_biaad_ef_wbg import ID_Inpaintor
from datasets.celeba_id_dataset256 import CelebaDataset
from training.ranger import Ranger

random.seed(0)
torch.manual_seed(0)


class Coach:
    def __init__(self, hp, prev_train_checkpoint=None):
        self.hp = hp
        self.global_step = 0
        self.device = self.hp.model.device
        self.device = torch.device(self.hp.model.device if torch.cuda.is_available() else "cpu" )
        torch.cuda.set_device(int(self.hp.model.device.split(':')[1]))
        print(self.device)
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor
        # Initialize network
        self.net = ID_Inpaintor(self.hp).to(self.device)
        #         self.net.A.latent_avg.to(self.device)
        self.gan_loss = GANLoss()
        self.id_loss = SP2_Loss(self.hp)
        self.criterionFeat = torch.nn.L1Loss()
        # self.mse_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.g_optimizer, self.d_optimizer = self.configure_optimizers()
        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.hp.model.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.hp.model.num_workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.hp.model.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.hp.model.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(self.hp.log.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(self.hp.log.exp_dir, 'chkpt')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.hp.model.save_interval is None:
            self.hp.model.save_interval = self.hp.model.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])
        if self.hp.model.keep_optimizer:
            self.g_optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        while self.global_step < self.hp.model.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                d_loss_dict = {}
                gt_img, occimg, id_img, mask, same = batch
                if self.is_training_discriminator():
                    d_loss_dict = self.train_discriminator(gt_img, occimg, id_img, mask)
                output, a_out, z_id, output_z_id, occimg, id_img = self.forward(occimg, id_img, mask)
                gt_img, same, mask = gt_img.to(self.device).float(), same.to(self.device).float(), mask.to(
                    self.device).float()
                loss, g_loss_dict = self.id_loss(output, gt_img, z_id, output_z_id, mask, same)

                output_multi_scale_val = self.net.D(output)
                #                 print(output_multi_scale_val.size())
                gt_multi_scale_val = self.net.D(gt_img)
                loss_GAN = self.gan_loss(output_multi_scale_val, True, for_discriminator=False)

                g_loss_dict['dg_loss'] = float(loss_GAN)

                num_D = len(output_multi_scale_val)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(output_multi_scale_val[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        # if j >= num_intermediate_outputs - 2:
                        if j >= 0:
                            unweighted_loss = self.criterionFeat(
                                output_multi_scale_val[i][j], gt_multi_scale_val[i][j].detach())
                            #                         print(i,j,unweighted_loss)
                            GAN_Feat_loss += unweighted_loss * self.hp.model.lambda_feat / num_D
                g_loss_dict['GAN_Feat'] = float(GAN_Feat_loss)

                loss = loss + loss_GAN + GAN_Feat_loss

                loss_dict = {**d_loss_dict, **g_loss_dict}
                self.g_optimizer.zero_grad()
                loss.backward()
                self.g_optimizer.step()

                # Logging related
                if self.global_step % self.hp.model.log_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    inpainted = output.detach() * mask + occimg * (1 - mask)
                    common.parse_and_log_images_v2(['masked', 'att', 'id_prior', 'output', 'inpainted', 'gt'],
                                                   [occimg, a_out, id_img, output.detach(), inpainted.detach(), gt_img],
                                                   subdir=self.logger.logdir, title='images/train/faces',
                                                   step=self.global_step, display_count=2)
                #                     self.parse_and_log_images(None,occimg,a_out,id_img, inpainted.detach(), title='images/train/faces')
                if self.global_step % self.hp.model.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.hp.model.sample_interval == 0 or self.global_step == self.hp.model.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.hp.model.save_interval == 0 or self.global_step == self.hp.model.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.hp.model.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            gt_img, occimg, id_img, mask, same = batch
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(gt_img, occimg, id_img, mask)
            with torch.no_grad():
                output, a_out, z_id, output_z_id, occimg, id_img = self.forward(occimg, id_img, mask)
                gt_img, same, mask = gt_img.to(self.device).float(), same.to(self.device).float(), mask.to(
                    self.device).float()
                inpainted = output * mask + occimg * (1 - mask)
                loss, cur_encoder_loss_dict = self.id_loss(output, gt_img, z_id, output_z_id, mask, same)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            common.parse_and_log_images_v2(['masked', 'att', 'id_prior', 'output', 'inpainted', 'gt'],
                                           [occimg, a_out, id_img, output, inpainted, gt_img],
                                           subdir=self.logger.logdir, title='images/test/faces',
                                           subscript='{:04d}'.format(batch_idx), step=self.global_step, display_count=2)
            #             self.parse_and_log_images(None, occimg,a_out, id_img, output,
            #                                       title='images/test/faces',
            #                                       subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        self.requires_grad(self.net.A, False)
        self.requires_grad(self.net.Z, False)
        lr_g = self.hp.model.learning_rate_E_G
        lr_d = self.hp.model.learning_rate_D
        b1 = self.hp.model.beta1
        b2 = self.hp.model.beta2
        opt_d = torch.optim.Adam(self.net.D.parameters(), lr=lr_d, betas=(b1, b2))
        if self.hp.model.optim_name == 'adam':
            opt_g = torch.optim.Adam(self.net.G.parameters(), lr=lr_g, betas=(b1, b2))
        else:
            opt_g = Ranger(self.net.G.parameters(), lr=lr_g)
        return opt_g, opt_d

    def configure_datasets(self):
        train_dataset = CelebaDataset(root_dir=self.hp.data.trainset_dir,
                                      pickle_path=self.hp.data.train_pickle_list,
                                      occ_dir=self.hp.data.occ_dir,
                                      occ_list=self.hp.data.occ_list,
                                      resolution=self.hp.model.resolution,
                                      picklimit=False,ratio=self.hp.data.sameratio)

        test_dataset = CelebaDataset(root_dir=self.hp.data.valset_dir,
                                     pickle_path=self.hp.data.val_pickle_list,
                                     occ_dir=self.hp.data.occ_dir,
                                     occ_list=self.hp.data.occ_list,
                                     resolution=self.hp.model.resolution,lenth=50,ratio=self.hp.data.sameratio)

        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def forward(self, occ_img, id_img, mask=None):
        occ_img, id_img = occ_img.to(self.device).float(), id_img.to(self.device).float()
        if mask is not None:
            mask = mask.to(self.device).float()
        output, imgout, z_id, output_z_id = self.net.forward(occ_img, id_img, mask)
        return output, imgout, z_id, output_z_id, occ_img, id_img

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, xa, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i]),
                'att_face': common.log_input_image(xa[i]),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.net.A.opts),
            'latent_avg': self.net.A.latent_avg

            #             'hp': self.hp
        }
        if self.hp.model.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.g_optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
        return save_dict

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return True

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, gt_img, occ_img, id_img, mask):
        loss_dict = {}
        gt_img = gt_img.to(self.device).float()
        self.requires_grad(self.net.D, True)
        with torch.no_grad():
            output, _, _, _, _, _ = self.forward(occ_img, id_img, mask)
        multi_scale_val = self.net.D(gt_img.detach())
        output_multi_scale_val = self.net.D(output.detach())
        loss_D_fake = self.gan_loss(multi_scale_val, True)
        loss_D_real = self.gan_loss(output_multi_scale_val, False)
        loss = loss_D_fake + loss_D_real
        loss_dict['discriminator_loss'] = float(loss)

        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
        # Reset to previous state
        self.requires_grad(self.net.D, False)

        return loss_dict

    def validate_discriminator(self, gt_img, occ_img, id_img, mask):
        with torch.no_grad():
            loss_dict = {}
            gt_img = gt_img.to(self.device).float()
            output, _, _, _, _, _ = self.forward(occ_img, id_img, mask)
            multi_scale_val = self.net.D(gt_img.detach())
            output_multi_scale_val = self.net.D(output.detach())
            loss_D_fake = self.gan_loss(multi_scale_val, True)
            loss_D_real = self.gan_loss(output_multi_scale_val, False)
            loss = loss_D_fake + loss_D_real
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

