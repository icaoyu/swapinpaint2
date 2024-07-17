"""
This file runs the main training/val loop
for structure and identity guided inpaintor256
"""
import argparse
import os
import json
import math
import sys
import pprint
import time

import cv2
from torchvision import transforms
from PIL import Image

import torch
from argparse import Namespace

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
# from training.coach_inpaintor import parse_and_log_images,print_metrics

import numpy as np
sys.path.append(".")
sys.path.append("..")
# from models.ID_inpaintor256_biaad import ID_Inpaintor


from utils import common, masktool
# from training.coach import Coach

def blending(content_img, output, mask):
    '''
    :param content_img: Image type with size(3,256,256)
    :param output: Image type with size(3,256,256)
    :param mask: (3,256,256), wiht 0 and 1
    :return: Image type with size(3,256,256)
    '''
    obj_img = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    bg_img = cv2.cvtColor(np.array(content_img), cv2.COLOR_RGB2BGR)
    mask = (1-mask)*255
    mask = mask.astype(np.uint8)
    monoMaskImage = cv2.split(mask)[0]
    br = cv2.boundingRect(monoMaskImage)  # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    out_img = cv2.seamlessClone(obj_img.astype(np.uint8), bg_img.astype(np.uint8), mask.astype(np.uint8), centerOfBR, cv2.NORMAL_CLONE)
    out_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

    return out_img





def test(net,occ_img,id_img,mask,save_dir):
    net.eval()
    # occimg = gt_img*mask

    # print_metrics(loss_dict, prefix='test')


def parse_and_log_images(titlelist,imglist, save_dir,display_count=1,flag=''):
    assert len(titlelist)==len(imglist),"length of titlelist is not equal to length of imglist"
    listlen = len(titlelist)
    im_data = []
    for i in range(display_count):
        cur_im_data = {}
        for j in range(listlen):
            cur_im_data[titlelist[j]] = common.log_input_image(imglist[j][i])
        im_data.append(cur_im_data)
    save_name = flag + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.jpg'
    log_images(im_data,titlelist,path=os.path.join(save_dir, save_name))

def log_images(im_data,titlelist,path):
    fig = vis_faces(im_data,titlelist)
    # path = os.path.join(save_dir, '{:04d}.jpg'.format(step))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def vis_faces(log_hooks,titlelist):
    display_count = len(log_hooks)
    titlelen = len(titlelist)
    fig = plt.figure(figsize=(16, 6 * display_count))
    gs = fig.add_gridspec(display_count,titlelen)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        for j in range(titlelen):
            fig.add_subplot(gs[i, j])
            plt.imshow(hooks_dict[titlelist[j]])
            plt.title(titlelist[j])
    plt.tight_layout()
    return fig

def gettransform(hp):
    transform_list = [
            transforms.Resize((256, 256)),
#             transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ]
    if hp.model.normflag:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)
    return transform


def main(ID_Inpaintor,args):
    hp = OmegaConf.load('models/fusor/train256_biaad.yaml')
    hp.log.resume_training_from_ckpt = args.chkpt
    hp.model.getIntermFeat = True
    device = hp.model.device
    # Initialize network
    model = ID_Inpaintor(hp).to(device)
    model.eval()
    trans = gettransform(hp)
    target_img = trans(Image.open(args.gt_img)).unsqueeze(0).to(device)
    source_img = trans(Image.open(args.id_img)).unsqueeze(0).to(device)
    g = [os.path.basename(args.gt_img)[:-2],os.path.basename(args.id_img)[:-2]]
    detector, predictor = masktool.initPredictor('utils/shape_predictor_68_face_landmarks.dat')

    if args.mask_type == 1:
        mask = masktool.face_mask(args.gt_img, detector, predictor)
    elif args.mask_type == 2:
        mask = masktool.get_eye_Mask(args.gt_img, detector, predictor)
    elif args.mask_type == 3:
        mask = masktool.half_face_mask(args.gt_img, detector, predictor,direct='left')
    elif args.mask_type == 4:
        mask = masktool.half_face_mask(args.gt_img, detector, predictor,direct='right')
    elif args.mask_type == 5:
        mask = masktool.load_mask(args.mask_path,256,threshold=128)
    elif args.mask_type == 6:
        mask = masktool.get_mouth_Mask(args.gt_img,detector,predictor)
    elif args.mask_type == 7:
        mask = masktool.nose_mask(args.gt_img, detector, predictor)
    elif args.mask_type == 8:
        mask = masktool.down_mask(args.gt_img, detector, predictor)
    else:
        mask = masktool.center_mask_size((1, 256, 256))
    if mask == None:
        print("mask is none")
        return
        # masked_img = target_img
    else:
        mask = mask.unsqueeze(0).to(device)
        masked_img = target_img * mask
        print(masked_img.size(),source_img.size())
    with torch.no_grad():
        output, imgout, _, _ = model.forward(masked_img,source_img,1-mask)

        inpainted = output*(1-mask)+masked_img*mask
        # Logging related
        # parse_and_log_images(['masked','att','id_prior','output','inpainted','gt'],[masked_img,imgout,source_img ,output,inpainted,target_img],save_dir=args.save_dir)

#         parse_and_log_images(masked_img, source_img, target_img,inpainted,save_dir=args.save_dir)
        output = (output + 1) / 2
        output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
        save_name = str(g[0])+'_'+str(g[1])+'_'+str(args.mask_type)+'_y.jpg'
        output.save(os.path.join(args.save_dir,save_name))
        
        source_img = (source_img + 1) / 2
        source_img = transforms.ToPILImage()(source_img.cpu().squeeze().clamp(0, 1))
        save_name = str(g[0])+'_'+str(g[1])+'_'+str(args.mask_type)+'_id.jpg'
        source_img.save(os.path.join(args.save_dir,save_name))
        
        masked_img = (masked_img + 1) / 2
        masked_img = transforms.ToPILImage()(masked_img.cpu().squeeze().clamp(0, 1))
        save_name = str(g[0])+'_'+str(g[1])+'_'+str(args.mask_type)+'_masked.jpg'
        masked_img.save(os.path.join(args.save_dir,save_name))

        inpainted = (inpainted + 1) / 2
        inpainted = transforms.ToPILImage()(inpainted.cpu().squeeze().clamp(0, 1))
        save_name = str(g[0])+'_'+str(g[1])+'_'+str(args.mask_type)+'.jpg'
        inpainted.save(os.path.join(args.save_dir,save_name))
        print("processed:",save_name)

def run_by_list(ID_Inpaintor,args,list,root):
    #list=[[occimg,piorimg,makstype],[36,36,0]^]
    hp = OmegaConf.load('models/fusor/train256_biaad.yaml')
    hp.log.resume_training_from_ckpt = args.chkpt
    hp.model.getIntermFeat = True
    hp.model.device = 'cuda:'+args.gpuid
    device = hp.model.device
    # Initialize network
    model = ID_Inpaintor(hp).to(device)
    model.eval()
    trans = gettransform(hp)
    detector, predictor = masktool.initPredictor('utils/shape_predictor_68_face_landmarks.dat')
    for g in list:
        print(g)
        target_img = trans(Image.open(root+str(g[0])+'.jpg')).unsqueeze(0).to(device)
        source_img = trans(Image.open(root+str(g[1])+'.jpg')).unsqueeze(0).to(device)

        mask_type = g[2]
        gt_img = root+str(g[0])+'.jpg'
        if mask_type == 1:
            mask = masktool.face_mask(gt_img, detector, predictor)
        elif mask_type == 2:
            mask = masktool.get_eye_Mask(gt_img, detector, predictor)
        elif mask_type == 3:
            mask = masktool.half_face_mask(gt_img, detector, predictor, direct='left')
        elif mask_type == 4:
            mask = masktool.half_face_mask(gt_img, detector, predictor, direct='right')
        elif mask_type == 5:
            mask = masktool.load_mask(args.mask_path, 256, threshold=128)
        elif mask_type == 6:
            mask = masktool.get_mouth_Mask(gt_img,detector,predictor)
        elif mask_type == 7:
            mask = masktool.nose_mask(gt_img, detector, predictor)
        elif mask_type == 8:
            mask = masktool.down_mask(gt_img, detector, predictor)
        else:
            mask = masktool.center_mask_size((1, 256, 256),hole=125)
        print('type',mask.size())
        if mask == None:
            print("mask is none")
            continue
        # masked_img = target_img
        else:
            mask = mask.unsqueeze(0).to(device)
            masked_img = target_img * mask
            # print(masked_img.size(), source_img.size())
            with torch.no_grad():
                output, imgout, _, _ = model.forward(masked_img, source_img, 1 - mask)
            
            if args.blend == 1:
                tempout = output
                output = (output + 1) / 2
                output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
                target_temp = (imgout + 1) / 2
                target_temp = transforms.ToPILImage()(target_temp.cpu().squeeze().clamp(0, 1))
                mask_temp = np.array((mask.cpu().squeeze(axis=0).clamp(0, 1))).transpose((1, 2, 0))
                inpainted = blending(target_temp, output, mask_temp)
            else:
                inpainted = output * (1 - mask) + masked_img * mask
                # parse_and_log_images(['masked', 'att', 'id_prior', 'output', 'inpainted', 'gt'],
                #                      [masked_img, imgout, source_img, output, inpainted, target_img], save_dir=args.save_dir,flag=str(g[0])+'_'+str(g[1]))
                
                inpainted = (inpainted + 1) / 2
                inpainted = transforms.ToPILImage()(inpainted.cpu().squeeze().clamp(0, 1))
            save_name = str(g[0])+'_'+str(g[1])+'_'+str(mask_type)+'.jpg'
            inpainted.save(os.path.join(args.save_dir,save_name))
            print("processed:",save_name)

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--mode", type=str, default="biaad_ef_wbg",
                        help="biaad_ef_wbg , biaad_ef_wbg")
    parser.add_argument("--gpuid", type=str, default="1",
                        help="The directory of the images to be inverted")
    parser.add_argument("--gt_img", type=str, default="data/test_jpg/79.jpg",
                        help="The directory of the images to be inverted")
    parser.add_argument("--id_img", type=str, default="data/test_jpg/57.jpg", help="batch size for the generator")
    parser.add_argument("--mask_type", type=int, default=1,help="0:center,1:face,2:eye,3:leftface,4:rihtface,5:path,6:mouth")
    parser.add_argument("--mask_path", type=str, default='data/mask.jpg',help="")
    parser.add_argument("--blend", type=int, default=0,help="0:false,1:true")

    parser.add_argument("--save_dir", type=str, default="workdir/biaad_ef_wbg50w_sameratio3/test_fig3/",
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    # parser.add_argument("--chkpt", default="workdir/biaad_ef_wbg50w_sameratio3/chkpt/iteration_100000.pt", help="path to generator checkpoint")
    parser.add_argument("--chkpt", default="chkpt/best_model.pt", help="path to generator checkpoint")

    args = parser.parse_args()


    from models.ID_inpaintor256_biaad_ef_wbg import ID_Inpaintor
    # l = [[36,23,0],[10,36,4],[13,10,2],[65,13,1],[5,65,1],[23,5,5]]
    # l = [[36,36,0],[13,13,2],[65,65,1],[23,23,5]]
    # l=[[23,85,7],[23,29,7],[23,17,7],[23,31,7],[23,57,7],[23,95,7]]
    # run_by_list(ID_Inpaintor,args,l,'data/test_jpg/')
    main(ID_Inpaintor,args)

#     python scripts/inference-ID256-biaad.py --gt_img data/celeba/test4/183613.jpg --id_img data/celeba/test4/183613.jpg --mask_type 1 --save_dir workdir/ID256_sp2_naad1/test/ --chkpt workdir/ID256_naad_wofmloss/chkpt/best_model.pt
# python scripts/inference-ID256-biaad.py --gt_img data/test_jpg/4.jpg --id_img data/test_jpg/23.jpg --mask_type 1 --save_dir workdir/ID256_biaad/test/ --chkpt workdir/ID256_biaad/chkpt/iteration_120000.pt

#python scripts/inference-ID256-biaad-wbg.py --gt_img data/test_jpg/36.jpg --id_img data/test_jpg/36.jpg --mask_type 0