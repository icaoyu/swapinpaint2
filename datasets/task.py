import scipy.stats as st
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random

###################################################################
# random mask generation
###################################################################
def occlist_reader(fileList):
    occList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            occPath = line.strip()
            occList.append(occPath)
    return occList

def get_occ_boxs_v2():
    ## 112x96
    # left_eye = (10, 30, 50, 70)
    # right_eye = (50, 30, 90, 70)
    # twoeyes = (10, 30, 90, 70)
    # nose = (34, 48, 62, 86)
    # mouth = (25, 82, 75, 107)
    # down_face = (8, 60, 92, 110)
    # up_face = (8, 8, 92, 72)
    # left_face = (8, 8, 50, 100)
    # right_face = (50, 8, 92, 100)
    # center_block = (24, 32, 72, 80)

    # 112x112
    left_eye = (18, 30, 58, 70)
    right_eye = (58, 30, 98, 70)
    twoeyes = (18, 30, 98, 70)
    nose = (42, 48, 70, 86)
    mouth = (33, 82, 83, 107)
    down_face = (16, 60, 100, 110)
    up_face = (16, 8, 100, 72)
    left_face = (16, 8, 58, 100)
    right_face = (58, 8, 100, 100)
    center_block = (32, 32, 80, 80)
    RMFD = (0, 60, 112, 112)


    boxes = [left_eye, right_eye, twoeyes, nose, mouth, down_face, up_face, left_face, right_face, center_block, RMFD]
    names = ['left_eye', 'right_eye', 'twoeyes', 'nose', 'mouth', 'down_face', 'up_face', 'left_face', 'right_face', 'center_block', 'RMFD']
    return boxes, names

def generate_mask(img,box):
    #img:(3,112,112)
    # print(img.size())
    # print(box)
    mask = torch.ones_like(img)
    start_x, start_y, end_x, end_y = box
    mask[:, start_y:end_y, start_x:end_x] = 0
    return mask

def get_box(mask_type):
    # based on size(112,112)
    if mask_type == 'random_block':
        s = (1,112,112)
        # s = (1, 112, 96)
        holesize = (64,64)
        N_mask = random.randint(1, 5)
        limy = s[1] - s[1] / (N_mask + 1)
        limx = s[2] - s[2] / (N_mask + 1)
        # for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        if y > s[1] - holesize[0]:
            y = s[1] - holesize[0]
        if x > s[2] - holesize[1]:
            x = s[2] - holesize[1]
        range_x = x + holesize[1]
        range_y = y + holesize[0]
        box = int(x), int(y), int(range_x), int(range_y)
    else:
        boxs, names = get_occ_boxs_v2()
        if mask_type == 'random_part':
            box = random.choice(boxs)
        else:
            box = boxs[names.index(mask_type)]
    return box

def PIL_Reader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def generate_occ(img,occList,Occluders,trans, box):
    #img 为Image类型
    # print(occList)
    occimg = img.copy()
    occ_path = random.choice(occList)
    occ = PIL_Reader(os.path.join(Occluders, occ_path))
    # print(occ)
    W, H = img.size
    occ_w, occ_h = box[2] - box[0], box[3] - box[1]
    new_w, new_h = min(W - 1, occ_w), min(H - 1, occ_h)
    occ = occ.resize((new_w, new_h))
    occimg.paste(occ, box)
    img = trans(img)
    occimg = trans(occimg)

    mask = torch.ones_like(img)

    start_x, start_y, end_x, end_y = box
    mask[:,start_y:end_y, start_x:end_x] = 0
    return img,occimg,1-mask

# def occluded_image(img, occ,occ_path):
#     W, H = img.size
#     occ_w, occ_h = occ.size
#     occ_w = (occ_w*W)//112
#     occ_h = (occ_h*H)//112

#     new_w, new_h = min(W - 1, occ_w), min(H - 1, occ_h)
#     occ = occ.resize((new_w, new_h))

#     if "mouth_mask" in occ_path or "scarf" in occ_path:
#         # put on the below half region
#         center_x = random.choice(range(0, W))
#         center_y = random.choice(range(H*2//3, H))

#     elif "eye_mask" in occ_path or "sunglasses" in occ_path:
#         center_x = random.choice(range(0, W))
#         center_y = random.choice(range(0, H//3))
#     else:
#         center_x = random.choice(range(0, W))
#         center_y = random.choice(range(0, H))
#     start_x = center_x - new_w // 2
#     start_y = center_y - new_h // 2

#     end_x = center_x + (new_w + 1) // 2
#     end_y = center_y + (new_h + 1) // 2
#     box = (start_x, start_y, end_x, end_y)
#     img.paste(occ, box)

#     return img

def generate_iregular_occ(img,occList,Occluders,trans,mlist):
    #img 为Image类型
    occimg = img.copy()

    H, W = img.size
    img = trans(img)
    #img,torch,size(3,h,w)
    mask = load_mask(img,mlist)
    occ_path = random.choice(occList)
    occ = PIL_Reader(os.path.join(Occluders, occ_path))
    occ = occ.resize((H-1,W-1))
    box = (0,0,W-1,H-1)
    occimg.paste(occ, box)
    occ = trans(occimg)

    occimg = img*mask+occ*(1-mask)
    return img,occimg

def generate_iregular_occ_mask(img,occList,Occluders,trans,mlist):
    #img 为Image类型
    occimg = img.copy()

    H, W = img.size
    img = trans(img)
    #img,torch,size(3,h,w)
    mask = load_mask(img,mlist)
    occ_path = random.choice(occList)
    occ = PIL_Reader(os.path.join(Occluders, occ_path))
    occ = occ.resize((H-1,W-1))
    box = (0,0,W-1,H-1)
    occimg.paste(occ, box)
    occ = trans(occimg)

    occimg = img*mask+occ*(1-mask)
    mask = mask[0:1,:,:]
    return img,occimg,(1-mask)

def load_mask(img,masktype):
    """Load different mask types for training and testing"""
    mask_type_index = random.randint(0, len(masktype) - 1)
    mask_type = masktype[mask_type_index]

    # center mask
    if mask_type == 0:
        return center_mask(img)

    # random regular mask
    if mask_type == 1:
        return random_regular_mask(img)

    # random irregular mask
    if mask_type == 2:
        return random_irregular_mask(img)

def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = torch.ones_like(img)
    s = img.size()
    N_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask


def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    # mask = mask[0:1,:,:]
    size = img.size()
    x = int(size[1] / 4)
    y = int(size[2] / 4)
    range_x = int(size[1] * 3 / 4)
    range_y = int(size[2] * 3 / 4)
    mask[:, x:range_x, y:range_y] = 0

    return mask

def rect_mask(img, mask_shapes, rand=True):
    #rect_mask(Image,(64,64),rand=True)
    mask = torch.ones_like(img)
    mask = mask[0:1,:,:]
    im_shapes = img.size()[1:]
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        of0 = (im_shapes[0]-mask_shapes[0])//2
        of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[:,of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 0
    return mask


def stroke_mask(img, parts=16, maxVertex=24, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = torch.zeros_like(img)
    mask = mask[0:1, :, :]
    h,w = img.size()[1:]
    for i in range(parts):
        stroke = np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w)
        mask = mask + torch.from_numpy(stroke.transpose(2,0,1))
    mask = mask.clamp(0, 1)
    return 1-mask

def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 64)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(5, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(5, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(5, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask

def random_irregular_mask_128(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 10
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(8, 32)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask

def gauss_kernel(size=21, sigma=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1,size, size))
    return out_filter


def make_guass_var(size, sigma):
    kernel = gauss_kernel(size, sigma)

    return torch.Tensor(kernel)

def priority_loss_mask(mask, hsize=64, sigma=1/40, iters=7):
    kernel = make_guass_var(hsize, sigma)
    kw = kernel.size()[2]
    init = mask
    mask_priority = torch.ones_like(mask)
    for i in range(iters):
        mask_priority = F.conv2d(init, kernel,stride=1, padding=round((kw-1)/2))  # 步长为1,外加1圈padding,即上下左右各补了1圈的0,
        mask_priority = mask_priority * (1-mask)
        init = mask_priority + mask
    return mask_priority

def getstrokemask(mask):
    mask_soft = priority_loss_mask(mask, hsize=15, iters=4) + (1-mask)
    return mask, mask_soft
###################################################################
# multi scale for image generation
###################################################################


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    img = F.interpolate(img, size=[128, 128], mode='bilinear', align_corners=True)
    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs






####################################################################
# for datasets1.py
####################################################################
def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new

def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加
        y1,y2,x1,x2为叠加位置坐标值
    """

    # 判断jpg图像是否已经为4通道
    if png_img.shape[2] == 3:
        png_img = add_alpha_channel(png_img )

    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]

    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]

    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0
    alpha_jpg = 1 - alpha_png
    ret, mask = cv2.threshold(alpha_png, 0, 255, cv2.THRESH_BINARY)
    maskbg = np.zeros((jpg_img.shape[0], jpg_img.shape[1]), dtype=jpg_img.dtype)
    maskbg[y1:y2, x1:x2] = mask
    # 直接叠加
    for c in range(0, 3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[yy1:yy2, xx1:xx2, c]))

    return jpg_img,maskbg


def occluded_image(pilimg,occ_path,size):
    '''
    img：pil image
    return :
    img,occ_path -> Image
    mask:h*w, numpy(0,1)
    '''
    # 读取图像
    # img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
    occ = cv2.imread(occ_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img,(size,size))
    H,W = img.shape[0:2]
    occ_h_old,occ_w_old= occ.shape[0:2]
    center_x = random.choice(range(W * 3 // 8, W * 5 // 8))

    if "mouth_mask" in occ_path or "scarf" in occ_path:
        # put on the below half region
        center_y = random.choice(range(H*3//4, H))
        occ_w = occ_w_old*128//96
        occ_h = occ_h_old*128//96

    elif 'eyeglasses' in occ_path or "eye_mask" in occ_path or "sunglasses" in occ_path:
        center_y = random.choice(range(H//4, H//2))
        occ_w = random.choice(range(90, 100))
        occ_h = occ_h_old*occ_w//occ_w_old

    elif 'cup' in occ_path:
        center_y = random.choice(range(H//4, H*3//4))

        if occ_w_old >= occ_h_old:
            occ_w = random.choice(range(50,70))
            occ_h = occ_h_old * occ_w // occ_w_old

        else:
            occ_h = random.choice(range(50, 70))
            occ_w = occ_w_old * occ_h // occ_h_old

    elif 'phone' in occ_path or 'hand' in occ_path:
        center_y = random.choice(range(0, H))
        if occ_w_old>occ_h_old:
            occ_w = random.choice(range(70, 80))
            occ_h = occ_h_old * occ_w // occ_w_old
        else:
            occ_w = random.choice(range(70, 80))
            occ_h = occ_h_old * occ_w // occ_w_old

    else:
        center_y = random.choice(range(0, H))
        occ_w = int(occ_w_old * 1.3)
        occ_h = int(occ_w_old * 1.3)
    occ = cv2.resize(occ,(occ_w,occ_h))
    x1 = center_x-occ_w//2
    y1 = center_y-occ_h//2
    x2 = x1 + occ_w
    y2 = y1 + occ_h
    occ_img = img.copy()
    occ_img,mask = merge_img(occ_img, occ, y1, y2, x1, x2)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    occ_img = Image.fromarray(cv2.cvtColor(occ_img, cv2.COLOR_BGR2RGB))
    return img,occ_img,mask//255




def occluded_image_v2(pilimg,occ_path,size):
    '''
    img：pil image
    return :
    img,occ_path -> Image
    mask:h*w, numpy(0,1)
    '''
    # 读取图像
    # img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
    occ = cv2.imread(occ_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img,(size,size))
    H,W = img.shape[0:2]
    occ_h_old,occ_w_old= occ.shape[0:2]
    center_x = random.choice(range(W * 3 // 8, W * 5 // 8))

    if "mouth_mask" in occ_path or "scarf" in occ_path:
        # put on the below half region
        center_y = random.choice(range(H*3//4, H))
        occ_w = occ_w_old*128//96
        occ_h = occ_h_old*128//96

    elif 'eyeglasses' in occ_path or "eye_mask" in occ_path or "sunglasses" in occ_path:
        center_y = random.choice(range(H//4, H//2))
        occ_w = random.choice(range(90, 100))
        occ_h = occ_h_old*occ_w//occ_w_old

    elif 'cup' in occ_path:
        center_y = random.choice(range(H//4, H*3//4))

        if occ_w_old >= occ_h_old:
            occ_w = random.choice(range(50,70))
            occ_h = occ_h_old * occ_w // occ_w_old

        else:
            occ_h = random.choice(range(50, 70))
            occ_w = occ_w_old * occ_h // occ_h_old

    elif 'phone' in occ_path or 'hand' in occ_path:
        center_y = random.choice(range(0, H))
        if occ_w_old>occ_h_old:
            occ_w = random.choice(range(70, 80))
            occ_h = occ_h_old * occ_w // occ_w_old
        else:
            occ_w = random.choice(range(70, 80))
            occ_h = occ_h_old * occ_w // occ_w_old

    else:
        center_y = random.choice(range(0, H))
        occ_w = int(occ_w_old * 1.3)
        occ_h = int(occ_w_old * 1.3)
    occ_w = int(occ_w*1.5)
    occ_h = int(occ_h*1.5)
    occ = cv2.resize(occ,(occ_w,occ_h))
    x1 = center_x-occ_w//2
    y1 = center_y-occ_h//2
    x2 = x1 + occ_w
    y2 = y1 + occ_h
    occ_img = img.copy()
    occ_img,mask = merge_img(occ_img, occ, y1, y2, x1, x2)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    occ_img = Image.fromarray(cv2.cvtColor(occ_img, cv2.COLOR_BGR2RGB))
    return img,occ_img,mask//255


