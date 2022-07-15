"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from torchvision.transforms import InterpolationMode,ToTensor,Normalize
import math

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data  # (batch,c,h,w)
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array  (c,h,w)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))  # 将灰度变为RGB，在维度0复制3遍，即(3,208,208),这里的范围是(-1,0)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

        image_numpy[image_numpy<0]=0   # 为了抑制雪花点，但是输出不用调吗？

        #print(image_numpy)
        # 转置，(208,208,3)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)   # 输出的是一个ndarray，将float转换成uint8的时候会损失掉小数


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), InterpolationMode.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), InterpolationMode.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def fft_2D(input_Tensor,mask_ratio = 0.25):
    """使用2D傅里叶变换，将经过base_dataset，transformer,标准化后的(C,H,W)的Tensor
    转换到频域，然后使用低通滤波器滤波，面积为原图*mask_ratio

    """
    #print(input_Tensor.shape)
    img = input_Tensor.squeeze(dim=0)  # (H,W)

    # 创建一个低通滤波器
    mask = np.zeros(img.shape)
    crow = int((img.shape[0]) / 2)  # 中心位置
    ccol = int((img.shape[1]) / 2)  # 中心位置
    mask_l = int(math.sqrt(img.shape[0] * img.shape[1] * mask_ratio) / 2)
    mask[crow - mask_l:crow + mask_l, ccol - mask_l:ccol + mask_l] = 1

    f = np.fft.fft2(img)  # 2D傅里叶变换
    fshift = np.fft.fftshift(f)  # 低频中心化
    # 滤波
    fshift = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)  # 逆中心化
    iimg = np.fft.ifft2(ishift)  # 傅里叶逆变换
    iimg = np.abs(iimg)  # 去掉虚部 此时数据为float64的ndarray
    iimg = np.expand_dims(iimg, axis=-1)  # 回到了原来的函数
    #print(iimg.shape)
    # 变成Tensor
    to_tensor = ToTensor()
    iimg_Tensor = to_tensor(iimg).float()  # 从double float64转换为float 32
    # 归一化
    nor = Normalize((0.5,), (0.5,))
    iimg_Tensor = nor(iimg_Tensor)

    return iimg_Tensor

def print_para(network):
    # 打印网络的层名和参
    i = 0
    for name, param in enumerate(network.named_parameters()):
        print('i is {}'.format(i))
        print(name)
        print(param)