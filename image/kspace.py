from util import save_image,mkdirs,fft_2D,read_image
import argparse
import os

"""
将文件夹中的图片进行k-space下采样后输出
"""

if __name__ == '__main__':
    # 初始化对象
    parser = argparse.ArgumentParser(description='Process some integers.')
    # 添加参数，包括type,default,help等等，注意有个必须的
    # 定义必须要的参数，输入，输出与下采样比例
    parser.add_argument('--input', type=str, required=True, help='the input dir path of images')
    parser.add_argument('--output', type=str, required=True, help='the output dir path of images')
    parser.add_argument('--ratio', type=float, required=True,help='The image is cropped in the middle of the k-space space to preserve the proportion')

    args = parser.parse_args()  # 初始化这个,-h才有显示

    input_dir = args.input  # /data/yuzun/stytr_008_SE10_out_256
    output_dir =args.output # /data/yuzun/stytr_008_SE10_out_256_kspace
    ratio = args.ratio # 0.25

    # 统计所处理的图片的数量
    count = 0
    # 批量读取input所指定的路径中的图片
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir,image_name)
        img = read_image(image_path,1)  # numpy 灰度图则是(H,W)
        img = fft_2D(img,ratio)   # 返回的是Tensor，归一化后的
        mkdirs(output_dir)
        output_path = os.path.join(output_dir,image_name)
        save_image(img,output_path)
        count=count+1
    print('convert {} images'.format(count))




