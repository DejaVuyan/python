 # 切1.5T ADNI的图片
import nibabel as nib
import os
import imageio

# 检查是否有伪影
rootPath = r'/data/yuzun/SE_0919/nii/reged_results'
output_img_path = r'/data/yuzun/SE_0919/SE_0921_reged_medium_imgs'
file_name = r'SE_0919_medium_reg.nii.gz'

MRI_path = os.path.join(rootPath,file_name)
T1_img = nib.load(MRI_path)  # 读取nii
# 图片保存时需要的数值是[0,255]uint8的值，但是getfdate是一个flat64的数值，所以转换数据类型
T1_array = T1_img.get_fdata()
# print('debug')


for i in range(0,312,1):   # 去掉头尾，中间选取两张图片
    #print('T1 IS     ' + subject + '_' + time + '_{}'.format(i))
    T1_sliced_img = T1_array[:, :, i]  # 将z方向的第三张图弄出来，(448,29,448,1)->(448,29,1)
    img_path = os.path.join(output_img_path,'SE_0919' + '_' + '_{}'.format(i)+'.png') # 定义文件路径
    print(img_path)
    imageio.imwrite(img_path ,T1_sliced_img)  # 保存文件

print('done!')



