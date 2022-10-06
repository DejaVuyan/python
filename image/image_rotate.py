import os
import cv2

output_dir = r'/data/yuzun/dataset/ADNI1_296sub/train_flip'
# 读取图片
image_dir = r'/data/yuzun/dataset/ADNI1_296sub/train'

for img in os.listdir(image_dir):
    image_path = os.path.join(image_dir,img)
    image = cv2.imread(image_path)

    # 逆时针90°
    rotate_90_cv = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 镜像翻转
    xImg = cv2.flip(rotate_90_cv,1,dst=None) #水平镜像
    # xImg1 = cv2.flip(rotate_90_cv,0,dst=None) #垂直镜像
    # xImg2 = cv2.flip(img,-1,dst=None) #对角镜像

    output_path = os.path.join(output_dir,img)
    cv2.imwrite(output_path,xImg)
    print(image_path)