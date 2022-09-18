import os
from xml.etree.ElementTree import parse

def convert_annotation(input_path):
    """
    将人工智能竞赛变电巡检项目的xml格式label文件转换为 yolov5要求的label格式
    :param input_path: label所存放的地址
    :return: 
    """
    index_dic = {'bj_bpmh':0,
                 'bj_bpps':1,
                 'bj_wkps':2,
                 'jyz_pl':3,
                 'sly_dmyw':4,
                 'hxq_gjtps':5,
                 'xmbhyc':6,
                 'yw_gkxfw':7,
                 'yw_nc':8,
                 'gbps':9,
                 'wcaqm':10,
                 'wcgz':11,
                 'xy':12,
                 'bjdsyc':13,
                 'ywzt_yfyc':14,
                 'hxq_gjbs':15,
                 'kgg_ybh':16
                 }
    
    #定义xml路径
    #input_path = r'D:\MyWrokspace\code\somethings\DataOperation\data_test'
    
    for item in (os.listdir(input_path)):
        # 遍历存放label的路径
        xml_path = os.path.join(input_path,item)
    
        xml = parse(xml_path)
        root = xml.getroot()
        # tag对应标签名
        # attrib对应标签的属性，是一个字典
    
        # findall查找root下面的所有子标签，二级子标签好像就找不到了，返回一个list,list中的每一个元素对应一个标签对象
        for size in root.findall("size"):
            width = int(size.find('width').text)
            height = int(size.find('height').text)
    
        # 找到标签中的每一个object
        for ob in root.findall("object"):
            name = ob.find('name').text
            bndbox = ob.find('bndbox')
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
    
            # 标签的类别
            index = index_dic[name]
            x_center = (xmax+xmin)/2/width
            w = (xmax - xmin)/width
            y_center = (ymax+ymin)/2/height
            h = (ymax-ymin)/height
            label_line = str(index) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n'
    
            # 写入txt文件
            f = open(xml_path.split('.')[0]+'.txt','a')  # 打开txt文件,如果没有就新建文件
            f.write(label_line)
            f.close()
            print(xml_path)
    print('label 转换完毕！')


def data_split(sourceRoot=r'/home/yyz/rope_2',
               targetDir=r"/home/yyz/n"):
    r"""
    将数据集按照比例随机分成训练集和测试集,注意是移动而不是复制，原有的数据集将消失
    请提前备份
    数据集组织格式: root\images  root\labels
    划分后组织格式  targetDir\train\images labels  targetDir\test\images labels

    要求的库:
    import os
    import random, shutil
    """
    import os
    import random, shutil
    #检查输出文件夹是否存在，如果不存在创建文件夹
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
        os.mkdir(os.path.join(targetDir,'test'))
        os.mkdir(os.path.join(targetDir, 'test','images'))
        os.mkdir(os.path.join(targetDir, 'test', 'labels'))
        os.mkdir(os.path.join(targetDir, 'train'))  # 后面将剩下的原文件文件夹都移动过去，所以不需要为train创建images
        print('targetDir is created!')
    else:
        print('targetDir already exits!')

    sourceImageDir = os.path.join(sourceRoot,'images')
    sourcelabelDir = os.path.join(sourceRoot,'labels')



    imageNumbers = len(os.listdir(sourceImageDir))
    rate = 0.1  # 测试集占总数据集的比例
    testNumbers = int(imageNumbers * rate)
    # trainNumbers = imageNumbers - testNumbers

    for file in os.listdir(sourceImageDir):
        # 将png格式的图片全部转换成jpg
        a = file.split('.')
        if a[1] == 'png':
            new_path = shutil.move(os.path.join(sourceImageDir,file),os.path.join(sourceImageDir,a[0]+'.jpg'))

    sample = random.sample(os.listdir(sourceImageDir), testNumbers)  # 随机选取testNumbers数量的样本图片
    for item in sample:
        imagePath = os.path.join(sourceImageDir,item) # 图片路径
        labelPath = os.path.join(sourcelabelDir,item.split('.')[0] + ".txt") # 标签路径，要求与图片路径同名，只有后缀名不同
        # 如果label文件不存在，说明图片中没有对象，就创建一个空文件
        if not os.path.exists(labelPath):
            new_label_file = open(labelPath,'w')
            new_label_file.close()  # 关闭指针
        shutil.move(imagePath,os.path.join(targetDir,'test','images',item))  # 移动到测试集文件夹中
        shutil.move(labelPath, os.path.join(targetDir, 'test', 'labels',item.strip("jpg") + "txt"))
    shutil.move(sourceImageDir,os.path.join(targetDir,'train'))  # 将原文件夹中剩下的图片-训练集图片，连带着文件夹移动
    shutil.move(sourcelabelDir,os.path.join(targetDir,'train'))
    print('Data partition completed!')

convert_annotation(r'F:\dataset\Yundian\label')

data_split(sourceRoot=r'F:\dataset\Yundian\BD44319_yolo',targetDir=r'F:\dataset\Yundian\BD44319_yolov5')














