"""
将x,y的散点连成光滑的曲线，然后显示出来
问题：make_interp_spline()函数不理解。
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
y1 = np.array([0.7894, 0.7940, 0.8450, 0.8490, 0.8569, 0.8569, 0.8632, 0.8649, 0.8963, 0.9013,
               0.9136, 0.9189, 0.9263, 0.9293, 0.9312, 0.9365, 0.9432, 0.9433, 0.9455, 0.9455])
y2 = np.array([0.2976, 0.3012, 0.3123, 0.3452, 0.3521, 0.3600, 0.3612, 0.3712, 0.3765, 0.3949,
               0.4111, 0.4167, 0.4234, 0.4315, 0.4319, 0.4387, 0.4355, 0.4369, 0.4370, 0.4373])

#平滑曲线
x_smooth = np.linspace(x.min(), x.max(), 300)
y1_smooth = make_interp_spline(x, y1)(x_smooth)     #B-spline是一种曲线拟合的算法，不过这两个()()是啥意思？
y2_smooth = make_interp_spline(x, y2)(x_smooth)     #两个括号说明第一个函数返回的还是一个函数，所以要调用多次？？还是不太懂


# 画图
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x_smooth, y1_smooth, label="图片大小 480*480")  # Plot some data on the axes.
ax.plot(x_smooth, y2_smooth, label="图片大小 60 *60 ")
x_major_locator=MultipleLocator(1)    # 设置x轴之间的间距为1
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlabel('周期')
ax.set_ylabel('准确率')  # Add a y-label to the axes.
#ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.show()   # 显示画出的图像，ax.plot,或者是ply.plot 就是画图用的，如果调用两个ax.plot两张图就会画在一个界面里，如果
#一个plot一个plt.show就会显示两个图片，叉掉一个另一个才会出来。
