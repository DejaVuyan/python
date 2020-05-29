"""
userGiude.py中用到的make_interp_spline()的官方examples
from:https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
python 3.7
"""
import numpy as np

def cheb_nodes(N):
    """
画出一个叫做cheb_nodes的图形
    :param N:
    :return:
    """
    jj = 2.*np.arange(N) + 1
    x = np.cos(np.pi * jj / 2 / N)[::-1]
    return x

x = cheb_nodes(20)
y = np.sqrt(1 - x**2)

from scipy.interpolate import BSpline, make_interp_spline

b = make_interp_spline(x, y)     #平滑曲线
np.allclose(b(x), y)
l, r = [(2, 0.0)], [(2, 0.0)]
b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
np.allclose(b_n(x), y)
x0, x1 = x[0], x[-1]
np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
phi = np.linspace(0, 2.*np.pi, 40)
r = 0.3 + np.cos(phi)
x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates

from scipy.interpolate import make_interp_spline
spl = make_interp_spline(phi, np.c_[x, y])   #？？

phi_new = np.linspace(0, 2.*np.pi, 100)
x_new, y_new = spl(phi_new).T

import matplotlib.pyplot as plt
plt.plot(x, y, 'o')        #使用oo画出x,y
#plt.show()
plt.plot(x_new, y_new, '-')   #在同一张图上使用--画出平滑后的x,y
plt.show()                   #显示画出的图形,如果想要将这两个图形在两个画面中显示，就在plt.plot(x, y, 'o') 后面加上plt.show()就行
