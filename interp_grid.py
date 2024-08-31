from scipy.interpolate import LinearNDInterpolator as interp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import lsq_linear

matplotlib.use('TkAgg')


xin = -9
xfin = 9
yin = -8
yfin = 8
zin = -5
zfin = 0
step_interp = .2
model = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/Joint_model_80perc.txt',skiprows=1, delimiter='\t')
interpolator = interp(model[:, :3], model[:, 3])
x = np.round(np.linspace(xin, xfin, num=int((xfin-xin+step_interp)/step_interp)),
             int(np.ceil(np.abs(np.log10(step_interp)))))
y = np.round(np.linspace(yin, yfin, num=int((yfin-yin+step_interp)/step_interp)),
             int(np.ceil(np.abs(np.log10(step_interp)))))
z = np.round(np.linspace(zin, zfin, num=int((zfin-zin+step_interp)/step_interp)),
             int(np.ceil(np.abs(np.log10(step_interp)))))
X, Y, Z = np.meshgrid(x, y, z)
X = np.reshape(X, [-1, 1])
Y = np.reshape(Y, [-1, 1])
Z = np.reshape(Z, [-1, 1])

qcoord = np.concatenate([X, Y, Z], axis=1)
val = interpolator(qcoord)
model_interp = np.hstack([qcoord, np.reshape(val, [-1, 1])])

url = 'C:/Users/lopan/PycharmProjects/STELLA/Res/XYZr_Joint_80perc.txt'
frmt = '{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.6f}\n'
with open(url, 'w') as fid:
    fid.write('x\ty\tz\trho\n')
    for i in np.arange(len(val)):
        fid.write(frmt.format(model_interp[i,0], model_interp[i,1], model_interp[i,2], model_interp[i,3]))


model_interp = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/ERT/XYZr_GCR.txt', skiprows=1, delimiter='\t')
colormap = np.loadtxt(r"G:\.shortcut-targets-by-id\1U5Mbs1kWT545VLL1mMUjiV8eOfupEFet\Paper Geoelettrica\Figure\Figure5\colorscale.lut", skiprows=1, delimiter=' ')
cmap_new = matplotlib.colors.ListedColormap(colormap)
ix = np.where(model_interp[:, 0] == 0.9)
plt.imshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])), len(np.unique(model_interp[ix, 2]))]).transpose(), axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=188)
