import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.use('TkAgg')
mpl.rcParams.update({'font.size': 26})

mod_j = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_Joint_80perc.txt', skiprows=1)
mod_ert = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/ERT/XYZr_GCR.txt', skiprows=1)
mod_ohm = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_CCR_80perc.txt', skiprows=1)

colors = np.array([
        [0, 0, 255, 255],
        [0, 60, 254, 255],
        [1, 119, 253, 255],
        [25, 181, 229, 255],
        [89, 186, 241, 255],
        [72, 228, 246, 255],
        [102, 233, 250, 255],
        [6, 247, 183, 255],
        [245, 251, 207, 255],
        [255, 255, 160, 255],
        [255, 203, 118, 255],
        [255, 180, 55, 255],
        [250, 148, 47, 255],
        [253, 83, 56, 255],
        [246, 63, 0, 255],
        [246, 0, 0, 255],
        [208, 0, 0, 255],
        [170, 28, 0, 255]
    ]) / 255.0
cscale_orig = np.r_[0, 2.7, 5.3, 8.0, 9.3, 12.0, 18.6, 22.6, 29.3, 38.6, 51.9, 70.5, 95.7, 119.7, 146.3, 172.9, 212.8, 250]
colorscale1 = np.r_[0, 2.0, 4.0, 6.5, 9.6, 12.9, 16.5, 20.5, 25.5, 31.0, 40.0, 52.0, 68.0,  88.0, 114.0, 148.0, 192.0, 250]
colorscale2 = np.linspace(0, 250, 256)
cmap_finterp_r = interp1d(colorscale1, colors[:, 0])
cmap_finterp_g = interp1d(colorscale1, colors[:, 1])
cmap_finterp_b = interp1d(colorscale1, colors[:, 2])
colormap = np.stack((cmap_finterp_r(colorscale2),
                     cmap_finterp_g(colorscale2),
                     cmap_finterp_b(colorscale2),
                     np.ones(len(colorscale2))), axis=1)
# scala colori continua
colmap2 = mpl.colors.ListedColormap(colormap)
# scala colori discreta
interpcol1 = interp1d(colorscale2, colormap[:, 0])
interpcol2 = interp1d(colorscale2, colormap[:, 1])
interpcol3 = interp1d(colorscale2, colormap[:, 2])
interpcol = np.concatenate([np.reshape(interpcol1(colorscale1), [-1, 1]),
                            np.reshape(interpcol2(colorscale1), [-1, 1]),
                            np.reshape(interpcol3(colorscale1), [-1, 1]),
                            np.ones([len(colorscale1), 1])], axis=1)
colmap1 = mpl.colors.ListedColormap(interpcol)

bin_j = np.histogram(mod_j[np.logical_not(np.isnan(mod_j[:, 3])), 3], bins=100)
bin_ert = np.histogram(mod_ert[np.logical_not(np.isnan(mod_ert[:, 3])), 3], bins=100)
bin_ohm = np.histogram(mod_ohm[np.logical_not(np.isnan(mod_ohm[:, 3])), 3], bins=100)

val_j = bin_j[0]
val_j = np.append(np.append(0, val_j * 100 / np.sum(val_j)), 0)
val_ert = bin_ert[0]
val_ert[0] = 200
val_ert = np.append(np.append(0, val_ert * 100 / np.sum(val_ert)), 0)
val_ohm = bin_ohm[0]
val_ohm = np.append(np.append(0, val_ohm * 100 / np.sum(val_ohm)), 0)

bin_j = np.append(np.append(0, bin_j[1][:-1] + np.mean(np.diff(bin_j[1]))), 250)
bin_ert = np.append(np.append(0, bin_ert[1][:-1] + np.mean(np.diff(bin_ert[1]))), 250)
bin_ohm = np.append(np.append(0, bin_ohm[1][:-1] + np.mean(np.diff(bin_ohm[1]))), 250)

fig, ax = plt.subplots(figsize=(22, 10))
frame = plt.gca()
ax.plot(bin_ohm, val_ohm, label='CCR', linewidth=5)
ax.plot(bin_ert, val_ert, label='GCR', linewidth=5)
ax.plot(bin_j, val_j, label='Joint', linewidth=5)
sct1 = ax.scatter(colorscale1, np.ones(len(colorscale1)) * 120, c=colorscale1, alpha=1, label='_foo', cmap=colmap1)
sct2 = ax.scatter(colorscale2, np.ones(len(colorscale2)) * 130, c=colorscale2, alpha=1, label='_foo', cmap=colmap2)
ax.plot()
ax.legend()
ax.grid()
ax.set_ylabel('Distribution [%]')
#frame.axes.xaxis.set_ticklabels([])
plt.xticks(colorscale1, [])
ax.set_xlim([0, 250])
ax.set_ylim([0.1, 6])
divider = make_axes_locatable(frame)
#axBar1 = divider.append_axes("bottom", '5%', pad='7%')
axBar2 = divider.append_axes("bottom", '5%', pad='7%')
#cbar1 = plt.colorbar(sct1, cax=axBar1, orientation='horizontal')
cbar2 = plt.colorbar(sct2, cax=axBar2, orientation='horizontal', ticks=colorscale1, format='%.0f')
cbar2.ax.tick_params(labelsize=12)
cbar2.set_label('Resistivity [Ω⋅m]')

plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Figure5.png')
frmt = '{0:.4f} {1:.4f} {2:.4f} {3:.4f}\n'
with open('colorscale.lut', 'w')as fid:
    fid.write('LOOKUP_TABLE UnnamedTable 256\n')
    for i in np.arange(len(colormap)):
        fid.write(frmt.format(colormap[i, 0], colormap[i, 1], colormap[i, 2], colormap[i, 3]))

