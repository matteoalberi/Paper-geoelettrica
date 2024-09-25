import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
#mpl.use('TkAgg')
mpl.rcParams.update({'font.size': 26})

# Carica i dati
mod_j = np.loadtxt('XYZr_Joint_80perc.txt', skiprows=1)
mod_ert = np.loadtxt('XYZr_GCR.txt', skiprows=1)
mod_ohm = np.loadtxt('XYZr_CCR_80perc.txt', skiprows=1)

# Rimuovi le righe con NaN
mod_j = mod_j[~np.isnan(mod_j).any(axis=1)]
mod_ert = mod_ert[~np.isnan(mod_ert).any(axis=1)]
mod_ohm = mod_ohm[~np.isnan(mod_ohm).any(axis=1)]

# Definizione dei colori
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
colorscale1 = np.r_[0, 2.0, 4.0, 6.5, 9.6, 12.9, 16.5, 20.5, 25.5, 31.0, 40.0, 52.0, 68.0, 88.0, 114.0, 148.0, 192.0, 250]
colorscale2 = np.linspace(0, 250, 256)

# Interpolazione dei colori
cmap_finterp_r = interp1d(colorscale1, colors[:, 0])
cmap_finterp_g = interp1d(colorscale1, colors[:, 1])
cmap_finterp_b = interp1d(colorscale1, colors[:, 2])
colormap = np.stack((cmap_finterp_r(colorscale2),
                     cmap_finterp_g(colorscale2),
                     cmap_finterp_b(colorscale2),
                     np.ones(len(colorscale2))), axis=1)

# Scala colori continua
colmap2 = mpl.colors.ListedColormap(colormap)

# Scala colori discreta
interpcol1 = interp1d(colorscale2, colormap[:, 0])
interpcol2 = interp1d(colorscale2, colormap[:, 1])
interpcol3 = interp1d(colorscale2, colormap[:, 2])
interpcol = np.concatenate([np.reshape(interpcol1(colorscale1), [-1, 1]),
                            np.reshape(interpcol2(colorscale1), [-1, 1]),
                            np.reshape(interpcol3(colorscale1), [-1, 1]),
                            np.ones([len(colorscale1), 1])], axis=1)
colmap1 = mpl.colors.ListedColormap(interpcol)

# Creazione degli istogrammi
bin_j = np.histogram(mod_j[np.logical_not(np.isnan(mod_j[:, 3])), 3], bins=100)
bin_ert = np.histogram(mod_ert[np.logical_not(np.isnan(mod_ert[:, 3])), 3], bins=100)
bin_ohm = np.histogram(mod_ohm[np.logical_not(np.isnan(mod_ohm[:, 3])), 3], bins=100)

# Calcolo dell'area delle curve originali
area_j_orig = np.trapz(bin_j[0], bin_j[1][:-1])
area_ert_orig = np.trapz(bin_ert[0], bin_ert[1][:-1])
area_ohm_orig = np.trapz(bin_ohm[0], bin_ohm[1][:-1])

# Normalizzazione delle curve e conversione in percentuale
val_j = np.append(np.append(0, (bin_j[0] / area_j_orig) * 100), 0)
val_ert = np.append(np.append(0, (bin_ert[0] / area_ert_orig) * 100), 0)
val_ohm = np.append(np.append(0, (bin_ohm[0] / area_ohm_orig) * 100), 0)

# Ricalcolo dei bin come differenze medie tra i punti
bin_j = np.append(np.append(0, bin_j[1][:-1] + np.mean(np.diff(bin_j[1]))), 250)
bin_ert = np.append(np.append(0, bin_ert[1][:-1] + np.mean(np.diff(bin_ert[1]))), 250)
bin_ohm = np.append(np.append(0, bin_ohm[1][:-1] + np.mean(np.diff(bin_ohm[1]))), 250)

# Creazione del plot
fig, ax = plt.subplots(figsize=(22, 10))
frame = plt.gca()

# Linea CCR
ax.plot(bin_ohm, val_ohm, label='CCR', color='black', linestyle='dashed', linewidth=2.5)

# Linea GCR con pallini come tratteggio
ax.plot(bin_ert, val_ert, label='GCR', color='black', linestyle=(0, (1, 1)), linewidth=2.5)

# Linea Joint
ax.plot(bin_j, val_j, label='Joint', color='black', linestyle='solid', linewidth=2.5)  

# Scatter per le scale colori
sct1 = ax.scatter(colorscale1, np.ones(len(colorscale1)) * 120, c=colorscale1, alpha=1, label='_foo', cmap=colmap1)
sct2 = ax.scatter(colorscale2, np.ones(len(colorscale2)) * 130, c=colorscale2, alpha=1, label='_foo', cmap=colmap2)
ax.plot()

# Rimozione del riquadro della legenda
ax.legend(frameon=False)

# Sposta leggermente l'etichetta dell'asse Y a sinistra
ax.set_ylabel('Relative frequency [%]', labelpad=20)  # labelpad gestisce la distanza
ax.yaxis.set_label_coords(-0.05, 0.5)  # Questo sposta leggermente più a sinistra l'etichetta

# Aggiungi tick ogni 0.5 e mantieni gli interi con etichette
yticks = np.arange(0, np.max([val_j.max(), val_ert.max(), val_ohm.max()]) + 1, 0.5)
ax.set_yticks(yticks)

# Imposta le etichette solo per i numeri interi
ytick_labels = ['' if tick % 1 != 0 else f'{int(tick)}' for tick in yticks]
ax.set_yticklabels(ytick_labels)

# Rimuovi i tick e le etichette dell'asse x
ax.set_xticks([])

ax.set_xlim([0, 250])
ax.set_ylim([0, np.max([val_j.max(), val_ert.max(), val_ohm.max()]) * 1.01])

# Creazione della barra colore con etichette personalizzate
divider = make_axes_locatable(frame)
axBar2 = divider.append_axes("bottom", '5%', pad='7%')

# Sposta l'etichetta della colorbar più in basso
cbar2 = plt.colorbar(sct2, cax=axBar2, orientation='horizontal', ticks=[0, 4, 10, 16, 26, 31, 40, 52, 68, 88, 114, 148, 192, 250], format='%.0f')
cbar2.ax.tick_params(labelsize=16)
cbar2.set_label('Resistivity [Ω⋅m]', labelpad=15)  # Usa labelpad per aumentare la distanza

# Calcolo e stampa delle aree normalizzate per verifica
area_j = np.trapz(val_j, bin_j)
area_ert = np.trapz(val_ert, bin_ert)
area_ohm = np.trapz(val_ohm, bin_ohm)

plt.savefig(r'Figure5_v3.png', dpi=500)
