
### PER INTERPOLAZIONE
import numpy as np
from scipy.interpolate import griddata

# Carica i dati dal file, ignorando la prima riga di intestazione
data = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_Joint_80perc.txt', skiprows=1)
x, y, z, rho = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Definisci i limiti del modello
x_min, x_max = -8, 8
y_min, y_max = -9, 9
z_min, z_max = -5, 0

# Filtra i dati per rimanere entro i limiti
mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
x, y, z, rho = x[mask], y[mask], z[mask], rho[mask]

# Creare una griglia regolare per i dati
grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:100j, y_min:y_max:100j, z_min:z_max:100j]

# Interpolare i dati di resistività sulla griglia
grid_rho = griddata((x, y, z), rho, (grid_x, grid_y, grid_z), method='linear', fill_value=rho.min())

# Salva i dati interpolati in un file .npz
np.savez('C:/Users/lopan/PycharmProjects/STELLA/Res/Figure8.npz', grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, grid_rho=grid_rho)


###########################
### PER VIASUALIZZAZIONE
import numpy as np
import mayavi.mlab as mlab

# Carica i dati dal file .npz
data = np.load('C:/Users/lopan/PycharmProjects/STELLA/Res/Figure8.npz')
grid_x = data['grid_x']
grid_y = data['grid_y']
grid_z = data['grid_z']
grid_rho = data['grid_rho']

# Limita l'asse z a -3.5
z_limit = -3.5
mask_z_limit = grid_z >= z_limit

grid_x = grid_x[mask_z_limit]
grid_y = grid_y[mask_z_limit]
grid_z = grid_z[mask_z_limit]
grid_rho = grid_rho[mask_z_limit]

# Assicura che gli array rimangano 3D
grid_x = np.reshape(grid_x, (100, 100, -1))
grid_y = np.reshape(grid_y, (100, 100, -1))
grid_z = np.reshape(grid_z, (100, 100, -1))
grid_rho = np.reshape(grid_rho, (100, 100, -1))

# Applica esagerazione verticale solo per la visualizzazione
vertical_exaggeration = 1.5  # Fattore di esagerazione
grid_z_visual = grid_z * vertical_exaggeration

# Definisci i valori per le maschere
mask_low_min, mask_low_max = 0, 20
mask_high_min, mask_high_max = 0, 0

# Applicare le maschere per visualizzare solo i dati compresi tra 0-20
mask_low = (grid_rho >= mask_low_min) & (grid_rho <= mask_low_max)
mask_combined = mask_low
grid_rho_masked = np.where(mask_combined, grid_rho, np.nan)

# Calcola il volume dei dati con resistività tra 0 e 20
voxel_volume = (grid_x[1, 0, 0] - grid_x[0, 0, 0]) * (grid_y[0, 1, 0] - grid_y[0, 0, 0]) * (grid_z[0, 0, 1] - grid_z[0, 0, 0])
total_volume = np.sum(mask_low) * voxel_volume

# Visualizzare il volume con le maschere
fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))

# Visualizza l'intero volume dei dati con alta trasparenza
vol_full = mlab.pipeline.volume(mlab.pipeline.scalar_field(grid_x, grid_y, grid_z_visual, grid_rho), vmin=np.nanmin(grid_rho), vmax=np.nanmax(grid_rho))
vol_full.module_manager.scalar_lut_manager.lut_mode = 'file'

# Specifica il percorso del file .lut da usare
lut_file_path = r'G:\.shortcut-targets-by-id\1U5Mbs1kWT545VLL1mMUjiV8eOfupEFet\Paper Geoelettrica\Draft\Figure\Figure5\colorscale.lut'
vol_full.module_manager.scalar_lut_manager.load_lut_from_file(lut_file_path)

# Imposta il range della colormap tra 0 e 250
vol_full.module_manager.scalar_lut_manager.use_default_range = False
vol_full.module_manager.scalar_lut_manager.data_range = np.array([0, 250])

# Imposta la funzione di trasferimento dell'opacità
otf_full = vol_full._volume_property.get_scalar_opacity()
otf_full.remove_all_points()
otf_full.add_point(np.nanmin(grid_rho), 0.1)
otf_full.add_point(np.nanmax(grid_rho), 0.1)

# Visualizza i dati con resistività tra 0-20 senza trasparenza, usando lo stesso file .lut
contour = mlab.contour3d(grid_x, grid_y, grid_z_visual, grid_rho_masked, contours=10, colormap='file', opacity=1)

# Carica e applica il file .lut al contour
contour.module_manager.scalar_lut_manager.load_lut_from_file(lut_file_path)
contour.module_manager.scalar_lut_manager.use_default_range = False
contour.module_manager.scalar_lut_manager.data_range = np.array([5, 250])
contour.actor.property.ambient = 0.4
# Aggiungi gli assi con le scritte in nero
axes = mlab.axes(vol_full, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', color=(0, 0, 0), line_width=0.05)
axes.title_text_property.italic = False
axes.label_text_property.italic = False
axes.title_text_property.bold = False
axes.label_text_property.bold = False
axes.label_text_property.orientation = 45
axes.axes.number_of_labels = 8
axes.title_text_property.color = (0, 0, 0)
axes.label_text_property.color = (0, 0, 0)
axes.title_text_property.font_size = 3
axes.label_text_property.font_size = 6
axes.title_text_property.opacity = 0

# Limita gli assi x, y e z
mlab.view(azimuth=0, elevation=90, distance='auto', focalpoint='auto')
mlab.axes(ranges=[-9, 9, -8, 8, z_limit, 0])

# Aggiungi le rette grigie sulla superficie
x_values = np.linspace(-8, 8, 100)
y_positions = [-7.2, -4.4, -1.6, 1.2, 4, 6.8]
z_value = 0
labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']

for y, label in zip(y_positions, labels):
    mlab.plot3d(x_values, [y] * len(x_values), [z_value] * len(x_values), color=(0.5, 0.5,0.5), tube_radius=0.08)
    mlab.text3d(x_values[1], y, z_value + 0.1, label, scale=(0.4, 0.4, 0.4), color=(0, 0, 0))


# Mostra la figura
mlab.show()
