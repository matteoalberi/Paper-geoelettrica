
import os

import time
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geotiff as gtf
import scipy
import autograd
import src.read_input as ri
import matplotlib.ticker as ticker
from simpeg import SolverLU as Solver

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz
from simpeg.utils import model_builder
from simpeg.utils.io_utils.io_utils_electromagnetics import write_dcip_xyz
from scipy.interpolate import interp2d
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    convert_survey_3d_to_2d_lines,
    plot_pseudosection
)

# To plot DC data in 3D, the user must have the plotly package
try:
    import plotly
    from simpeg.electromagnetics.static.utils.static_utils import plot_3d_pseudosection

    has_plotly = True
except ImportError:
    has_plotly = False
    pass

try:
    from pymatsolver import Pardiso
except ImportError:
    from simpeg import SolverLU

mpl.use('TkAgg')
mpl.rcParams.update({"font.size": 16})
write_output = False


#####################
# write topography
quote_step = 10
quote_fine = .1


def write_topo_grid(filepath_tif, quote_step, quote_fine, ZERO):
    quote_tif = gtf.GeoTiff(filepath_tif)
    quote_bbox = quote_tif.tif_bBox
    quote_val = quote_tif.read()
    quote_tmpe = np.abs(quote_bbox[0][0] - quote_bbox[1][0]) / (
                np.shape(quote_val)[0] * 2)  # definisco centro cella del DEM in x
    quote_tmpn = np.abs(quote_bbox[0][1] - quote_bbox[1][1]) / (
                np.shape(quote_val)[1] * 2)  # definisco centro cella del DEM in y
    quote_mine = min(quote_bbox[0][0], quote_bbox[1][0]) + quote_tmpe - ZERO[
        0]  # riposiziono in funzione della centralina
    quote_maxe = max(quote_bbox[0][0], quote_bbox[1][0]) - quote_tmpe - ZERO[0]
    quote_minn = min(quote_bbox[0][1], quote_bbox[1][1]) + quote_tmpn - ZERO[1]
    quote_maxn = max(quote_bbox[0][1], quote_bbox[1][1]) - quote_tmpn - ZERO[1]
    quote_x = np.linspace(quote_mine, quote_maxe,
                          np.shape(quote_val)[0])  # attribuisco ad ogni centro cella un valore in direzione x su bordo
    quote_y = np.linspace(quote_minn, quote_maxn,
                          np.shape(quote_val)[1])  # attribuisco ad ogni centro cella un valore in direzione y su bordo

    # interpolo per avere valori in tutte le celle lungo direzione x e y (centro)
    interp = interp2d(quote_x, quote_y, quote_val)

    quote_xinterp = np.linspace(quote_mine, quote_maxe, int(np.shape(quote_val)[0] * quote_step / quote_fine))
    quote_yinterp = np.linspace(quote_minn, quote_maxn, int(np.shape(quote_val)[1] * quote_step / quote_fine))

    # interpolo per avere valori (X,Y,Z) in tutte le celle (centro)
    X, Y = np.meshgrid(quote_xinterp, quote_yinterp[::-1])
    quote_grid = interp(quote_xinterp, quote_yinterp) - ZERO[2]

    # scrittura file di testo
    quote = np.empty([np.size(X), 3])
    quote[:, 0] = np.reshape(X, (np.size(X),))
    quote[:, 1] = np.reshape(Y, (np.size(Y),))
    quote[:, 2] = np.reshape(quote_grid, (np.size(X),))
    FRMT = '{0:.3f} {1:.3f} {2:.2f}\n'
    with open("quote_grid.xyz", 'w') as fid:
        for i in range(0, np.shape(quote)[0]):
            fid.write(FRMT.format(quote[i, 0], quote[i, 1], quote[i, 2]))


# scrittura file di testo

def WriteXYZ(output_filename, data_raw_filename, gps_filename, survey_type, ZERO, data_type, quad_filename=None):
    if survey_type == 'ERT':
        data_inp = ri.ReadIris(data_raw_filename)
        gps_inp = ri.ReadGps(gps_filename)
        data_obj = ri.ReadQuadConfig(quad_filename)
        data_obj.GetErtIrisRho(data_inp)
        data_obj.GetErtGpsElect(gps_inp)
    if survey_type == 'OHM':
        data_inp = ri.ReadStn(data_raw_filename)
        gps_inp = ri.ReadGps(gps_filename)
        data_inp.LocalizeOhm(gps_inp)
        data_obj = ri.ElectData(typ='OHM')
        data_obj.GetOhmData(data_inp, 0.1)

        # Aggiunta di {14:.5e} per res_app
    FRM = "{0:.8e} {1:.8e} {2:.8e} {3:.8e} {4:.8e} {5:.8e} {6:.8e} {7:.8e} {8:.8e} {9:.8e} {10:.8e} {11:.8e} {12:.5e} {13:.5e} \n"
    mdl=np.min(data_inp.measure.vp[data_inp.measure.vp>0])/np.sqrt(2)

    with open(output_filename, 'w') as fid:
        # fid.write("XA    YA    ZA    XB    YB    ZB    XM    YM    ZM    XN    YN    ZN    V/A    UNCERT LINEID\n RES_APP")
        for i in np.arange(len(data_obj.quad.quad_number)):
            xa = data_obj.elect.elect_lat[int(data_obj.quad.a[i] - 1)] - ZERO[0]
            ya = data_obj.elect.elect_lon[int(data_obj.quad.a[i] - 1)] - ZERO[1]
            za = data_obj.elect.elect_elev[int(data_obj.quad.a[i] - 1)] - ZERO[2]
            xb = data_obj.elect.elect_lat[int(data_obj.quad.b[i] - 1)] - ZERO[0]
            yb = data_obj.elect.elect_lon[int(data_obj.quad.b[i] - 1)] - ZERO[1]
            zb = data_obj.elect.elect_elev[int(data_obj.quad.b[i] - 1)] - ZERO[2]
            xm = data_obj.elect.elect_lat[int(data_obj.quad.m[i] - 1)] - ZERO[0]
            ym = data_obj.elect.elect_lon[int(data_obj.quad.m[i] - 1)] - ZERO[1]
            zm = data_obj.elect.elect_elev[int(data_obj.quad.m[i] - 1)] - ZERO[2]
            xn = data_obj.elect.elect_lat[int(data_obj.quad.n[i] - 1)] - ZERO[0]
            yn = data_obj.elect.elect_lon[int(data_obj.quad.n[i] - 1)] - ZERO[1]
            zn = data_obj.elect.elect_elev[int(data_obj.quad.n[i] - 1)] - ZERO[2]

            rho = data_obj.quad.rho[i]

            # ch = data_inp.measure.channel[i]
            if data_type == 'rho':
                va = rho
            elif data_type == 'v':
                vn = rho
            if survey_type == 'ERT':
                vp = data_inp.measure.vp[i]
                inj = data_inp.measure.in_[i]
                res_app = rho
                Q = data_inp.measure.dev[i]
                #if vp==0:
                    #vp=mdl
                vai = abs(vp / inj)+1e-9
                if not Q <= 3:
                    continue
            if survey_type == 'OHM':
                vai = rho + 1e-9
                res_app = rho
                # if not vai > 0:
                # continue
            if res_app <= -10e40:
                continue

            # if not 0 <= Q <= 5 or not 0.000001 <= rho:
            # #if Q <= -10:
            # if not (0.001 <= rho <= 100) or not (abs(vp) <= 500):
            # s = data_obj.quad.dev[i]
            va = np.log(vai)
            # Aggiungere res_app al format
            fid.write(FRM.format(xa, ya, za, xb, yb, zb, xm, ym, zm, xn, yn, zn, va, res_app))


#########################################################################
# Load data and Construct the DC Survey
# -----------------------
#
# Qui definiamo 5 linee DC che utilizzano una configurazione elettrodica dipolo-dipolo;
# tre linee lungo la direzione Est-Ovest e 2 linee lungo la direzione Nord-Sud.
# Per ogni sorgente, dobbiamo definire le posizioni degli elettrodi AB. Per ogni ricevitore,
# dobbiamo definire le posizioni degli elettrodi MN. Invece di creare l'indagine da zero
# (vedi esempio 1D), utilizzeremo *generat_dcip_sources_line*. Questa utilità ci fornirà
# l'elenco delle sorgenti per una data linea DC/IP. Possiamo aggiungere le sorgenti per più linee
# per creare l'indagine.

#
def SurveyFromData(dobs):
    # Extract source and receiver electrode locations and the observed data
    A_electrodes = dobs[0:, 0:3]
    B_electrodes = dobs[0:, 3:6]
    M_electrodes = dobs[0:, 6:9]
    N_electrodes = dobs[0:, 9:12]

    # Define survey
    unique_tx, k = np.unique(np.c_[A_electrodes, B_electrodes], axis=0, return_index=True)
    n_sources = len(k)
    k = np.sort(k)
    k = np.r_[k, len(dobs) + 1]

    source_list = []
    for ii in range(0, n_sources):
        # MN electrode locations for receivers. Each is an (N, 3) numpy array
        M_locations = M_electrodes[k[ii]: k[ii + 1], :]
        N_locations = N_electrodes[k[ii]: k[ii + 1], :]
        receiver_list = [
            dc.receivers.Dipole(
                M_locations,
                N_locations,
                data_type="volt",
            )
        ]

        # AB electrode locations for source. Each is a (1, 3) numpy array
        A_location = A_electrodes[k[ii], :]
        B_location = B_electrodes[k[ii], :]
        source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

    # Define survey
    return dc.Survey(source_list)


#################################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict DC data.
#
# Defining domain side and minimum cell size
def MeshFromSurvey(dxy, dz, dom_width_x, dom_width_y, dom_width_z, topo_xyz, survey_locations):
    nbcx = 2 ** int(np.round(np.log(dom_width_x / dxy) / np.log(2.0)))  # num. base cells x
    nbcy = 2 ** int(np.round(np.log(dom_width_y / dxy) / np.log(2.0)))  # num. base cells y
    nbcz = 2 ** int(np.round(np.log(dom_width_z / dz) / np.log(2.0)))  # num. base cells z

    # Define the base mesh
    hx = [(dxy, nbcx)]
    hy = [(dxy, nbcy)]
    hz = [(dz, nbcz)]
    mesh = TreeMesh([hx, hy, hz], x0="CCN")

    # Mesh refinement based on topography
    k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)).mean()
    mesh.refine_surface(topo_xyz[k, :], finalize=False)

    # Mesh refinement near sources and receivers.

    mesh.refine_points(survey_locations, padding_cells_by_level=[4, 6, 4], finalize=False)

    # Finalize the mesh
    mesh.finalize()
    return mesh


#########################################################################
# input filepath
# -------------------
filepath_tif = "C:/Users/lopan/PycharmProjects/STELLA/data/vigna_quote.tif"
topo_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/quote_grid.xyz"
ert_data_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/VIGNA_3D_DICEMBRE_23.xyz"
ohm_data_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/VIGNA_ohm_DICEMBRE_23.xyz"
ert_raw_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/VIGNA_3D_DICEMBRE_23.bin"
ert_gps_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/vigna_ert_gau-o.txt"
ert_quad_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/vigna_dip-dip.txt"
ohm_raw_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/vigna_ohm_mod.stn"
ohm_gps_filename = "C:/Users/lopan/PycharmProjects/STELLA/data/vigna_ohm_gau-o.txt"
out_fold = "C:/Users/lopan/PycharmProjects/STELLA/Res"
# centralina
ZERO = np.r_[1700614.677, 4761544.891, 229.644]
ERT = True
OHM = False
alpha = 0.07  # ohm weight


# mesh param
dxy = .4  # base cell width
dz = .25  # base cell width in z axis
dom_width_x = 30.0  # domain width x
dom_width_y = 30.0  # domain width y
dom_width_z = 7.0  # domain width z

##############################


#########################################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. In our case, our survey takes place within a circular
# depression.
#

# Load topography

topo_xyz = np.loadtxt(topo_filename)

# Load data

if ERT:
    ert_dobs = np.loadtxt(str(ert_data_filename))
    ert_survey = SurveyFromData(ert_dobs)
    Q = ert_dobs[:, -3]
    apparent_resistivity = ert_dobs[:, -1]
    ert_dobs = ert_dobs[:, -2]
    ert_data_object = data.Data(ert_survey, dobs=ert_dobs, standard_deviation=0.3 * np.abs(ert_dobs))

if OHM:
    ohm_dobs = np.loadtxt(str(ohm_data_filename))
    ohm_survey = SurveyFromData(ohm_dobs)
    ohm_dobs = ohm_dobs[:, -2]
    ohm_data_object = data.Data(ohm_survey, dobs=ohm_dobs, standard_deviation=0.2 * np.abs(ohm_dobs))

# mesh

survey_locations = np.empty([0, 3])
if ERT:
    survey_locations = np.r_[
        survey_locations,
        ert_survey.unique_electrode_locations,
    ]
if OHM:
    survey_locations = np.r_[
        survey_locations,
        ohm_survey.unique_electrode_locations,
    ]

mesh = MeshFromSurvey(dxy, dz, dom_width_x, dom_width_y, dom_width_z, topo_xyz, survey_locations)

# Define resistivity model in S/m (or resistivity model in Ohm m)
air_value = np.log(1e-8)


# media res_app ohm+mapper = 8.35 ohm*m

# Find active cells in forward modeling (cell below surface)
ind_active = active_from_xyz(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
active_map = maps.InjectActiveCells(mesh, ind_active, np.exp(air_value))
conductivity_map = active_map * maps.ExpMap()

##########################################################
# Project Survey to Discretized Topography
# ----------------------------------------
#
# It is important that electrodes are not modeled as being in the air. Even if the
# electrodes are properly located along surface topography, they may lie above
# the *discretized* topography. This step is carried out to ensure all electrodes
# lie on the discretized surface.
#
#
if ERT:
    ert_survey.drape_electrodes_on_topography(mesh, ind_active, option="top")
if OHM:
    ohm_survey.drape_electrodes_on_topography(mesh, ind_active, option="top")


# Plot conductivity model
if ERT and OHM:

    filename = 'C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/joint_14it_fil.txt'
    res_mod = np.loadtxt(filename)

if ERT and not OHM:

    filename = 'C:/Users/lopan/PycharmProjects/STELLA/Res/ERT/res_mod.txt'
    res_mod = np.loadtxt(filename)

if OHM and not ERT:

    filename = 'C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/ohm_13it_fil.txt'
    res_mod = np.loadtxt(filename)



## Calcolare le statistiche richieste
#num_data_res_mod = len(res_mod)
#min_value_res_mod = np.min(res_mod)
#max_value_res_mod = np.max(res_mod)
#mean_value_res_mod = np.mean(res_mod)
#median_value_res_mod = np.median(res_mod)
#std_deviation_res_mod = np.std(res_mod)
#variance_res_mod = np.var(res_mod)
#
## Stampare i risultati
#print(f"Number of data points: {num_data_res_mod}")
#print(f"Minimum value: {min_value_res_mod}")
#print(f"Maximum value: {max_value_res_mod}")
#print(f"Mean value: {mean_value_res_mod}")
#print(f"Median value: {median_value_res_mod}")
#print(f"Standard deviation: {std_deviation_res_mod}")
#print(f"Variance: {variance_res_mod}")
#
### Calcolare i valori di +3σ e -3σ
##plus_2sigma = mean_value_res_mod + 2 * std_deviation_res_mod
##minus_2sigma = mean_value_res_mod - 2 * std_deviation_res_mod
##
### Creare la figura e l'istogramma
fig = plt.figure(figsize=(10, 5))  # Imposta la dimensione del grafico
plt.hist(res_mod, bins=30, color='b', edgecolor='k', alpha=0.7)
##
### Aggiungere linee per la media e le deviazioni standard
##plt.axvline(mean_value_res_mod, color='r', linestyle='dashed', linewidth=2, label='Mean')
##plt.axvline(plus_2sigma, color='g', linestyle='dashed', linewidth=2, label='+2σ')
##plt.axvline(minus_2sigma, color='g', linestyle='dashed', linewidth=2, label='-2σ')
##
### plt.xscale('log')
### Aggiungere il titolo e le etichette degli assi
plt.title('Distribution of Resistivity Values')  # Titolo del grafico
plt.xlabel('Resistivity [Ohm/m]')  # Etichetta dell'asse X
plt.ylabel('Frequency')  # Etichetta dell'asse Y
plt.grid(True)
plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM/risultati_13_iter/res_mod_distribution_log.png', dpi=300, bbox_inches='tight')
plt.close(fig)  # Chiude la figura per liberare memoria

if ERT and OHM:
    # Definizione dei colori

    model_interp = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_Joint_80perc.txt', skiprows=1, delimiter='\t')
    colormap = np.loadtxt(
        r"G:\.shortcut-targets-by-id\1U5Mbs1kWT545VLL1mMUjiV8eOfupEFet\Paper Geoelettrica\Draft\Figure\Figure5\colorscale.lut",
        skiprows=1, delimiter=' ')
    cmap_new = matplotlib.colors.ListedColormap(colormap)

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
    cscale = np.r_[0, 2, 4, 6, 7, 9, 14, 17, 22, 29, 39, 53, 72, 90, 110, 130, 160, 188]

    finterp_color1 = scipy.interpolate.interp1d(cscale, colors[:, 0])
    finterp_color2 = scipy.interpolate.interp1d(cscale, colors[:, 1])
    finterp_color3 = scipy.interpolate.interp1d(cscale, colors[:, 2])

    # Calcola i percentili desiderati
    # percentiles = np.percentile(res_mod, [0, 1, 5, 10, 20, 40, 60, 80, 90, 95, 99, 100])
    # percentile_labels = ['0th', '1th', '5th', '10th', '20th', '40th', '60th', '80th', '90th', '95th', '99th', '100th']
    linear_scale = np.linspace(1, 188, 15)
    log_scale = np.logspace(0, np.log10(189), 15) - 1

    percentiles = linear_scale + (log_scale - linear_scale) * .9
    percentiles[0] = 0
    percentiles[-1] = 188

    colors = np.concatenate([np.reshape(finterp_color1(percentiles), [-1, 1]),
                             np.reshape(finterp_color2(percentiles), [-1, 1]),
                             np.reshape(finterp_color3(percentiles), [-1, 1]),
                             np.ones([len(percentiles), 1]) * 0], axis=1)

    percentile_labels = [str(np.round(i, 2)) for i in percentiles]
    # Crea una mappa colori personalizzata
    cmap = mcolors.ListedColormap(colors)

    bounds = percentiles
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Istogramma con linee verticali per i percentili
    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(res_mod, bins=100, weights=np.ones_like(res_mod) / len(res_mod) * 100, color='gray',
                               edgecolor='black', alpha=0.7)

    # Aggiungi linee verticali per i percentili con colori specifici
    for percentile, color in zip(percentiles[1:-1], colors):
        ax.axvline(percentile, color=color, linestyle='dashed', linewidth=2)

    # Imposta etichette e titolo
    ax.set_xlabel('Resistivity (Ohm*m)')
    ax.set_ylabel('Frequency (%)')
    ax.set_ylim(0, 10)

    # Configurazione della barra colori sulla destra
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Barra colori verticale sulla destra
    cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')

    # Aggiungi etichette dei valori di resistività e percentili
    cbar.set_ticks(bounds)
    cbar_labels = [f'{p_label} ' for p_label in zip(percentile_labels)]
    cbar.ax.set_yticklabels(cbar_labels)
    cbar.set_label("Resistivity [Ohm*m]")

    # Salvare la figura
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\perc.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 7.8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mat1=ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])
    fig.colorbar(mat1, ax=ax1)

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=47,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\x7.8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 7.8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_x7_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=47,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_x7.8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    #########
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 2.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=40,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\x3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 2.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_x3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=40,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_x3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    #########
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 0.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=34,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\x1.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 0.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_x1.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=34,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_x1.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    #########
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -2.1)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\x-2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -2.1)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_x-2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_x-2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    #########
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -5)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\x-5.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] ==- 5)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_x-5.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_x-5.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    #########
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=15,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\x-8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_x-8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=15,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_x-8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########
    model_interp = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/Joint/XYZr_Joint.txt', skiprows=1,
                              delimiter='\t')
    colormap = np.loadtxt(
         r"G:\.shortcut-targets-by-id\1U5Mbs1kWT545VLL1mMUjiV8eOfupEFet\Paper Geoelettrica\Draft\Figure\Figure5\colorscale.lut",
        skiprows=1, delimiter=' ')
    cmap_new = matplotlib.colors.ListedColormap(colormap)
  # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==5.8)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=48,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\y5_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == 5.8)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_y5_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=48,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_y5_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria


############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==3.1)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=42,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\y3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == 3.1)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_y3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=42,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_y3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==1.3)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=35,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\y1_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == 1.3)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_y1_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=35,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_y1_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==-1.5)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\y-1_6.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == - 1.5)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_y-1_6.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_y-1_6.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == - 4.2)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\y-4_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == - 4.2)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_y-4_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_y-4_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==-7)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=14,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\y-7.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == -7)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\model_y-7.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=14,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\mesh_y-7.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria


#########################################
if OHM:
    # Definizione dei colori

    model_interp = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_CCR_80perc.txt', skiprows=1, delimiter='\t')
    colormap = np.loadtxt(
        r"G:\.shortcut-targets-by-id\1U5Mbs1kWT545VLL1mMUjiV8eOfupEFet\Paper Geoelettrica\Draft\Figure\Figure5\colorscale.lut",
        skiprows=1, delimiter=' ')
    cmap_new = matplotlib.colors.ListedColormap(colormap)

    colors = np.array([
        [0, 0, 255, 255],
        [0, 60, 254, 255],
        [1, 119, 253, 255],
        [25, 181, 229, 255],
        [89, 186, 241, 255],
        [72,  228,  246, 255],
        [102,  233,  250, 255],
        [6, 247, 183, 255],
        [245, 251, 207, 255],
        [255, 255, 160, 255],
        [255, 203, 118, 255],
        [255, 180, 55, 255],
        [250, 148, 47, 255],
        [253,  83,  56, 255],
        [246, 63, 0, 255],
        [246, 0, 0, 255],
        [208, 0, 0, 255],
        [170, 28, 0, 255]
    ]) / 255.0
    cscale=np.r_[0, 2, 4, 6, 7, 9, 14, 17, 22, 29, 39, 53, 72, 90, 110, 130, 160, 270]


    finterp_color1 = scipy.interpolate.interp1d(cscale, colors[:, 0])
    finterp_color2 = scipy.interpolate.interp1d(cscale, colors[:, 1])
    finterp_color3 = scipy.interpolate.interp1d(cscale, colors[:, 2])


    # Calcola i percentili desiderati
    #percentiles = np.percentile(res_mod, [0, 1, 5, 10, 20, 40, 60, 80, 90, 95, 99, 100])
    #percentile_labels = ['0th', '1th', '5th', '10th', '20th', '40th', '60th', '80th', '90th', '95th', '99th', '100th']
    linear_scale=np.linspace(1, 250, 15)
    log_scale=np.logspace(0, np.log10(251), 15)-1

    percentiles = linear_scale+(log_scale-linear_scale)*.9
    percentiles[0] =0
    percentiles[-1] = 250

    colors = np.concatenate([np.reshape(finterp_color1(percentiles), [-1, 1]),
                             np.reshape(finterp_color2(percentiles), [-1, 1]),
                             np.reshape(finterp_color3(percentiles), [-1, 1]),
                             np.ones([len(percentiles), 1])*0], axis=1)

    percentile_labels = [str(np.round(i, 2)) for i in percentiles]
    # Crea una mappa colori personalizzata
    cmap = mcolors.ListedColormap(colors)
    bounds = percentiles
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Istogramma con linee verticali per i percentili
    fig, ax = plt.subplots(figsize=(10, 5))

    # Aggiungi linee verticali per i percentili con colori specifici
    for percentile, color in zip(percentiles[1:-1], colors):
        ax.axvline(percentile, color=color, linestyle='dashed', linewidth=2)

    # Imposta etichette e titolo
    ax.set_xlabel('Resistivity (Ohm*m)')
    ax.set_ylabel('Frequency (%)')
    ax.set_ylim(0, 5)

    # Configurazione della barra colori sulla destra
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Barra colori verticale sulla destra
    cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')

    # Aggiungi etichette dei valori di resistività e percentili
    cbar.set_ticks(bounds)
    cbar_labels = [f'{p_label} ' for p_label in zip(percentile_labels)]
    cbar.ax.set_yticklabels(cbar_labels)
    cbar.set_label("Resistivity [Ohm*m]")
    # Salvare la figura
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\perc_50_13it.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==5.8)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=48,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\y5_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == 5.8)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\model_y5_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=48,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\mesh_y5_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria


############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==3.1)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=42,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\y3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == 3.1)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\model_y3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=42,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\mesh_y3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==1.3)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mat1=ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])
    fig.colorbar(mat1, ax=ax1)
    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=35,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\y1_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == 1.3)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\model_y1_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=35,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\mesh_y1_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==-1.5)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\y-1_6.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == - 1.5)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\model_y-1_6.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\mesh_y-1_6.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == - 4.2)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\y-4_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == - 4.2)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\model_y-4_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\mesh_y-4_2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria

############
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] ==-7)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=14,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\y-7.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
######
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 1] == -7)[0]
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 0])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 0]), np.max(model_interp[ix, 0]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\model_y-7.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="y",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=14,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("x (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-3, 0])
    ax1.set_xlim([-9, 9])
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\mesh_y-7.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria



########################################################################################################################

if ERT:
    # Definizione dei colori
    model_interp = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/ERT/XYZr_GCR.txt', skiprows=1, delimiter='\t')
    colormap = np.loadtxt(
        r"G:\.shortcut-targets-by-id\1U5Mbs1kWT545VLL1mMUjiV8eOfupEFet\Paper Geoelettrica\Draft\Figure\Figure5\colorscale.lut",
        skiprows=1, delimiter=' ')
    cmap_new = matplotlib.colors.ListedColormap(colormap)

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
    cscale = np.r_[0, 2, 4, 6, 7, 9, 14, 17, 22, 29, 39, 53, 72, 90, 110, 130, 160, 188]

    finterp_color1 = scipy.interpolate.interp1d(cscale, colors[:, 0])
    finterp_color2 = scipy.interpolate.interp1d(cscale, colors[:, 1])
    finterp_color3 = scipy.interpolate.interp1d(cscale, colors[:, 2])

    # Calcola i percentili desiderati
    # percentiles = np.percentile(res_mod, [0, 1, 5, 10, 20, 40, 60, 80, 90, 95, 99, 100])
    # percentile_labels = ['0th', '1th', '5th', '10th', '20th', '40th', '60th', '80th', '90th', '95th', '99th', '100th']
    linear_scale = np.linspace(1, 188, 15)
    log_scale = np.logspace(0, np.log10(189), 15) - 1

    percentiles = linear_scale + (log_scale - linear_scale) * .9
    percentiles[0] = 0
    percentiles[-1] = 188

    colors = np.concatenate([np.reshape(finterp_color1(percentiles), [-1, 1]),
                             np.reshape(finterp_color2(percentiles), [-1, 1]),
                             np.reshape(finterp_color3(percentiles), [-1, 1]),
                             np.ones([len(percentiles), 1])*0], axis=1)

    percentile_labels = [str(np.round(i, 2)) for i in percentiles]
    # Crea una mappa colori personalizzata
    cmap = mcolors.ListedColormap(colors)
    bounds = percentiles
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Istogramma con linee verticali per i percentili
    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(res_mod, bins=100, weights=np.ones_like(res_mod) / len(res_mod) * 100, color='gray',
                               edgecolor='black', alpha=0.7)

    # Aggiungi linee verticali per i percentili con colori specifici
    for percentile, color in zip(percentiles[1:-1], colors):
        ax.axvline(percentile, color=color, linestyle='dashed', linewidth=2)

    # Imposta etichette e titolo
    ax.set_xlabel('Resistivity (Ohm*m)')
    ax.set_ylabel('Frequency (%)')
    ax.set_ylim(0, 10)

    # Configurazione della barra colori sulla destra
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Barra colori verticale sulla destra
    cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')

    # Aggiungi etichette dei valori di resistività e percentili
    cbar.set_ticks(bounds)
    cbar_labels = [f'{p_label} ' for p_label in zip(percentile_labels)]
    cbar.ax.set_yticklabels(cbar_labels)
    cbar.set_label("Resistivity [Ohm*m]")

    # Salvare la figura
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\perc.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] ==7.8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0+(model_interp[ix[i], 1] - y0) * (z1-z0)/(y1-y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mat1=ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])
    fig.colorbar(mat1, ax=ax1)
    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=47,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\x7.8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 7.8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\model_x7_8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=47,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\mesh_x7.8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########
    ####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 2.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=40,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\x3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 2.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\model_x3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
    ####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=40,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\mesh_x3.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########
####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] ==0.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0+(model_interp[ix[i], 1] - y0) * (z1-z0)/(y1-y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=47,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\x1.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == 0.9)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\model_x1.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=34,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\mesh_x1.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########
####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] ==-2.1)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0+(model_interp[ix[i], 1] - y0) * (z1-z0)/(y1-y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\x-2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -2.1)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -5.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\model_x-2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=28,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\mesh_x-2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########
####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] ==-5)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0+(model_interp[ix[i], 1] - y0) * (z1-z0)/(y1-y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\x-5.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == - 5)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4.5
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\model_x-5.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=21,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\mesh_x-5.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########
####
    # Modello di resistività
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] ==-8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0+(model_interp[ix[i], 1] - y0) * (z1-z0)/(y1-y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])), len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new,vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]),np.min(model_interp[ix, 2]), np.max(model_interp[ix, 2])])

    # Creare il grafico del modello di resistività
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=15,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\x-8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####

    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ix = np.where(model_interp[:, 0] == -8)[0]
    for i in np.arange(len(model_interp[ix, 0])):
        y0 = -8
        z0 = -1
        y1 = 0
        z1 = -4
        y2 = 8
        z2 = -1
        p1 = z0 + (model_interp[ix[i], 1] - y0) * (z1 - z0) / (y1 - y0)
        p2 = z1 + (model_interp[ix[i], 1] - y1) * (z2 - z1) / (y2 - y1)
        if model_interp[ix[i], 2] < max(p1, p2):
            model_interp[ix[i], 3] = np.nan
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    ax1.matshow(np.flip(np.reshape(model_interp[ix, 3], [len(np.unique(model_interp[ix, 1])),
                                                         len(np.unique(model_interp[ix, 2]))]).transpose(),
                        axis=0), interpolation='bilinear', cmap=cmap_new, vmin=0, vmax=250,
                extent=[np.min(model_interp[ix, 1]), np.max(model_interp[ix, 1]), np.min(model_interp[ix, 2]),
                        np.max(model_interp[ix, 2])])

    # Modello di resistività2
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\model_x-8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
####
    fig = plt.figure(figsize=(10, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
    mesh.plot_slice(
        plotting_map * res_mod,
        ax=ax1,
        normal="x",  # se x sono perpendicolare ai filari se y sono parallelo ai filari
        ind=15,
        grid=True,
        pcolor_opts={"cmap": cmap, "norm": norm}
    )
    ax1.set_title("Resistivity Model")
    ax1.set_xlabel("y (m)")
    ax1.xaxis.set_label_position('bottom')
    ax1.xaxis.tick_bottom()
    ax1.set_ylabel("z (m)")
    ax1.set_ylim([-5, 0])
    ax1.set_xlim([-8, 8])

    # Salvare la figura del modello di resistività senza la colorbar
    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\ERT\mesh_x-8.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)  # Chiude la figura per liberare memoria
#########



#################################################


Q1 = np.percentile(res_mod, 25)
Q3 = np.percentile(res_mod, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR-30
upper_bound = Q3 + 1.5 * IQR

outliers = res_mod[(res_mod < lower_bound) | (res_mod > upper_bound)]  # Trova gli indici degli outliers
outlier_indices = np.where((res_mod < lower_bound) | (res_mod > upper_bound))[0]

ix_active_ind = np.where(ind_active)[0]
centers = mesh.cell_centers
sizes = mesh.h_gridded
add_val = np.min(sizes) / 5

def replace_outliers_with_nearest_mean(res_mod, outliers, outlier_indices):

    while np.any(np.r_[(res_mod < lower_bound) | (res_mod > upper_bound)]):
        # print('inizio correzioni')
        for idx in outlier_indices:
            # Trova i quattro indici più vicini
            cent = centers[ix_active_ind[idx], :]
            siz = sizes[ix_active_ind[idx], :]
            ix_mesh = []
            ix_mesh.append(mesh.closest_points_index(cent+np.r_[siz[0]/2 + add_val, 0, 0]))
            ix_mesh.append(mesh.closest_points_index(cent+np.r_[siz[0]/2 - add_val, 0, 0]))
            ix_mesh.append(mesh.closest_points_index(cent+np.r_[0, siz[1]/2 + add_val, 0]))
            ix_mesh.append(mesh.closest_points_index(cent+np.r_[0, siz[1]/2 - add_val, 0]))
            ix_mesh.append(mesh.closest_points_index(cent+np.r_[0, 0, siz[2]/2 + add_val]))
            ix_mesh.append(mesh.closest_points_index(cent+np.r_[0, 0, siz[2]/2 - add_val]))
            nearest_indices = np.array([np.where(ix_active_ind == index)[0] for index in ix_mesh if len(np.where(ix_active_ind == index)[0]) == 1])

            # Calcola la media dei valori ai quattro indici più vicini
            if np.mean(res_mod[nearest_indices]) <= upper_bound and np.mean(res_mod[nearest_indices]) >= lower_bound:
                nearest_mean = np.mean(res_mod[nearest_indices])
            elif np.mean(res_mod[nearest_indices]) > upper_bound:
                nearest_mean = upper_bound
            else:
                nearest_mean = lower_bound

            # Sostituisci il valore dell'outlier con la media calcolata
            res_mod[idx] = nearest_mean
            # print('fatta la correzione ' + str(np.where(outlier_indices == idx)[0]) + ' di ' + str(len(outlier_indices)))
        # aggiorno gli indici degli outlayers
        outlier_indices = np.where((res_mod < lower_bound) | (res_mod > upper_bound))[0]

    return res_mod


# Sostituisci ciascun outlier
res_mod = replace_outliers_with_nearest_mean(res_mod, outliers, outlier_indices)


# Creare il box plot con personalizzazione
fig, ax = plt.subplots()
box = ax.boxplot(res_mod, vert=False, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', color='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='orange'),
                 flierprops=dict(marker='o', color='red', markersize=5))


#ax.text(lower_bound, 1.02, ' "Minimum"\n(Q1 - 1.5*IQR)', verticalalignment='center', horizontalalignment='center', fontsize=10)
#ax.text(upper_bound, 1.02, ' "Maximum"\n(Q3 + 1.5*IQR)', verticalalignment='center', horizontalalignment='center', fontsize=10)
#ax.set_xlim(-1, 300)
plt.title('Box Plot per identificare outliers')
plt.xlabel('Valori ')

# Aggiungere legenda dettagliata
handles = [
    plt.Line2D([0], [0], color='red', lw=2, label='Box Plot'),
    plt.Line2D([0], [0], marker='o', color='black', markersize=5, linestyle='None', label='Outliers'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Mediana')
]
ax.legend(handles=handles, loc='upper right')

plt.savefig(r'C:/Users/lopan/PycharmProjects/STELLA/Res/JOINT/box_plot.png')
plt.close()

numero_totale_dati = len(res_mod)
numero_outliers = len(outliers)

# Calcolare la percentuale di outliers

percentuale_outliers = (numero_outliers / numero_totale_dati) * 100

if OHM and ERT:
    with open(out_fold + '/' + time.strftime('joint_%Y%m%d-%H%M_fil') + '.txt', 'w') as fid:
        for i in np.arange(len(res_mod)):
            fid.write(str(res_mod[i]) + '\n')

if OHM and not ERT:
    with open(out_fold + '/' + time.strftime('ohm_%Y%m%d-%H%M_fil') + '.txt', 'w') as fid:
        for i in np.arange(len(res_mod)):
            fid.write(str(res_mod[i]) + '\n')

if ERT and not OHM:
    with open(out_fold + '/' + time.strftime('ert_%Y%m%d-%H%M_fil') + '.txt', 'w') as fid:
        for i in np.arange(len(res_mod)):
            fid.write(str(res_mod[i]) + '\n')

################


# Create a mask for values between 0 and 0.1
mask = (res_mod > 0) & (res_mod <= 0.1)

# Redistribute these values to be between 0.1 and 1 linearly
min_val, max_val = 1e-4, 1
values_to_redistribute = res_mod[mask]

# Linear transformation: new_value = min_val + (old_value - old_min) * (new_range / old_range)
old_min, old_max = values_to_redistribute.min(), values_to_redistribute.max()
new_range = max_val - min_val
old_range = old_max - old_min

res_mod[mask] = min_val + (values_to_redistribute - old_min) * (new_range / old_range)

# Display the modified matrix
print(data_matrix)