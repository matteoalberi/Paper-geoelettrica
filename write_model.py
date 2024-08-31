

import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geotiff as gtf
import scipy
import autograd
import src.read_input as ri

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
OHM = True
alpha = 0.07  # ohm weight


# mesh param
dxy = .4  # base cell width
dz = .25  # base cell width in z axis
dom_width_x = 30.0  # domain width x
dom_width_y = 30.0  # domain width y
dom_width_z = 7.0  # domain width z

##############################


# Load topography

topo_xyz = np.loadtxt(topo_filename)

# Load data

if ERT:
    ert_dobs = np.loadtxt(str(ert_data_filename))
    ert_survey = SurveyFromData(ert_dobs)
    #Q = ert_dobs[:, -1]
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

#
# Define resistivity model in S/m (or resistivity model in Ohm m)
air_value = np.log(1e-8)
if all([ERT, OHM]):
    background_value = np.log((np.r_[1 / np.mean(apparent_resistivity), 1e-2]).mean())  # modificare
elif ERT:
    background_value = np.log(1 / np.mean(apparent_resistivity))  # ert_dobs.mean()
elif OHM:
    background_value = np.log(1e-2)

# media res_app ohm+mapper = 8.35 ohm*m

# Find active cells in forward modeling (cell below surface)
ind_active = active_from_xyz(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
active_map = maps.InjectActiveCells(mesh, ind_active, np.exp(air_value))
conductivity_map = active_map * maps.ExpMap()
# Create Wires Map that maps from stacked models to individual model components
# m1 refers to density model, m2 refers to susceptibility

# Define model
x = background_value * np.ones(nC)

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

if ERT and not OHM:
    url = 'C:/Users/lopan/PycharmProjects/STELLA/Res/ERT/ert_model.txt'# inserisci url completa .txt
    filename = 'C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/GCR_inv.txt'
    res_mod = np.loadtxt(filename)

if OHM and not ERT:
    url = 'C:/Users/lopan/PycharmProjects/STELLA/Res/Ohm_model_80perc.txt'# inserisci url completa .txt
    filename = 'C:/Users/lopan/PycharmProjects/STELLA/Res/ohm_20240828-1845_fil.txt'
    res_mod = np.loadtxt(filename)

if ERT and OHM:
    url = 'C:/Users/lopan/PycharmProjects/STELLA/Res/Joint/Joint_model_80perc.txt'# inserisci url completa .txt
    filename = 'C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/joint_14it_fil.txt'
    res_mod = np.loadtxt(filename)

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
coord = plotting_map.mesh.cell_centers
model = plotting_map*res_mod
frmt = '{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.6f}\n'
with open(url, 'w') as fid:
    fid.write('x\ty\tz\trho\n')
    for i in np.arange(len(model)):
        fid.write(frmt.format(coord[i, 0], coord[i, 1], coord[i, 2], model[i]))
