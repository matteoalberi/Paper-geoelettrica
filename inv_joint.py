# -*- coding: utf-8 -*-

# Import modules
# --------------
#
#

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


mpl.rcParams.update({"font.size": 16})
write_output = False

# plt.switch_backend('agg')
# sphinx_gallery_thumbnail_number = 2


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
    #mdl=np.min(data_inp.measure.vp[data_inp.measure.vp>0])/np.sqrt(2)

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
                if not Q <= 5:
                    continue
            if survey_type == 'OHM':
                vai = rho
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


# Define the DC simulation
def DcFw(x, mesh, survey, conductivity_map):
    simulation = dc.simulation.Simulation3DNodal(
        mesh, survey=survey, sigmaMap=conductivity_map, solver=Solver
    )
    # Predici i dati eseguendo la simulazione. I dati sono la tensione misurata
    # normalizzata dalla corrente di sorgente in unità di V/A.

    vpred = simulation.dpred(x)
    return apparent_resistivity_from_voltage(
        survey,
        vpred,
    )


#######################################################################
# Define Inverse Problem
# ----------------------
#
## Il problema inverso è definito da 3 elementi:
#
#     1) Data Misfit: misura di quanto bene il nostro modello ricostruito spiega i dati di campo
#     2) Regularization: vincoli posti sul modello ricostruito e informazioni a priori
#     3) Optimization: l'approccio numerico utilizzato per risolvere il problema inverso
#
#

# Definiamo il data misfit. Qui la discrepanza dei dati è la norma L2 del residuo pesato
# tra i dati osservati e i dati predetti per un determinato modello. All'interno del data misfit,
# il residuo tra dati predetti e osservati è normalizzato per la deviazione standard dei dati.

def InvProb(mesh, survey, conductivity_map, data_object, ind_active, starting_conductivity_model, maxIter, maxIterLS,
            maxIterCG, tolCG, tolF, tolX, eps, tolG):
    simulation = dc.simulation.Simulation3DNodal(
        mesh, survey=survey, sigmaMap=conductivity_map, solver=SolverLU, storeJ=True
    )
    dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
    reg = regularization.WeightedLeastSquares(
        mesh, active_cells=ind_active, reference_model=starting_conductivity_model
    )
    reg.reference_model_in_smooth = True
    opt = optimization.InexactGaussNewton(
        maxIter=maxIter, maxIterLS=maxIterLS, maxIterCG=maxIterCG, tolCG=tolCG, tolF=tolF, tolX=tolX, eps=eps,
        tolG=tolG)
    return inverse_problem.BaseInvProblem(dmis, reg, opt)


global gamma
def InvProbJoin(mesh, ert_survey, ohm_survey, conductivity_map, ert_data_object, ohm_data_object, ind_active, alpha,
                starting_conductivity_model, maxIter, maxIterLS, maxIterCG, tolCG, tolF, tolX, eps, tolG):
    global gamma
    ert_simulation = dc.simulation.Simulation3DNodal(
        mesh, survey=ert_survey, sigmaMap=conductivity_map, solver=Solver, storeJ=True
    )
    ohm_simulation = dc.simulation.Simulation3DNodal(
        mesh, survey=ohm_survey, sigmaMap=conductivity_map, solver=Solver, storeJ=True
    )
    ert_simulation.model = starting_conductivity_model
    ohm_simulation.model = starting_conductivity_model
    ert_dmis = data_misfit.L2DataMisfit(simulation=ert_simulation, data=ert_data_object)
    ohm_dmis = data_misfit.L2DataMisfit(simulation=ohm_simulation, data=ohm_data_object)
    phi_ert = ert_dmis(ert_dmis.simulation.model)
    phi_ohm = ohm_dmis(ohm_dmis.simulation.model)
    gamma = (alpha * phi_ert) / (alpha * phi_ert + (1 - alpha) * phi_ohm)
    print(gamma)
    dmis = (1 - gamma) * ert_dmis + gamma * ohm_dmis

    # frmt = '{0:.4f}\t{1:.4f}\t{2:.4f}\n'
    # with open('C:/Users/lopan/PycharmProjects/STELLA/prova.txt', 'a') as fid:
    #    fid.write(frmt.format(dmis))

    reg = regularization.WeightedLeastSquares(
        mesh,
        active_cells=ind_active,
        reference_model=starting_conductivity_model
    )
    reg.reference_model_in_smooth = True
    opt = optimization.InexactGaussNewton(
        maxIter=maxIter, maxIterLS=maxIterLS, maxIterCG=maxIterCG, tolCG=tolCG, tolF=tolF, tolX=tolX, eps=eps, tolG=tolG
    )


    return inverse_problem.BaseInvProblem(dmis, reg, opt)


#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Qui definiamo le direttive che vengono eseguite durante l'inversione. Questo
# include lo schema di raffreddamento per il parametro di trade-off (beta), i criteri
# di arresto per l'inversione e il salvataggio dei risultati dell'inversione ad ogni iterazione.
# Definiamo anche il monitoraggio del data misfit, d ibeta e della regolarizzazione

############################################################
# Predict DC Resistivity Data
# ---------------------------
#
# Qui prediciamo i dati di resistività DC. Se l'argomento chiave *sigmaMap* è definito,
# la simulazione si aspetterà un modello di conduttività. Se l'argomento chiave
# *rhoMap* è definito, la simulazione si aspetterà un modello di resistività.
#
#


class MonitorInversion(directives.InversionDirective):
    def initialize(self):
        self.data_misfit_vals = []
        # self.data_misfit_ert_vals= []
        # self.data_misfit_ohm_vals = []
        self.beta_vals = []
        self.reg_vals = []
        self.chi_vals = []
        # self.chi_squared_ert_vals=[]
        # self.chi_squared_ohm_vals=[]

    def endIter(self):
        data_misfit = self.invProb.dmisfit(self.invProb.model)

        if ERT and OHM:
            # data_misfit_ert=self.invProb.dmisfit(self.invProb.model)/(2*alpha1)
            # data_misfit_ohm =  self.invProb.dmisfit(self.invProb.model)/ (2*alpha2)
            # chi_squared_ert = data_misfit_ert / ert_dobs.size
            # chi_squared_ohm = data_misfit_ohm/ ohm_dobs.size
            global gamma
            chi_squared = data_misfit / ((1 - gamma) * ert_dobs.size + gamma * ohm_dobs.size)
        if OHM and not ERT:
            chi_squared = self.invProb.dmisfit(self.invProb.model) / ohm_dobs.size
        if ERT and not OHM:
            chi_squared = self.invProb.dmisfit(self.invProb.model) / ert_dobs.size
        beta = self.invProb.beta
        reg = self.invProb.reg(self.invProb.model)

        self.data_misfit_vals.append(data_misfit)
        # self.data_misfit_ert_vals.append(data_misfit_ert)
        # self.data_misfit_ohm_vals.append(data_misfit_ohm)
        self.chi_vals.append(chi_squared)
        # self.chi_squared_ert_vals.append(chi_squared_ert)
        # self.chi_squared_ohm_vals.append(chi_squared_ohm)
        self.beta_vals.append(beta)
        self.reg_vals.append(reg)

        print(f"Iteration: {self.opt.iter}")
        print(f"Data Misfit: {data_misfit}")
        # print(f"Data Misfit: {data_misfit}")
        # print(f"Data Misfit_ert: {data_misfit_ert}")
        # print(f"Data Misfit_ohm: {data_misfit_ohm}")
        print(f"Chi_squared: {chi_squared}")
        # print(f"Chi_squared_ert: {chi_squared_ert}")
        # print(f"Chi_squared_ohm: {chi_squared_ohm}")
        print(f"Beta: {beta}")
        print(f"Regularization: {reg}")
        print("-" * 50)

    def save_results(self, filepath):
        with open(filepath, 'w') as f:
            f.write(
                'Iteration, Data misfit joint,  Beta, Regularization, Chi_squared\n')
            for i, (dm, beta, reg, chi) in enumerate(
                    zip(self.data_misfit_vals, self.beta_vals, self.reg_vals, self.chi_vals)):
                f.write(f'{i}, {dm}, {beta}, {reg}, {chi}\n')


def DirProb(beta0_ratio, coolingFactor, coolingRate, chifact):
    # Defining a starting value for the trade-off parameter (beta) between the data
    # misfit and the regularization.
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)

    # Set the rate of reduction in trade-off parameter (beta) each time the
    # the inverse problem is solved. And set the number of Gauss-Newton iterations
    # for each trade-off paramter value.
    beta_schedule = directives.BetaSchedule(coolingFactor=coolingFactor, coolingRate=coolingRate)

    # Apply and update sensitivity weighting as the model updates
    update_sensitivity_weights = directives.UpdateSensitivityWeights()

    # Options for outputting recovered models and predicted data for each beta.
    save_iteration = directives.SaveOutputEveryIteration(save_txt=True)

    # Setting a stopping criteria for the inversion.
    target_misfit = directives.TargetMisfit(chifact=chifact)

    # Apply and update preconditioner as the model updates
    update_jacobi = directives.UpdatePreconditioner()

    monitor_inversion = MonitorInversion()

    # The directives are defined as a list.
    return [
        update_sensitivity_weights,
        starting_beta,
        beta_schedule,
        save_iteration,
        target_misfit,
        update_jacobi,
        monitor_inversion,
    ]


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
# write xyz ERT
if ERT:
    WriteXYZ(ert_data_filename, ert_raw_filename, ert_gps_filename, 'ERT', ZERO, 'v', quad_filename=ert_quad_filename)
# write xyz ohm
if OHM:
    WriteXYZ(ohm_data_filename, ohm_raw_filename, ohm_gps_filename, 'OHM', ZERO, 'v')
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

#CALCOLO STATISTICHE DATI INPUT

#if ERT:
#    # Calcolare le statistiche richieste
#    num_data = len(ert_dobs)
#    min_value = np.min(ert_dobs)
#    max_value = np.max(ert_dobs)
#    mean_value = np.mean(ert_dobs)
#    median_value = np.median(ert_dobs)
#    std_deviation = np.std(ert_dobs)
#    variance = np.var(ert_dobs)
#
#    # Stampare i risultati
#    print(f"Number of data points: {num_data}")
#    print(f"Minimum value: {min_value}")
#    print(f"Maximum value: {max_value}")
#    print(f"Mean value: {mean_value}")
#    print(f"Median value: {median_value}")
#    print(f"Standard deviation: {std_deviation}")
#    print(f"Variance: {variance}")
#
#
#
#if OHM:
#    # Calcolare le statistiche richieste
#    num_data = len(ohm_dobs)
#    min_value = np.min(ohm_dobs)
#    max_value = np.max(ohm_dobs)
#    mean_value = np.mean(ohm_dobs)
#    median_value = np.median(ohm_dobs)
#    std_deviation = np.std(ohm_dobs)
#    variance = np.var(ohm_dobs)
#
#    # Stampare i risultati
#    print(f"Number of data points: {num_data}")
#    print(f"Minimum value: {min_value}")
#    print(f"Maximum value: {max_value}")
#    print(f"Mean value: {mean_value}")
#    print(f"Median value: {median_value}")
#    print(f"Standard deviation: {std_deviation}")
#    print(f"Variance: {variance}")
#
#    # Creare la figura e l'istogramma per V/I
#    fig = plt.figure(figsize=(10, 5))  # Imposta la dimensione del grafico
#    # bins = np.arange(0, 0.05, 0.001)
#    plt.hist(ohm_dobs, bins=30, color='b', edgecolor='k', alpha=0.7)
#    # plt.xscale('log')
#    plt.yscale('log')
#    # Aggiungere la media
#    #plt.axvline(mean_value, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
#    # Aggiungere la mediana
#    #plt.axvline(median_value, color='g', linestyle='-', linewidth=2, label=f'Median: {median_value:.2f}')
#    # Aggiungere le bande di ±2 deviazioni standard
#    #plt.axvline(mean_value - 2 * std_deviation, color='m', linestyle='--', linewidth=2,
#                #label=f'-2 Std Dev: {mean_value - 2 * std_deviation:.2f}')
#    #plt.axvline(mean_value + 2 * std_deviation, color='m', linestyle='--', linewidth=2,
#                #label=f'+2 Std Dev: {mean_value + 2 * std_deviation:.2f}')
#    # Aggiungere il titolo e le etichette degli assi
#    plt.title('Distribution log (V/I)')  # Titolo del grafico
#    plt.xlabel('log (V/I)')  # Etichetta dell'asse X
#    plt.ylabel('log(Frequency)')  # Etichetta dell'asse Y
#    # plt.xlim(0, 0.05)
#    plt.grid(True)
#    #plt.legend()
#    plt.savefig(r'C:\Users\lopan\PycharmProjects\STELLA\Res\OHM\VI_ohm_mapper.png', dpi=300, bbox_inches='tight')
#    plt.close(fig)  # Chiude la figura per liberare memoria
################################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here we define the conductivity model that will be used to predict DC
# resistivity data. The model consists of a conductive sphere and a
# resistive sphere within a moderately conductive background. Note that
# you can carry through this work flow with a resistivity model if desired.
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

#######################
# inversion

if ERT and OHM:

    inv_problem = InvProbJoin(
        mesh, ert_survey, ohm_survey, conductivity_map, ert_data_object, ohm_data_object,
        ind_active, alpha, x, maxIter=14, maxIterLS=30, maxIterCG=35, tolCG=1e-2, tolF=1e-3, tolX=1e-2, eps=1e-2,
        tolG=1e-3
    )

elif ERT:

    inv_problem = InvProb(
        mesh, ert_survey, conductivity_map, ert_data_object, ind_active, x,
        maxIter=12, maxIterLS=30, maxIterCG=35, tolCG=1e-1, tolF=1e-2, tolX=1e-2, eps=1e-2, tolG=1e-1
    )

elif OHM:

    inv_problem = InvProb(
        mesh, ohm_survey, conductivity_map, ohm_data_object, ind_active, x,
        maxIter=13, maxIterLS=30, maxIterCG=35, tolCG=1e-1, tolF=1e-3, tolX=1e-3, eps=1e-3, tolG=1e-1
    )

dir_prob = DirProb(beta0_ratio=200, coolingFactor=4.5, coolingRate=3, chifact=10)

#####################################################################
# Running the Inversion
# ---------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_problem, dir_prob)

# Run the inversion
t = time.time()
recovered_model = inv.run(x)
elapsed_time = time.time() - t
print(elapsed_time)
cond_mod = np.exp(recovered_model)
res_mod = (1 / cond_mod)


#if ERT and OHM:
#    filename = 'E:/Nicola_Lopane/Codici/STELLA/JOINT/joint_20240722-0941.txt'
#    res_mod = np.loadtxt(filename)
#
#if ERT and not OHM:
#    filename = 'C:/Users/lopan/PycharmProjects/STELLA/FW_model/ert_inv.txt'
#    res_mod = np.loadtxt(filename)
#
#if OHM and not ERT:
#    filename = 'ohm_20240722-1034.txt'
#    res_mod = np.loadtxt(filename)


#Salva i risultati monitorati
for directive in dir_prob:
    if isinstance(directive, MonitorInversion):
        directive.save_results('inversion_monitor.txt')

if OHM and ERT:
    with open(out_fold + '/' + time.strftime('joint_%Y%m%d-%H%M') + '.txt', 'w') as fid:
        for i in np.arange(len(res_mod)):
            fid.write(str(res_mod[i]) + '\n')
elif OHM:
    with open(out_fold + '/' + time.strftime('ohm_%Y%m%d-%H%M') + '.txt', 'w') as fid:
        for i in np.arange(len(res_mod)):
            fid.write(str(res_mod[i]) + '\n')
elif ERT:
    with open(out_fold + '/' + time.strftime('ert_%Y%m%d-%H%M') + '.txt', 'w') as fid:
        fid.write(str(elapsed_time) + '\n')
        for i in np.arange(len(res_mod)):
            fid.write(str(res_mod[i]) + '\n')



