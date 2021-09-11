# File for solving the inverse problem for poisson PDE
# Domain x \in (0, L)
# PDE:
#   -k \Laplacian u = m(x) 
#              u(0) = 0
#              u(L) = 0
# Data: u(x) + noise
# Paramter: m(x)

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sp
import scipy.sparse.linalg as la 

SMALL_FONT = 16
MEDIUM_FONT = 20
LARGE_FONT = 24
FIG_SIZE = (12,8)
FIG_SIZE_SMALL = (6,6)
PATH = ''

plt.rc('font', size=SMALL_FONT)
plt.rc('axes', titlesize=SMALL_FONT)
plt.rc('axes', labelsize=SMALL_FONT)
plt.rc('xtick', labelsize=SMALL_FONT)
plt.rc('ytick', labelsize=SMALL_FONT)
plt.rc('legend', fontsize=SMALL_FONT)
plt.rc('figure', titlesize=MEDIUM_FONT)

##############################################################################

def mTrueFun(x):
    """ plot returns the true m evaluated at x 
    where m is 1D function """
    return np.maximum(0, 1 - np.abs(1 - 4*x)) + 100 * x**10 * (1-x)**2

def assembleMatrix(k, h, n_dof):
    """ assembles the stiffness matrix K based on central differences """ 

    diagonals = np.zeros((3, n_dof))
    diagonals[0,:] = -1.0
    diagonals[1,:] = 2.0
    diagonals[2,:] = -1.0

    K = k/(h**2) * sp.spdiags(diagonals, [-1, 0, 1], n_dof, n_dof)
    K = K.tocsr()
    return K 

def generateData(u_true, sigma):
    """ generates data by adding to the true solution gaussian white noise
    with standard dev = sigma """ 

    return u_true + np.random.randn(u_true.shape[0])*sigma

def addEndPoints(arr, arr_end=[0,0]):
    """ adds endpoints to the array """
    arr_new = np.zeros(arr.shape[0]+2)
    arr_new[1:-1] = arr 
    arr_new[0] = arr_end[0]
    arr_new[-1] = arr_end[1]
    return arr_new

def solveFwd(K,f):
    """ solves the forward problem given formed K and f=rhs 
    i.e. Ku=f"""

    return la.spsolve(K,f)


def applyInvF(K, d):
    """ apply the inverse of the P2O operator to obtain inversion results 
    noting that F = K-1 """

    return K.dot(d)

def assembleF(K, n_dof):
    """ assembles the matrix of parameter to observable map
    by solving the forward problem for the basis"""

    F = np.zeros((n_dof, n_dof))
    m_i = np.zeros(n_dof) # basis vectors 

    for i in range(n_dof):
        m_i[i] = 1.0
        F[:,i] = solveFwd(K, m_i)
        m_i[i] = 0

    return F


def solveTikhonov(d, F, alpha):
    """ assuming R = Id """

    H = np.dot(F.transpose(), F) + alpha * np.identity(F.shape[1])
    rhs = np.dot(F.transpose(), d)

    return np.linalg.solve(H, rhs)



FIG_SIZE = (12,8)
np.random.seed(100)

k = 1
L = 1 
nx = 200 # number of elements
h = L/nx

x = np.linspace(0, L, nx+1)
m_true = mTrueFun(x)

K = assembleMatrix(k,h,nx-1)
F = assembleF(K, nx-1)

x_interior = x[1:-1]
f_true = m_true[1:-1] # interior nodes of the force 

# solving the system for true m 
u_true = solveFwd(K, f_true)

# converting to plotting by adding endpoints
u_plot = np.zeros(x.shape)
u_plot[1:-1] = u_true

# generate data
noise_variance = 0
noise_sigma = np.sqrt(noise_variance)
noise_sigma = 1e-2
d_plot = generateData(u_plot, noise_sigma)
d = d_plot[1:-1]

plt.figure()
plt.plot(x, m_true, 'k')
plt.grid(True)
plt.title("$m_{true}$")

plt.figure()
plt.plot(x, u_plot, '-k')
plt.plot(x, d_plot, 'ob')
plt.grid(True)
plt.title("Synthetic observations")
plt.legend(["$u_{true}$", "Noisy data"])

plt.show()

##############################################################################

alphas = np.array([0.0001, 0.001, 0.01, 0.1, 1])
colors = ['b', 'r', 'g', 'c', 'm']

legend_reg = ["$\\alpha = %f$" %(alpha) for alpha in alphas]
legend = ["Truth"] + legend_reg 

plt.figure(figsize=FIG_SIZE)
plt.plot(x, m_true, '-k')

for (alpha, c) in zip(alphas, colors):
    m_reg = solveTikhonov(d, F, alpha)
    plt.plot(x_interior, m_reg, color = c)
plt.legend(legend, loc="lower right")
plt.title("Inversion results with regularization")
plt.axis([0, 1, -1.0, 1.2])
plt.grid(True)
plt.savefig(PATH + "figures/p2_part_d_regularization_%.0e.png" %(alphas[0]))
plt.show()

##############################################################################

# L-curve vs morozov discrepancy 
# also compare with exact solution 

delta = np.linalg.norm(d - u_true)

norm_m = np.zeros(alphas.shape)
norm_residual = np.zeros(alphas.shape)
norm_diff_true = np.zeros(alphas.shape)


for ii in range(alphas.size):
    m_reg = solveTikhonov(d, F, alphas[ii])
    norm_m[ii] = np.linalg.norm(m_reg)

    u_pred = solveFwd(K, m_reg)
    norm_residual[ii] = np.linalg.norm(u_pred - d)

    norm_diff_true[ii] = np.linalg.norm(m_reg - f_true)

plt.figure(figsize=FIG_SIZE)
plt.loglog(norm_residual, norm_m, '-ob')
plt.grid(True, which="both")
plt.xlabel("$\|Fm-d\|$")
plt.ylabel("$\|m\|$")
plt.title("L-curve")
plt.savefig(PATH + "figures/p2_part_e_lCurve_%.0e.png" %(alphas[0]))

delta_plot = np.ones(alphas.shape) * delta

plt.figure(figsize=FIG_SIZE)
plt.loglog(alphas, norm_residual, '-ob')
plt.loglog(alphas, delta_plot, '--k')
plt.grid(True, which="both")
plt.xlabel("$\\alpha$")
plt.ylabel("$\|Fm-d\|$")
plt.legend(["Regularized solution", "$\delta$"])
plt.title("Morozov discrepancy principle")
plt.savefig(PATH + "figures/p2_part_f_morozov_%.0e.png" %(alphas[0]))
plt.show()

plt.figure(figsize=FIG_SIZE)
plt.loglog(alphas, norm_diff_true, '-ob')
plt.grid(True, which="both")
plt.xlabel("$\\alpha$")
plt.ylabel("$\|m-m_{true}\|$")
plt.title("Comparison with $m_{true}$")
plt.savefig(PATH + "figures/p2_part_g_bestAlpha_%.0e.png" %(alphas[0]))
plt.show()

##############################################################################



