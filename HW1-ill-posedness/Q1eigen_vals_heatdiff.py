#Decay of Eigenvalues in 1D heat diffusion
#Mohammad Afzal Shadab
#Date modified: 09/04/21

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm         #color map
plt.rcParams.update({'font.family': "Serif"})
plt.rcParams.update({'font.size': 22})


eigen_cont = lambda k,T,L,i: np.exp(-k*T*(np.pi/L)**2*i**2) #eigen value of continuous operator
eigen_disc = lambda dt,k,h,i,nx,nt: (1 + dt*k*4/h**2*(np.sin(np.pi*i/(2*nx)))**2)**(-nt) #eigen value of discrete operator

#Colors
brown  = [181/255 , 101/255, 29/255]
red    = [255/255 ,0/255 ,0/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]

color_array = [red,green,orange,brown,blue,purple]


#PART (c)
#Parameters
T = 1 #Final time
L = 1 #Length of the domain [0,1]
nx= 100#Number of cellls in x
nt= 100#Number of cells in time

i = np.linspace(1,nx - 1,nx - 1) #Modes
h = L/nx #Cell sixe
dt= T/nt #Time step
k_array = np.logspace(-4,0,5)


plt.figure(figsize=(12,10))
for count, k in enumerate(k_array):
    plot = plt.semilogy(i,eigen_cont(k,T,L,i),c=color_array[count],label=r'$k=%0.4f$'%k)

plt.legend(loc='best')
plt.ylim([1e-16,10])
plt.xlabel(r'$i$')
plt.ylabel(r'$\lambda_i$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"1c_discvscont_evals.png")


#PART (d)
#Parameters
T = 0.1 #Final time
L = 1 #Length of the domain [0,1]
k = 0.01#Diffusion coefficient

nxnt = [160,80,40,20]

plt.figure(figsize=(12,10))
for count, nx in enumerate(nxnt):
    i = np.linspace(1,nx- 1,nx - 1) #Modes
    nt = nx
    h = L/nx #Cell sixe
    dt= T/nt #Time step
    plot = plt.semilogy(i,eigen_disc(dt,k,h,i,nx,nt),'ro',c=color_array[count],label=f'({nx},{nt})')

i = np.linspace(1,160 - 1,160 - 1) #Modes

plot = plt.semilogy(i,eigen_cont(k,T,L,i),'k--',label=f'Analytical')

plt.legend(loc='lower left')
plt.xlabel(r'$i$')
plt.xlim([0,80])
plt.ylim([1e-16,10])
plt.ylabel(r'$\lambda_i$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"1d_discvscont_eval_discretization.png")
