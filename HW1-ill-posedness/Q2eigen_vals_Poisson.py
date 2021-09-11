#Decay of Eigenvalues in 1D Poisson equation
#Mohammad Afzal Shadab
#Date modified: 09/04/21


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm         #color map


eigen_cont = lambda k,L,i: 1/(k*(i*np.pi/L)**2) #eigen value of continuous operator
eigen_disc = lambda k,h,i,nx: 1/(k*4/h**2*(np.sin(np.pi*i/(2*nx)))**2) #eigen value of discrete operator

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


#PART (b)
#Parameters
L = 1 #Length of the domain [0,1]
nx= 100#Number of cellls in x

i = np.linspace(1,nx - 1,nx - 1) #Modes
h = L/nx #Cell sixe
k_array = [1]


plt.figure(figsize=(15,10))
for count, k in enumerate(k_array):
    plot = plt.semilogy(i,eigen_cont(k,L,i),c=color_array[count],label=r'$\mathcal{F}, k=%0.4f$'%k)
    plot = plt.semilogy(i,eigen_disc(k,h,i,nx),'ro',c=color_array[count],label=r'$ {\bf{F}}, k=%0.4f$'%k)

plt.legend(loc='upper left')
plt.xlabel(r'$i$')
plt.ylabel(r'$\lambda_i$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"2b_discvscont_evals.png")

