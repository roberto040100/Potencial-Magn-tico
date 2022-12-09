from func import temp_1d
from func import xi_1d
from func import gauss_seidel
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

# Definindo o número de nós
n_z = 200

# Definindo as dimensões do domínio e espaçamento entre nós
l_z = 0.03
delta_z = l_z/(n_z-1)

# Definindo constantes físicas

mu_0 = 1.26e-6
k_b = 1.380648e-23

# Definindo as condições de contorno do problema

H1 = 2430.229
H2 = 1722.452
perfil_temp = 1  #1 para perfil de Temperatura linear e 2 para perfil gerado pelo OpenFOAM
T1 = 300.0
T2 = 1300.0

# Definindo as propriedades do ferrofluido

raio = 1.0e-08/2
vp = (4.0/3.0)*np.pi*(raio**3.0)
Md = 5.18e+05
phi = 0.1

# Criando vetor posição

z = np.linspace(0.0,l_z,n_z)

# Criando vetor temperatura

T = temp_1d(T1,T2,perfil_temp,z,n_z,l_z)

# Criando vetor susceptibilidade

Xi = xi_1d(T,n_z,Md,mu_0,vp,phi,k_b)

# Criando matriz A e vetor B

A = np.identity(n_z)
B = np.zeros(n_z)
c = np.zeros(n_z)

for i in range(n_z):

  if i>=1 and i<=(n_z-2):
    c[i] = (Xi[i+1] - Xi[i-1])/(4*(1+Xi[i])) 
    A[i][i+1] = (c[i]-1)/2
    A[i][i-1] = - (c[i]+1)/2

    
  if i==0:
    A[i][i+1] = -1
    B[i]= -H1*(delta_z)

  if i==(n_z-1):
    A[i][i-1] = - 1
    B[i]= H2*(delta_z)

X2 = gauss_seidel(A,B,1e-6,1e6)

r = np.allclose(np.dot(A, X2[0]), B)


H = np.zeros(n_z)
H[0]=H1
H[n_z-1]=H2

for k in range(n_z):

  if k>=1 and k<=(n_z-2):

    H[k]= (X2[0][k+1]-X2[0][k-1])/(2*delta_z)

fig = plt.figure('sucetibilidade', [10,10])
ax = plt.axes()
ax.plot(z,X2[0])
ax.set_title("Susceptibilidade")
plt.savefig('sucetibilidade.png')
