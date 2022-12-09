from func import temp_1d
from func import xi_1d
from func import gauss_seidel
import matplotlib.pyplot as plt
import numpy as np

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
d = np.zeros(n_z)
u_mais = np.zeros(n_z) 
u_menos = np.zeros(n_z)

for i in range(n_z):

  if i>=2 and i<=(n_z-3):

    c[i] = 4*(1+Xi[i]) 
    u_mais[i] = (3*Xi[i]-4*Xi[i-1]+Xi[i-2]) 
    u_menos[i] = (-3*Xi[i]+4*Xi[i+1]-Xi[i+2]) 

    if u_mais[i]<0 and u_menos[i]<0: 

      d[i] = u_menos[i]/c[i]
      A[i][i-1] = 1/(3*d[i]-2)
      A[i][i+1] = -(4*d[i]-1)/(3*d[i]-2)
      A[i][i+2] = d[i]/(3*d[i]-2)

    if u_mais[i]>0 and u_menos[i]>0:

      d[i] = u_mais[i]/c[i]
      A[i][i+1] = 1/(3*d[i]+2)
      A[i][i-1] = -(4*d[i]+1)/(3*d[i]+2)
      A[i][i-2] = d[i]/(3*d[i]+2)
  
  if i==0:
    A[i][i+1] = - 4/3
    A[i][i+2] = 1/3
    B[i]= -(H1*2*delta_z)/3

  if i==(n_z-1):
    A[i][i-1] = - 4/3
    A[i][i-2] = 1/3
    B[i]= (H2*2*delta_z)/3

  if i==1:
    A[i][i+1] = - 4/3
    A[i][i+2] = 1/3
    B[i]= -(H1*2*delta_z)/3

  if i==(n_z-2):
    A[i][i-1] = - 4/3
    A[i][i-2] = 1/3
    B[i]= (H2*2*delta_z)/3

X1 = gauss_seidel(A,B)

r = np.allclose(np.dot(A, X1[0]), B)
print(r)

H = np.zeros(n_z)
H[0]=H1
H[1]=H1
H[n_z-1]=H2
H[n_z-2]=H2

for k in range(n_z):

  if k>=2 and k<=(n_z-3):

    H[k]=((-3*X1[0][k])+(4*X1[0][k+1])-X1[0][k+2])/(2*delta_z)



plt.plot(z,X1[0])
plt.title("Susceptibilidade")
plt.show()

plt.plot(z,H)
plt.title("Campo H")
plt.show()