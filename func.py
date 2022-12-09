import numpy as np
import math
from numpy import linalg as la
from numba import njit

def Hr_3D(j,mi_0,imas,dom):  # Campo H na direção X para 1 ímã
  CTE = j/(4*np.pi*mi_0)
  Sum_x = 0.0
  Sum_y = 0.0
  Sum_z = 0.0
  L_x = dom[0][0]
  L_y = dom[0][1]
  L_z = dom[0][2]
  n_x = int(dom[1][0])
  n_y = int(dom[1][1])
  n_z = int(dom[1][2])
  delta_x = L_x/(n_x-1)
  delta_y = L_y/(n_y-1)
  delta_z = L_z/(n_z-1)
  x_0 = - delta_x
  y_0 = - delta_y
  z_0 = - delta_z

  H = []
  for i in range(n_z):
    z_0 = z_0 + delta_z
    for j in range(n_y):
      y_0 = y_0 + delta_y
      for k in range(n_x):
        x_0 = x_0 + delta_x
        for l in range(len(imas)):
          a = imas[l][0]
          b = imas[l][1]
          c = imas[l][2]
          x = -imas[l][3] + x_0
          y = -imas[l][4] + y_0
          z = -imas[l][5] + z_0
          H_x = CTE*np.log(((y+b+np.sqrt((y+b)**2+(x-a)**2+z**2))/\
                          (y-b+np.sqrt((y-b)**2+(x-a)**2+z**2)))\
                          *((y-b+np.sqrt((y-b)**2+(x+a)**2+z**2))\
                          /(y+b+np.sqrt((y+b)**2+(x+a)**2+z**2))))\
              -CTE*np.log(((y+b+np.sqrt((y+b)**2+(x-a)**2+(z+c)**2))/\
                          (y-b+np.sqrt((y-b)**2+(x-a)**2+(z+c)**2)))\
                          *((y-b+np.sqrt((y-b)**2+(x+a)**2+(z+c)**2))\
                          /(y+b+np.sqrt((y+b)**2+(x+a)**2+(z+c)**2))))
              
          H_y = CTE*np.log(((x+a+np.sqrt((y-b)**2+(x+a)**2+z**2))/\
                        (x-a+np.sqrt((y-b)**2+(x-a)**2+z**2)))\
                        *((x-a+np.sqrt((y+b)**2+(x-a)**2+z**2))\
                        /(x+a+np.sqrt((y+b)**2+(x+a)**2+z**2))))\
            -CTE*np.log(((x+a+np.sqrt((y-b)**2+(x+a)**2+(z+c)**2))/\
                        (x-a+np.sqrt((y-b)**2+(x-a)**2+(z+c)**2)))\
                        *((x-a+np.sqrt((y+b)**2+(x-a)**2+(z+c)**2))\
                        /(x+a+np.sqrt((y+b)**2+(x+a)**2+(z+c)**2))))
            
          H_z = CTE*(np.arctan(((x+a)*(y+b))/(z*np.sqrt((x+a)**2+(y+b)**2+z**2)))\
                  +np.arctan(((x-a)*(y-b))/(z*np.sqrt((x-a)**2+(y-b)**2+z**2)))\
                  -np.arctan(((x+a)*(y-b))/(z*np.sqrt((x+a)**2+(y-b)**2+z**2)))\
                  -np.arctan(((x-a)*(y+b))/(z*np.sqrt((x-a)**2+(y+b)**2+z**2))))\
            -CTE*(np.arctan(((x+a)*(y+b))/((z+c)*np.sqrt((x+a)**2+(y+b)**2+(z+c)**2)))\
                  +np.arctan(((x-a)*(y-b))/((z+c)*np.sqrt((x-a)**2+(y-b)**2+(z+c)**2)))\
                  -np.arctan(((x+a)*(y-b))/((z+c)*np.sqrt((x+a)**2+(y-b)**2+(z+c)**2)))\
                  -np.arctan(((x-a)*(y+b))/((z+c)*np.sqrt((x-a)**2+(y+b)**2+(z+c)**2))))
            
          Sum_x = Sum_x + H_x
          Sum_y = Sum_y + H_y
          Sum_z = Sum_z + H_z 
          res = ((Sum_x)**2+(Sum_y)**2+(Sum_z)**2)**(0.5)
        Sum = np.array(([res,Sum_x,Sum_y,Sum_z,x_0,y_0,z_0]))
        H.append(Sum)
        Sum_x = 0.0
        Sum_y = 0.0
        Sum_z = 0.0   
      x_0 = 0.0
    y_0 = 0.0

  return H

def Hr_1D(j,mi_0,imas,dom):  # Campo H na direção X para 1 ímã
  CTE = j/(4*np.pi*mi_0)
  Sum_x = 0.0
  Sum_y = 0.0
  Sum_z = 0.0
  L_z = dom[0][2]
  n_z = int(dom[1][2])
  delta_z = L_z/(n_z-1)
  z_0 = - delta_z

  H = []
  for i in range(n_z):
    z_0 = z_0 + delta_z
    for l in range(len(imas)):
      a = imas[l][0]
      b = imas[l][1]
      c = imas[l][2]
      x = -imas[0][3] 
      y = -imas[0][4] 
      z = -imas[l][5] + z_0
      H_x = CTE*np.log(((y+b+np.sqrt((y+b)**2+(x-a)**2+z**2))/\
                      (y-b+np.sqrt((y-b)**2+(x-a)**2+z**2)))\
                      *((y-b+np.sqrt((y-b)**2+(x+a)**2+z**2))\
                      /(y+b+np.sqrt((y+b)**2+(x+a)**2+z**2))))\
          -CTE*np.log(((y+b+np.sqrt((y+b)**2+(x-a)**2+(z+c)**2))/\
                      (y-b+np.sqrt((y-b)**2+(x-a)**2+(z+c)**2)))\
                      *((y-b+np.sqrt((y-b)**2+(x+a)**2+(z+c)**2))\
                      /(y+b+np.sqrt((y+b)**2+(x+a)**2+(z+c)**2))))
          
      H_y = CTE*np.log(((x+a+np.sqrt((y-b)**2+(x+a)**2+z**2))/\
                    (x-a+np.sqrt((y-b)**2+(x-a)**2+z**2)))\
                    *((x-a+np.sqrt((y+b)**2+(x-a)**2+z**2))\
                    /(x+a+np.sqrt((y+b)**2+(x+a)**2+z**2))))\
        -CTE*np.log(((x+a+np.sqrt((y-b)**2+(x+a)**2+(z+c)**2))/\
                    (x-a+np.sqrt((y-b)**2+(x-a)**2+(z+c)**2)))\
                    *((x-a+np.sqrt((y+b)**2+(x-a)**2+(z+c)**2))\
                    /(x+a+np.sqrt((y+b)**2+(x+a)**2+(z+c)**2))))
        
      H_z = CTE*(np.arctan(((x+a)*(y+b))/(z*np.sqrt((x+a)**2+(y+b)**2+z**2)))\
              +np.arctan(((x-a)*(y-b))/(z*np.sqrt((x-a)**2+(y-b)**2+z**2)))\
              -np.arctan(((x+a)*(y-b))/(z*np.sqrt((x+a)**2+(y-b)**2+z**2)))\
              -np.arctan(((x-a)*(y+b))/(z*np.sqrt((x-a)**2+(y+b)**2+z**2))))\
        -CTE*(np.arctan(((x+a)*(y+b))/((z+c)*np.sqrt((x+a)**2+(y+b)**2+(z+c)**2)))\
              +np.arctan(((x-a)*(y-b))/((z+c)*np.sqrt((x-a)**2+(y-b)**2+(z+c)**2)))\
              -np.arctan(((x+a)*(y-b))/((z+c)*np.sqrt((x+a)**2+(y-b)**2+(z+c)**2)))\
              -np.arctan(((x-a)*(y+b))/((z+c)*np.sqrt((x-a)**2+(y+b)**2+(z+c)**2))))
        
      Sum_x = Sum_x + H_x
      Sum_y = Sum_y + H_y
      Sum_z = Sum_z + H_z 
      res = ((Sum_x)**2+(Sum_y)**2+(Sum_z)**2)**(0.5)
    Sum = np.array(([res,Sum_x,Sum_y,Sum_z,x,y,z_0]))
    H.append(Sum)
    Sum_x = 0.0
    Sum_y = 0.0
    Sum_z = 0.0   


  return H

def temp_1d(t1,t2,pf,x,n,l_x):
  t = np.zeros(n)
  for i in range(n):
    if pf==1:
      t[i] = t1 +((t2-t1)/l_x)*x[i]
    if pf==2 and t2 == 400:
      t[i] = t1 + (3012*x[i]) -(2594*(x[i]**2)) + (873961*(x[i]**3)) -(1.43e+07*(x[i]**4))
    if pf==2 and t2 == 500:
      t[i] = t1 + (6093*x[i]) -(21730*(x[i]**2)) + (2.43e+06*(x[i]**3)) -(3.53e+07*(x[i]**4))
    if pf==2 and t2 == 600:
      t[i] = t1 + (9139*x[i]) + (68068*(x[i]**2)) -(1.14e+07*(x[i]**3)) + (7.31e+08*(x[i]**4)) - (1.72e+10*(x[i]**5)) + (1.34e+11*(x[i]**6))
    if pf==2 and t2 == 700:
      t[i] = t1 + (12658*x[i]) + (158808*(x[i]**2)) -(2.99e+07*(x[i]**3)) + (1.76e+09*(x[i]**4)) - (3.94e+10*(x[i]**5)) + (2.97e+11*(x[i]**6))
    if pf==2 and t2 == 800:
      t[i] = t1 + (16362*x[i]) + (277523*(x[i]**2)) -(5.46e+07*(x[i]**3)) + (3.13e+09*(x[i]**4)) - (6.9e+10*(x[i]**5)) + (5.14e+11*(x[i]**6))
    if pf==2 and t2 == 900:
      t[i] = t1 + (20211*x[i]) + (404420*(x[i]**2)) -(8.26e+07*(x[i]**3)) + (4.7e+09*(x[i]**4)) - (1.03e+11*(x[i]**5)) + (7.67e+11*(x[i]**6))
    if pf==2 and t2 == 1000:
      t[i] = t1 + (25067*x[i]) + (95744*(x[i]**2)) -(2.5e+07*(x[i]**3)) -(2.49e+09*(x[i]**4)) + (3.54e+11*(x[i]**5)) -(1.41e+13*(x[i]**6)) + (2.38e+14*(x[i]**7)) -(1.45e+15*(x[i]**8))
    if pf==2 and t2 == 1100:
      t[i] = t1 + (29276*x[i]) + (127388*(x[i]**2)) -(3.69e+07*(x[i]**3)) -(2.63e+09*(x[i]**4)) + (4.21e+11*(x[i]**5)) -(1.7e+13*(x[i]**6)) + (2.86e+14*(x[i]**7)) -(1.73e+15*(x[i]**8))
    if pf==2 and t2 == 1200:
      t[i] = t1 + (33522*x[i]) + (161617*(x[i]**2)) -(5.06e+07*(x[i]**3)) -(2.61e+09*(x[i]**4)) + (4.81e+11*(x[i]**5)) -(1.98e+13*(x[i]**6)) + (3.33e+14*(x[i]**7)) -(2.00e+15*(x[i]**8))
    if pf==2 and t2 == 1300:
      t[i] = t1 + (37796*x[i]) + (199682*(x[i]**2)) -(6.62e+07*(x[i]**3)) -(2.38e+09*(x[i]**4)) + (5.3e+11*(x[i]**5)) -(2.23e+13*(x[i]**6)) + (3.76e+14*(x[i]**7)) -(2.24e+15*(x[i]**8))
  
  return t

def xi_1d(t,n,md,mi0,v,phi_p,kb):
  xi = np.zeros(n)
  cte = (mi0*(md**2)*phi_p*v)/(3*kb)
  for i in range(n):
    xi[i] = cte/t[i]

  return xi


def Gauss(A, b):
    """
    Gaussian Elimination with Backward Substitution
    """
    lines, columns = A.shape
    n = lines

    M = np.zeros((n, n + 1))
    M[:, :-1] = A
    M[:, -1] = b

    x = np.zeros(n)
    for i in range(n - 1):
        p = i
        while p < n - 1 and M[p][i] == 0:
            p += 1
        if p != i:
            M[i], M[p] = M[p], M[i]  # swap lines
        if p == n - 1:
            # print("Solution doesn't exist - 1")
            return None
        for j in range(i + 1, n):
            m = M[j][i] / M[i][i]
            M[j] = M[j] - m * M[i]
            # print(M)
    if M[n - 1][n - 1] == 0:
        # print("Solution doesn't exist - 2")
        return None
    x[-1] = M[-1][-1] / M[-1][-2]
    for i in range(n - 2, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += M[i][j] * x[j]
        x[i] = (M[i][-1] - soma) / M[i][i]
    return x

def Partial(A, b):
    """
    Gaussian Elimination with Partial Pivoting
    """
    lines, columns = A.shape
    n = lines

    M = np.zeros((n, n + 1))
    M[:, :-1] = A
    M[:, -1] = b

    x = np.zeros(n)
    nrow = np.zeros(n, dtype=type(int))
    for i in range(n):
        nrow[i] = i
    for i in range(n - 1):
        maxim = 0
        p = i
        for j in range(i, n):
            if np.abs(M[nrow[j]][i]) > maxim:
                maxim = np.abs(M[nrow[j]][i])
                p = j
        if M[nrow[p]][i] == 0:
            # print("Solution doesn't exist - 1")
            return None
        if nrow[i] != nrow[p]:
            nrow[i], nrow[p] = nrow[p], nrow[i]  # M troca de linhas
        for j in range(i + 1, n):
            m = M[nrow[j]][i] / M[nrow[i]][i]
            M[nrow[j]] = M[nrow[j]] - m * M[nrow[i]]
            # print(M)
    if M[nrow[-1]][n] == 0:
        # print("Solution doesn't exist - 2")
        return None
    x[-1] = M[nrow[-1]][-1] / M[nrow[-1]][-2]
    for i in range(n - 2, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += M[nrow[i]][j] * x[j]
        x[i] = (M[nrow[i]][-1] - soma) / M[nrow[i]][i]
    return x


def Scaled(A, b):
    """
    Gaussian Elimination with Scaled Partial Pivoting
    """
    lines, columns = A.shape
    n = lines

    M = np.zeros((n, n + 1))
    M[:, :-1] = A
    M[:, -1] = b

    x = np.zeros(n)
    nrow = np.zeros(n, dtype=type(int))

    # Only this difference between this algorithm and the precedent
    s = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if s[i] < np.abs(M[i][j]):
                s[i] = np.abs(M[i][j])
        if s[i] == 0:
            return None
    # Until here

    for i in range(n):
        nrow[i] = i
    for i in range(n - 1):
        maximo = 0
        p = i
        for j in range(i, n):
            if np.abs(M[nrow[j]][i]) / s[nrow[j]] > maximo:  # Alteracao desse
                maximo = np.abs(M[nrow[j]][i]) / s[nrow[j]]  # E desse
                p = j

        if M[nrow[p]][i] == 0:
            # print("Solution doesn't exist - 1")
            return None
        if nrow[i] != nrow[p]:
            nrow[i], nrow[p] = nrow[p], nrow[i]  # M troca de linhas
        for j in range(i + 1, n):
            m = M[nrow[j]][i] / M[nrow[i]][i]
            M[nrow[j]] = M[nrow[j]] - m * M[nrow[i]]
            # print(M)
    if M[nrow[-1]][n] == 0:
        # print("Solution doesn't exist - 2")
        return None
    x[-1] = M[nrow[-1]][-1] / M[nrow[-1]][-2]
    for i in range(n - 2, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += M[nrow[i]][j] * x[j]
        x[i] = (M[nrow[i]][-1] - soma) / M[nrow[i]][i]
    return x


def LUFactorization(A):
    """
    LU Factorization
    This algorithm is good if you have always the same matrix A, and with different configurations of b.
    """
    lines, columns = A.shape
    n = lines

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    if A[0, 0] == 0:
        # Impossible factorization
        return None, None
    L[0, 0] = A[0, 0]

    U[0] = A[0] / L[0, 0]
    L[:, 0] = A[:, 0] / U[0, 0]

    for i in range(1, n - 1):
        soma = 0
        for k in range(i):
            soma += L[i, k] * U[k, i]
        L[i, i] = A[i, i] - soma
        U[i, i] = 1
        if L[i, i] == 0:
            # Impossible factorization
            return None, None

        for j in range(i + 1, n):
            soma = 0
            for k in range(i):
                soma += L[i, k] * U[k, j]
            U[i, j] = (A[i, j] - soma) / L[i, i]
            soma = 0
            for k in range(i):
                soma += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - soma) / U[i, i]
    soma = 0
    for k in range(n - 1):
        soma += L[-1, k] * U[k, -1]
    L[-1, -1] = (A[-1, -1] - soma)
    U[-1, -1] = 1
    if L[-1, -1] == 0:
        # Impossible, matrix A is singular
        return None, None
    return L, U

def solvewithLU(L, U, b):

    # Now that A * x = b
    # and A = L * U
    # So, L*U*x = b is equal to
    #   L*y = b
    #   U*x = y
    # So, we find frist y and after that x
    # Now, it's solve using
    if L is None:  # If it couldn't decompose A in LU method
        return None
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        soma = 0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = (b[i] - soma) / L[i, i]

    x[-1] = y[-1] / U[-1, -1]
    for i in range(n - 1, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]

    return x

def gauss_seidel(A: np.ndarray, b: np.ndarray, tol=1e-3, max_iter=1e4):
    
    """Gauss Seidel Method for solving linear systems of equations.

    Raises:
        ValueError: A is not diagonally dominant
        ValueError: A is not a square matrix
        ValueError: A and b are not the same size
        
    Args:
        A (np.ndarray): Coefficient matrix
        b (np.ndarray): Vector of constants
        tol (float, optional): Tolerance. Defaults to 1e-10.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1e4.
    """
    
    for i in range(len(b)):
        
        if A[i][i] == 0:
            raise ValueError("A is not diagonally dominant")
        
        elif len(A) != len(A[0]):
            raise ValueError("A is not a square matrix")
        
        elif len(A) != len(b):
            raise ValueError("A and b are not the same size")
    
    N = len(b)
    x = np.zeros(N, float)
    R = np.zeros(N, float)
    
    iteration = 0
    x, R = gauss_seidel_main_loo(A, b, x, R, N, tol, max_iter)
    return([x, R])

@njit(fastmath=True)
def gauss_seidel_main_loo(A, b, x, R,  N, tol=1e-3, max_iter=1e4, iteration=0):
    while True and iteration < max_iter:
        for i in range(N):
            sumation = 0.
            
            for j in range(N):
                sumation += A[i,j] * x[j]
                
            R[i] = (1.0/A[i,i]) * (b[i] - sumation)
                
            x[i] += R[i]
        
        print(np.max(np.abs(R)), iteration)
        if np.max(np.abs(R)) < tol:
          break
            
        iteration += 1
    return([x, R])









