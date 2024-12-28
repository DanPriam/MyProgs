# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:49:10 2023

@author: Даниил
"""

from numpy import zeros, linspace, sin, pi, complex64, real, empty
from matplotlib.pyplot import style, figure, axes
from matplotlib import pyplot as plt
from celluloid import Camera
from  numpy import savez
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv



 # Набор команд, за счёт которых анимация строится в отдельном окне
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
def kronecker(a,b):
    s = 0
    if (a==b):
        s = 1
    return s
def u_init(x):
    u_init = sin(pi*x/2)
    return u_init
'''def f_obs(x):
    f_obs = sin(5*pi*x/2)
    return f_obs
 '''
 
N = 200
f_obs = zeros(N+1) 
q = zeros(N+1)
#Правая часть
def f(y,h,eps,N):
    f = zeros(N-1)
    f[0] = -((eps/(h**2))*(y[1]-2*y[0])+y[0]*y[1]/(2*h) - y[0]*q[1]) 
    #Здесь можно добавить зависящее от времени u_left и u_right
    f[N-2] = -((eps/(h**2))*(y[N-3]-2*y[N-2]+1)+(1-y[N-3])*y[N-2]/(2*h) - y[N-2]*q[N-1])
    for n in range(1,N-2):
        f[n] =  -((eps/(h**2))*(y[n+1]-2*y[n]+y[n-1])+y[n]*(y[n+1]-y[n-1])/(2*h) - y[n]*q[n+1])
    return f
def f2(y,h,eps,N):
    f2 = zeros(N-1)
    f2[0] = -(-(eps/(h**2))*(y[1]-2*y[0])+y[0]*y[1]/(2*h) + y[0]*q[1]) 
    f2[N-2] = -(-(eps/(h**2))*(y[N-3]-2*y[N-2]) - y[N-3]*y[N-2]/(2*h) + y[N-2]*q[N-1])
    for n in range(1,N-2):
        f2[n] =  -(-(eps/(h**2))*(y[n+1]-2*y[n]+y[n-1])+y[n]*(y[n+1]-y[n-1])/(2*h) + y[n]*q[n+1])
    return f2
#Матрица Fu; Затем a, b ,c и алгоритм прогонки 
def Fu(y, h, eps, N):
    Fu = zeros((N-1,N-1))
    for n in range(N-1):
        Fu[0,n] = kronecker(n,0)*(-2*(eps/(h**2))+y[1]/(2*h) - q[1]) + kronecker(n,1)*((eps/(h**2))+y[0]/(2*h))
        Fu[N-2,n] = kronecker(N-2,n)*(-2*(eps/(h**2))+(1-y[N-3])/(2*h) - q[N-1]) + kronecker(N-3,n)*((eps/(h**2))-y[N-2])
    for k in range(1,N-2):
        for n in range(N-1):
            Fu[k,n] = kronecker(k-1,n)*((eps/(h**2)) - y[k] / (2*h)) + kronecker(k,n)*(-2*(eps/(h**2)) + (y[k+1] - y[k-1])/(2*h) - q[k+1]) + kronecker(k + 1,n)*((eps/(h**2)) + y[k] / (2*h))
    return Fu
def Fu2(y, h, eps, N):
    Fu2 = zeros((N-1,N-1))
    for n in range(N-1):
        Fu2[0,n] = kronecker(n,0)*(2*(eps/(h**2))+y[1]/(2*h) + q[1])
        Fu2[N-2,n] = kronecker(N-2,n)*(2*(eps/(h**2))- y[N-3]/(2*h) + q[N-1])
    for k in range(1,N-2):
        for n in range(N-1):
            Fu2[k,n] = kronecker(k-1,n)*(-(eps/(h**2)) - y[k] / (2*h)) + kronecker(k,n)*(2*(eps/(h**2)) + (y[k+1] - y[k-1])/(2*h) + q[k+1]) + kronecker(k + 1,n)*(-(eps/(h**2)) + y[k] / (2*h))
    return Fu2
def DiagonalsPreparation(Fuu,h,N,tau,alpha) :
    a = zeros(N-2,dtype=complex64)
    b = zeros(N-2,dtype=complex64)
    c = zeros(N-1,dtype=complex64)
    Fu = zeros((N-1,N-1))
    Fu = Fuu
    for i in range(N-2):
        a[i] = -alpha*tau*Fu[i+1][i]
        b[i] = -alpha*tau*Fu[i][i+1]
    for i in range(N-1):
        c[i] = -(1 - alpha*tau*Fu[i][i])
    return a,b,c
def DiagonalsPreparation2(Fuu,h,N,tau,alpha) :
    a = zeros(N-2,dtype=complex64)
    b = zeros(N-2,dtype=complex64)
    c = zeros(N-1,dtype=complex64)
    Fu = zeros((N-1,N-1))
    Fu = Fuu
    for i in range(N-2):
        a[i] = alpha*tau*Fu[i+1][i]
        b[i] = alpha*tau*Fu[i][i+1]
    for i in range(N-1):
        c[i] = -(1 + alpha*tau*Fu[i][i])
    return a,b,c
def AlgTriDiag(a,b,c,F):
    Alph = zeros(N-2,dtype=complex64)
    Bet = zeros(N-1, dtype=complex64)
    Alph[0] = b[0]/c[0]
    Bet[0] = F[0]/c[0]
    for i in range(N-3):
        Alph[i+1] = b[i+1]/(c[i+1] - a[i]*Alph[i])
        Bet[i+1] = (F[i+1] + a[i]*Bet[i])/(c[i+1] - a[i]*Alph[i])
    Bet[N-2] = (F[N-2] + a[N-3]*Bet[N-3])/(c[N-2] - a[N-3]*Alph[N-3])
    w = zeros(N-1,dtype=complex64)
    v = zeros(N-1)
    w[N-2] = Bet[N-2]
    for ii in range(1,N-1):
        w[N-2-ii] = Alph[N-2-ii]*w[N-2-(ii-1)]+Bet[N-2-ii]
    for n in range(N-1):
        v[n] = real(w[n])
    return v
def DirectSol(a,b,N,t_0,T,M,u_init,alpha):
    h = (b - a)/N; x = linspace(a,b,N+1)
    tau = (T - t_0)/M
    u = zeros((M + 1,N + 1))
    y = zeros(N - 1)
    u[0] = u_init(x)
    y = u_init(x[1:(N)])
    for m in range(M) :
       a1,b1,c1 = DiagonalsPreparation( Fu(y, h, eps, N) ,h,N,tau,alpha)
       v = AlgTriDiag(a1,b1,c1,f(y,h, eps, N))
       y = y + tau*v
       u[m + 1, 0] = 0
       u[m + 1, N] = 1
       u[m + 1,1:N ] = y
    return u
def dfdq(dy,y,h,eps,N):
    dfdq = zeros((N-1,N-1))
    for j in range(N-1):
        dfdq[0][j] = ((eps/(h**2))*(dy[1][j]-2*dy[0][j])+dy[0][j]*y[1]/(2*h)+y[0]*dy[1][j]/(2*h) - dy[0][j]*q[1]-kronecker(0,j)*y[0]) 
        dfdq[N-2][j] = ((eps/(h**2))*(dy[N-3][j]-2*dy[N-2][j])+(1-y[N-3])*dy[N-2][j]/(2*h) - dy[N-3][j]*y[N-2]/(2*h) - dy[N-2][j]*q[N-1] - y[N-2]*kronecker(N-2,j)) 
        for n in range(1,N-2):
            dfdq[n][j] =  ((eps/(h**2))*(dy[n+1][j]-2*dy[n][j]+dy[n-1][j]) + dy[n][j]*(y[n+1]-y[n-1])/(2*h) +y[n]*(dy[n+1][j]-dy[n-1][j])/(2*h) - dy[n][j]*q[n+1] - y[n]*kronecker(n,j))
    return dfdq
def dAdq(dy,y,h,eps,N,t_0,T,M,alpha):
    tau = (T - t_0)/M
    dAdq = zeros((N-1,N-1,N-1),dtype=complex64)
    for j in range(N-1):
        for n in range(N-1):
            dAdq[0,n,j] = -(alpha*tau)*kronecker(n,0)*(dy[1][j]/(2*h) - kronecker(j,0))
            dAdq[N-2,n,j] = -(alpha*tau)*kronecker(N-2,n)*(-(dy[N-3][j])/(2*h) - kronecker(j,N-2))
        for k in range(1,N-2):
            for n in range(N-1):
                dAdq[k,n,j] = -(alpha*tau)*(kronecker(k-1,n)*(- dy[k][j]/(2*h)) + kronecker(k,n)*( (dy[k+1][j] - dy[k-1][j])/(2*h) - kronecker(k ,j)) + kronecker(k + 1,n)*( dy[k][j] / (2*h)))
    return dAdq
# # # # # # #Разреженная матрица производных по qj# # # # # #
def SpdAdq(dy,y,h,eps,N,t_0,T,M,alpha,j):
    I = empty(3*(N-3)+4)
    J = empty(3*(N-3)+4)
    V = empty(3*(N-3)+4,dtype=complex64)
    tau = (T - t_0)/M
    k = 0
    I[k] = 0
    J[k] = 0
    V[k] = - alpha*tau*(dy[1][j]/(2*h) - kronecker(j,0))
    k = k + 1
    I[k] = N - 2
    J[k] = N - 2
    V[k] = - alpha*tau*(-(dy[N-3][j])/(2*h) - kronecker(j,N-2))
    k = k + 1
    I[k] = N - 2
    J[k] = N - 3
    V[k] = - alpha*tau*(- dy[N-3][j]/(2*h))
    k = k + 1
    I[k] = 0
    J[k] = 1
    V[k] = - alpha*tau*( dy[0][j]/(2*h))
    for k in range(4,N+1):
        I[k] = k-3
        J[k] = k-2
        V[k] = - alpha*tau*( dy[k-3][j]/(2*h))
    for k in range(N+1,2*N-2):
        I[k] = k-N
        J[k] = k-N
        V[k] = - alpha*tau*( (dy[k - N + 1][j] - dy[k - N - 1][j])/(2*h) - kronecker(k - N,j))
    for k in range(2*N-2,3*N - 5):
        I[k] = k-(2*N - 3)
        J[k] = k-(2*N - 2)
        V[k] = - alpha*tau*( -dy[k-(2*N - 3)][j]/(2*h))
    DAdq = sparse.coo_matrix((V,(I,J)),shape = ((N-1),(N-1))).tocsr()
    return DAdq

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #Разреженная матрица Якоби# # # # # #
def SpFu(y, h, eps, N):
    I = empty(3*(N-3)+4)
    J = empty(3*(N-3)+4)
    V = empty(3*(N-3)+4)
    k = 0
    I[k] = 0
    J[k] = 0
    V[k] = (-2*(eps/(h**2))+y[1]/(2*h) - q[1])
    k = k + 1
    I[k] = N - 2
    J[k] = N - 2
    V[k] = (-2*(eps/(h**2))+(1-y[N-3])/(2*h) - q[N-1])
    k = k + 1
    I[k] = N - 2
    J[k] = N - 3
    V[k] = ((eps/(h**2))-y[N-2])
    k = k + 1
    I[k] = 0
    J[k] = 1
    V[k] = ((eps/(h**2))+y[0]/(2*h))
    for k in range(4,N+1):
        I[k] = k-3
        J[k] = k-2
        V[k] = ((eps/(h**2)) + y[k-3] / (2*h))
    for k in range(N+1,2*N-2):
        I[k] = k-N
        J[k] = k-N
        V[k] = (-2*(eps/(h**2)) + (y[k - N + 1] - y[k - N + 1])/(2*h) - q[k - N + 1])
    for k in range(2*N-2,3*N - 5):
        I[k] = k-(2*N - 3)
        J[k] = k-(2*N - 2)
        V[k] = ((eps/(h**2)) - y[k-(2*N - 3)] / (2*h))
    Fu = sparse.coo_matrix((V,(I,J)),shape = ((N-1),(N-1))).tocsr()
    return Fu
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
def dfdq1(dy,y,h,eps,N):
    dfdq1 = zeros((N-1,N-1))
    for j in range(N-1):
        dfdq1[0][j] = - ( kronecker(0,j)*y[0]) 
        dfdq1[N-2][j] = - y[N-2]*kronecker(N-2,j)
        for n in range(1,N-2):
            dfdq1[n][j] =  - y[n]*kronecker(n,j)
    return dfdq1
def dAdq1(dy,y,h,eps,N,t_0,T,M,alpha):
    tau = (T - t_0)/M
    dAdq1 = zeros((N-1,N-1,N-1),dtype=complex64)
    for j in range(N-1):
        for n in range(N-1):
            dAdq1[0,n,j] = -(alpha*tau)*kronecker(j,1)*kronecker(n,0)
            dAdq1[N-2,n,j] = -(alpha*tau)*kronecker(j,N-1)*kronecker(n,N-2)
        for k in range(1,N-2):
            for n in range(N-1):
                dAdq1[k,n,j] = -(alpha*tau)*kronecker(k + 1,j)*kronecker(k, n) 
    return dAdq1
# # # # # # #Разреженная первая матрица производных по qj# # # # # #
def SpdAdq1(dy,y,h,eps,N,t_0,T,M,alpha,j):
    tau = (T - t_0)/M
    I = empty(N-1)
    J = empty(N-1)
    V = empty(N-1,dtype=complex64)
    for k in range(N-1):
        I[k] = k
        J[k] = k
        V[k] = (alpha*tau)*kronecker(k,j)
    dAdq1 = sparse.coo_matrix((V,(I,J)),shape = ((N-1),(N-1))).tocsr()
    return dAdq1
a = 0
b = 1
t_0 = 0
T = 1
alpha = (1 + 1j)/2
eps = 0.1
M = 200
Steps = 200
fig1 = plt.figure()
camera = Camera(fig1)
UU = zeros(Steps)
x = linspace(a,b,N+1)
q = sin(3*pi*x)
u1 = DirectSol(a,b,N,t_0,T,M,u_init,alpha)
f_obs = u1[M]
q = zeros(N+1)
Q = zeros((Steps+1,N+1))
Q[0] = q
dudq = zeros((M + 1, N - 1, N - 1))
tau = (T-t_0)/M
h = (b-a)/N

for i in range( Steps):
    print(i)
    func = 0;
    u = DirectSol(a,b,N,t_0,T,M,u_init,alpha)
    y = zeros((M + 1,N - 1))
    for m in range(M + 1):
        y[m] = u[m,1:N]
    ff = -f(y[0], h, eps, N)
    fu = SpFu(y[0], h, eps, N)
    #dAq = dAdq1(dudq[0],y[0],h,eps,N,t_0,T,M,alpha)
    dfq = dfdq1(dudq[0],y[0],h,eps,N)
    RA = inv(sparse.eye((N-1),format="csc") - alpha*tau*(fu.tocsc()))
    #RA = (eye(N - 1) + alpha*tau*fu)
   # убрать здесь
    for j in range(N - 1):
        dAqL = SpdAdq1(dudq[0],y[0],h,eps,N,t_0,T,M,alpha,j)
        dfqL = dfq[:,j]
        A0 = dAqL@(RA)
        A1 = A0@(ff)
        A2 = tau*(RA@(-A1 + dfqL))
        dudq[1,:,j] = real(A2)
    for k in range(1, M):
        ff = -f(y[k], h, eps, N)
        #fu = Fu(y[k], h, eps, N)
        fu = SpFu(y[k], h, eps, N)
        #dAq = dAdq(dudq[k],y[k],h,eps,N,t_0,T,M,alpha)
        dfq = dfdq(dudq[k],y[k],h,eps,N)
        #RA = (eye(N - 1) + alpha*tau*fu)
        RA = inv(sparse.eye((N-1),format="csc") - alpha*tau*(fu.tocsc()))
        for j in range(N - 1):
            dAqL = SpdAdq(dudq[k],y[k],h,eps,N,t_0,T,M,alpha,j)
            dfqL = dfq[:,j]
            A0 = dAqL@(RA)
            A1 = A0@(ff)
            A2 = tau*(RA@(-A1 + dfqL))
            dudq[k + 1,:,j] = dudq[k,:,j] + real(A2)
    jjj = zeros(N - 1)
    f_obs1 = f_obs[1:N]
    for j in range(N - 1):
        for l in range(N - 1):
            jjj[j] = jjj[j] + 2*(y[M,l] - f_obs1[l])*dudq[M,l,j]/N
    jj =  zeros(N + 1)
    jj[1:N] = jjj
    for n in range(N + 1):
        func = func + (1/N)*(u[M,n]-f_obs[n])**2
    UU[i] = func


   # plt = axes(xlim=(a,b), ylim=(-10,10))
   # plt.set_xlabel('x'); plt.set_ylabel('q')ффф

    plt.plot(x,q)
    camera.snap()
    q = q - 1000*jj
    Q[i+1] = q
animation = camera.animate()
style.use('dark_background')
fig = figure()
ax = axes(xlim=(0,Steps), ylim=(0,0.005))
ax.set_xlabel('Iteration'); ax.set_ylabel('F')
ax.plot(linspace(0,Steps-1,Steps),UU, color='y', ls='-', lw=2)
print(UU)
savez('Results_type2_M200_N200_St200_beta1000', Steps=Steps, x=x, Q=Q, UU = UU )
