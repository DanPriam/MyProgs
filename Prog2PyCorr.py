# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:35:52 2023

@author: Даниил
"""

#
from numpy import zeros, linspace, sin, pi, complex64, real
from matplotlib.pyplot import style, figure, axes
from matplotlib import pyplot as plt
from celluloid import Camera
from  numpy import savez


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
#Правые части прямой и сопряженной задачи
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
#Матрица Fu; матрица сопряженной задачи. Затем a, b ,c и алгоритм прогонки 
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
        Fu2[0,n] = kronecker(n,0)*(2*(eps/(h**2))+y[1]/(2*h) + q[1]) + kronecker(1,n)*(-(eps/(h**2)) + y[0] / (2*h))
        Fu2[N-2,n] = kronecker(N-2,n)*(2*(eps/(h**2))- y[N-3]/(2*h) + q[N-1]) + kronecker(N-3,n)*(-(eps/(h**2)) - y[N-2] / (2*h))
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
       u[m + 1,1:(N) ] = y
    return u
def ConjugateSol(a,b,N,t_0,T,M,u_init,alpha, V):
    h = (b - a)/N;# x = linspace(a,b,N+1)
    tau = (T - t_0)/M
    u = zeros((M + 1,N + 1))
    y = zeros(N - 1)
    v = zeros(N + 1)
    v = V
    u[M] = -2*(v - f_obs)
    y = -2*(v[1:N] - f_obs[1:N])
    for m in range(M,0,-1) :
       a,b,c = DiagonalsPreparation2(Fu2(y, h, eps, N) ,h,N,tau,alpha)
       v = AlgTriDiag(a,b,c,f2(y,h, eps, N))
       y = y - tau*v
       u[m - 1, 0] = 0
       u[m - 1, N] = 0
       u[m - 1,1:N ] = y
    return u
def Gradient(U, V,M,N):
    J = zeros(N + 1)
    u = zeros((M + 1, N + 1))
    v = zeros((M + 1, N + 1))
    u = U
    v = V
    for i in range(N + 1):
        S = 0
        for j in range(M + 1):
            S = S + u[j,i]*v[j,i]*(1/M)
        J[i] = S;
    return J

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

for i in range( Steps):
    func = 0;
    print(i)
    u = DirectSol(a,b,N,t_0,T,M,u_init,alpha)
    v = ConjugateSol(a,b,N,t_0,T,M,u_init,alpha, u[M])
    for n in range(N + 1):
        func = func + (1/N)*(u[M,n]-f_obs[n])**2
    UU[i] = func
    jj = Gradient(u, v, M, N)
    ''' 
    plt = axes(xlim=(a,b), ylim=(-10,10))
    plt.set_xlabel('x'); plt.set_ylabel('q')ффф
    '''
    plt.plot(x,q)
    camera.snap()
    q = q - 1*jj
    Q[i+1] = q
animation = camera.animate()
style.use('dark_background')
fig = figure()
ax = axes(xlim=(0,Steps), ylim=(0,0.005))
ax.set_xlabel('Iteration'); ax.set_ylabel('F')
ax.plot(linspace(0,Steps-1,Steps),UU, color='y', ls='-', lw=2)
print(UU)
savez('Results_M200_N200_St200', Steps=Steps, x=x, Q=Q, UU = UU )