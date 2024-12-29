# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:01:12 2024

@author: Даниил
"""
from numpy import zeros, array, size, dot, random, linalg
import time

N = 100
M = 100

tau = 0.00001
St = 1000
n = 0.3
A = random.sample((M, N))
# A = zeros((M,N))
# for i in range(M):
#     for j in range(N):
#         if (j/(N - 1) >= i/(M - 1)):
#             A[i,j] = (i/(M - 1))*(1 - j/(N - 1))*(1/(N - 1))
#         else:
#             A[i,j] = (j/(N - 1))*(1 - i/(M - 1))*(1/(N - 1))
from numpy import sin, pi
x_model = array([sin(2*pi*i/(N-1)) for i in range(N)])
# x_model = array([2 for i in range(N)])
b = dot(A, x_model)

Delta = 0.5
random.seed(1)
b_delta = b + Delta*(random.random(len(b))-
                     random.random(len(b)))

delta2 = sum((b - b_delta)**2)/len(b)

nu = 0.5
'''
norm_A = linalg.norm(A)
A = A/norm_A; b_delta = b_delta/norm_A; delta2 = delta2/norm_A**2
'''
x = zeros(N)
v = zeros(N)
def fv(A, b_delta, t, vv, x):
    # if (t == 0):
    #     t = t + 0.01*tau
    return (1/t)*dot(A.T, b_delta) - (1/t)*(t**(-n)-n)*vv - t**(n)*dot(A.T,dot(A,vv)) - (1/t)*dot(A.T,dot(A,x))

def fx(t, vv):
    return vv
    
def func_solver(A, b_delta, x, v, tau, St):
    s = 0
    err_list = []
    err = sum((dot(A,x) - b_delta)**2)/len(b_delta)
    err_list.append(err)
    # while sum((dot(A,x) - b_delta)**2)/len(b_delta) > delta2 :
    for s in range(St):
        kx1 = fx(tau*s + 0.25*tau, v)
        kv1 = fv(A, b_delta, tau*s + 0.25*tau, v, x)
        kx2 = fx(tau*s + 0.75*tau, v + 1*tau*kv1)
        kv2 = fv(A, b_delta, tau*s + 0.75*tau, v + 1*tau*kv1, x + 1*tau*kx1)
        # kx1 = fx(tau*s, v)
        # kv1 = fv(A, b_delta, tau*s, v, x)
        # kx2 = fx(tau*s + 0.5*tau, v + 0.5*tau*kv1)
        # kv2 = fv(A, b_delta, tau*s + 0.5*tau, v + 0.5*tau*kv1, x + 0.5*tau*kx1)
        # kx3 = fx(tau*s + 0.5*tau, v + 0.5*tau*kv2)
        # kv3 = fv(A, b_delta, tau*s + 0.5*tau, v + 0.5*tau*kv2, x + 0.5*tau*kx2)
        # kx4 = fx(tau*s + tau, v + tau*kv3)
        # kv4 = fv(A, b_delta, tau*s + tau, v + tau*kv3, x + tau*kx3)
        # x = x + (tau/6)*(kx1 + 2*kx2 + 2*kx3 + kx4)
        # v = v + (tau/6)*(kv1 + 2*kv2 + 2*kv3 + kv4)
        x = x + (tau/2)*(kx1 + kx2)
        v = v + (tau/2)*(kv1 + kv2)
        err = sum((dot(A,x) - b_delta)**2)/len(b_delta)
        err_list.append(err)
        s = s + 1
    return x, s, err_list


time_start = time.time()

x, s, err_list = func_solver(A, b_delta, x, v, tau, St)

total_time = time.time() - time_start

print(delta2)
print(err_list[-1])

from matplotlib.pyplot import figure, axes, show
from numpy import arange
fig = figure()
ax = axes(xlim=(0, N), ylim=(-3, 3))
ax.set_xlabel('i'); ax.set_ylabel('x[i]')
ax.plot(arange(N), x_model, '-g', lw=7)
ax.plot(arange(N), x, '-r', lw=2)
show()


fig = figure()
ax = axes(xlim=(0, s+1))
ax.set_xlabel('i'); ax.set_ylabel('err')
ax.plot(arange(s + 1), err_list, '-r', lw=2)
show()

print(f'Time for consecutive algorithm: {total_time:9.3f} sec')
print(f'Number of iteration is {s} (N={N})')
print(f'Time for consecutive algorithm: {total_time:9.3f} sec')
print(f'Number of iteration is {s} (N={N})')

