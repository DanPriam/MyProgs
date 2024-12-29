# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:14:07 2024

@author: Даниил
"""

from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, size, dot, sqrt, sum, random
#from module import *

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()

n = 0.5
tau = 0.00001
St = 1000

N = 200
M = 300
nu = 0.5

num_row = num_col = int32(sqrt(numprocs))

comm_cart = comm.Create_cart(dims=(num_row, num_col), 
                             periods=(True, True), reorder=True)

rank_cart = comm_cart.Get_rank()

my_row, my_col = comm_cart.Get_coords(rank_cart)

comm_col = comm_cart.Split(rank_cart % num_col, rank_cart)
comm_row = comm_cart.Split(rank_cart // num_col, rank_cart)

def auxiliary_arrays(M, num): 
    ave, res = divmod(M, num) 
    rcounts = [0] * num 
    displs = [0] * num 
    for k in range(0, num): 
        if k < res: 
            rcounts[k] = ave + 1 
        else: 
            rcounts[k] = ave 
        if k == 0: 
            displs[k] = 0 
        else: 
            displs[k] = displs[k-1] + rcounts[k-1] 
    return rcounts, displs

rcounts_M, displs_M = auxiliary_arrays(M, num_row)
rcounts_N, displs_N = auxiliary_arrays(N, num_col)

M_part = rcounts_M[my_row]
N_part = rcounts_N[my_col]

# ----------------------------------------------------
A_part = empty((M_part, N_part), dtype=float64)

from numpy import random
A_part = random.random_sample((M_part, N_part))

if rank_cart == 0 :
    from numpy import sin, pi
    x_model = array([sin(2*pi*i/(N-1)) for i in range(N)], 
                    dtype=float64)
else :
    x_model = None

x_part = empty(N_part, dtype=float64) 

if rank_cart in range(num_col) :
    comm_row.Scatterv([x_model, rcounts_N, displs_N, MPI.DOUBLE], 
                      [x_part, N_part, MPI.DOUBLE], root=0)

comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)
b_part_temp = dot(A_part, x_part)
b_part = empty(M_part, dtype=float64)
comm_row.Allreduce([b_part_temp, M_part, MPI.DOUBLE], 
                   [b_part, M_part, MPI.DOUBLE], op=MPI.SUM)
                   
if rank_cart in range(0, numprocs , num_col) :
    b = None
    b_delta = None
if rank_cart == 0 :
    b = zeros(M, dtype=float64)
    
if rank_cart in range(0, numprocs , num_col) :
    comm_col.Gatherv([b_part, M_part, MPI.DOUBLE], 
                     [b, rcounts_M, displs_M, MPI.DOUBLE], 
                     root=0)       

if rank_cart == 0 : 
    Delta = 0.5
    random.seed(1)
    b_delta = b + Delta*(random.random(len(b))-
                         random.random(len(b)))
    delta2 = sum((b - b_delta)**2)/len(b)
else : 
    delta2 = array(0, dtype=float64)
    
comm_cart.Bcast([delta2, 1, MPI.DOUBLE], root=0)  
    
b_delta_part = empty(M_part, dtype=float64)
if rank_cart in range(0, numprocs , num_col) :
    comm_col.Scatterv([b_delta, rcounts_M, displs_M, MPI.DOUBLE],
                      [b_delta_part, M_part, MPI.DOUBLE], 
                      root=0)  
comm_row.Bcast([b_delta_part, M_part, MPI.DOUBLE], root=0)                     
# ----------------------------------------------------
    
x_part = zeros(N_part, dtype=float64)
v_part = zeros(N_part, dtype=float64)



def fv(A_part, b_delta_part, t, vv_part, x_part):
    # if (t == 0):
    #     t = t + 0.01*tau
    Ax_part_temp = empty(M_part, dtype=float64)
    Ax_part = empty(M_part, dtype=float64)

    Avv_part_temp = empty(M_part, dtype=float64)
    Avv_part = empty(M_part, dtype=float64)
    
    ATAx_part_temp = empty(N_part, dtype=float64)
    ATAx_part = empty(N_part, dtype=float64)

    ATAvv_part_temp = empty(N_part, dtype=float64)
    ATAvv_part = empty(N_part, dtype=float64)
    
    ATb_part_temp = empty(N_part, dtype=float64)
    ATb_part = empty(N_part, dtype=float64)
    
    Ax_part_temp[:] = dot(A_part, x_part)
    Ax_part[:] = zeros(M_part, dtype=float64)
    comm_row.Allreduce([Ax_part_temp, M_part, MPI.DOUBLE], 
                       [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM)
    ATAx_part_temp = dot(A_part.T, Ax_part)
    ATAx_part[:] = zeros(N_part, dtype=float64)
    comm_col.Allreduce([ATAx_part_temp, N_part, MPI.DOUBLE], 
                       [ATAx_part, N_part, MPI.DOUBLE], op=MPI.SUM)
    
    Avv_part_temp[:] = dot(A_part, vv_part)
    Avv_part[:] = zeros(M_part, dtype=float64)
    comm_row.Allreduce([Avv_part_temp, M_part, MPI.DOUBLE], 
                       [Avv_part, M_part, MPI.DOUBLE], op=MPI.SUM)
    ATAvv_part_temp = dot(A_part.T, Avv_part)
    ATAvv_part[:] = zeros(N_part, dtype=float64)
    comm_col.Allreduce([ATAvv_part_temp, N_part, MPI.DOUBLE], 
                       [ATAvv_part, N_part, MPI.DOUBLE], op=MPI.SUM)
    ATb_part_temp[:] = dot(A_part.T, b_delta_part)
    ATb_part[:] = zeros(N_part, dtype=float64)
    comm_col.Allreduce([ATb_part_temp, N_part, MPI.DOUBLE], 
                       [ATb_part, N_part, MPI.DOUBLE], op=MPI.SUM)
    return (1/t)*ATb_part - (1/t)*(t**(-n)-n)*vv_part - t**(n)*ATAvv_part - (1/t)*ATAx_part

def fx(t, vv_part):
    return vv_part
    
def func_solver(A_part, b_delta_part, x_part, v_part, tau, n, St):
    s = 0
    #err_list = []
    #err = sum((dot(A,x) - b_delta)**2)/len(b_delta)
    #err_list.append(err)
    # while sum((dot(A,x) - b_delta)**2)/len(b_delta) > delta2 :
    for s in range(St):
        kx1_part = fx(tau*s + 0.25*tau, v_part)
        kv1_part = fv(A_part, b_delta_part, tau*s + 0.25*tau, v_part, x_part)
        kx2_part = fx(tau*s + 0.75*tau, v_part + 1*tau*kv1_part)
        kv2_part = fv(A_part, b_delta_part, tau*s + 0.75*tau, v_part + 1*tau*kv1_part, x_part + 1*tau*kx1_part)
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
        x_part = x_part + (tau/2)*(kx1_part + kx2_part)
        v_part = v_part + (tau/2)*(kv1_part + kv2_part)
        #err = sum((dot(A,x) - b_delta)**2)/len(b_delta)
        #err_list.append(err)
        s = s + 1
    return x_part, s



time_start = empty(1, dtype=float64)
elapsed_time = empty(1, dtype=float64)
total_time = empty(1, dtype=float64)

comm.Barrier()

time_start[0] = MPI.Wtime()


x_part, s = func_solver(A_part, b_delta_part, x_part, v_part, tau, n, St)

elapsed_time[0] = MPI.Wtime() - time_start[0]
comm_cart.Reduce([elapsed_time, 1, MPI.DOUBLE], 
                 [total_time, 1, MPI.DOUBLE], op=MPI.MAX, root=0)

if rank_cart == 0 :
    x = zeros(N, dtype=float64)
elif rank_cart in range(1, num_col) :
    x = None
    
if rank_cart in range(num_col) :
    comm_row.Gatherv([x_part, N_part, MPI.DOUBLE], 
                     [x, rcounts_N, displs_N, MPI.DOUBLE], 
                     root=0)

if rank_cart == 0 :
    from matplotlib.pyplot import figure, axes, show
    from numpy import arange
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(arange(N), x_model, '-g', lw=7)
    ax.plot(arange(N), x, '-r', lw=2)
    show()
    
if rank_cart == 0 :
    print(f'Time for {numprocs} processes is {total_time[0]:9.3f} sec, s={s-1}, N={N}')