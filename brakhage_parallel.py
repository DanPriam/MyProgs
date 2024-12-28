from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, size, dot, sqrt, sum, random
#from module import *

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()


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

def brakhage_nu_method(A_part, b_delta_part, x_part, delta2,
                              comm_row, comm_col, M, N, nu) :
    #neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
    #neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)
    
    N_part = size(x_part);  M_part = size(b_delta_part)
    
    normAsq = array(0, dtype=float64)
    normA_part = array(0, dtype=float64)
    
    for j in range(M_part):
        for i in range(N_part):
            normA_part = normA_part + (A_part[j,i])**2

    comm_cart.Allreduce([normA_part, 1, MPI.DOUBLE], [normAsq, 1, MPI.DOUBLE], op=MPI.SUM)
    normA = sqrt(normAsq)
    A_part = A_part*(0.99/normA); b_delta_part = b_delta_part*(0.99/normA); delta2 = delta2 * ((0.99/normA)**2) 
    s = 1
    
    
    criterion = False
    
    Ax_part_temp = empty(M_part, dtype=float64)
    Ax_part = empty(M_part, dtype=float64)
    
    gr_part = empty(N_part, dtype=float64)
    gr_part_temp = empty(N_part, dtype=float64)
    
    requests = [MPI.Request() for i in range(2)]
    requests[0] = comm_row.Allreduce_init([Ax_part_temp, M_part, MPI.DOUBLE], 
                       [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM)
    requests[1] = comm_col.Allreduce_init([gr_part_temp, N_part, MPI.DOUBLE], 
                       [gr_part, N_part, MPI.DOUBLE], op=MPI.SUM)
    while criterion == False :

        if s == 1 :
            w = (4*nu + 2)/(4*nu + 1)
            x_prev_part = x_part.copy()
            Ax_part_temp[:] = dot(A_part, x_part)
            Ax_part[:] = zeros(M_part, dtype=float64)
            #comm_row.Allreduce([Ax_part_temp, M_part, MPI.DOUBLE], 
            #                   [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM)
            MPI.Prequest.Start(requests[0])
            MPI.Request.Wait(requests[0], status=None)

            b_part = b_delta_part - Ax_part
            gr_part_temp[:] = dot(A_part.T, b_part)
            gr_part[:] = zeros(N_part, dtype=float64)
            #comm_col.Allreduce([gr_part_temp, N_part, MPI.DOUBLE], 
            #                   [gr_part, N_part, MPI.DOUBLE], op=MPI.SUM)
            MPI.Prequest.Start(requests[1])
            MPI.Request.Wait(requests[1], status=None)

            x_part = x_part + w*gr_part
        else :
            mu = (s - 1)*(2*s - 3)*(2*s + 2*nu - 1)/((s + 2*nu - 1)*(2*s + 4*nu - 1)*(2*s + 2*nu - 3))
            w = 4*((2*s + 2*nu - 1)*(s + nu - 1))/((s + 2*nu - 1)*(2*s + 4*nu - 1))
            c_part = x_prev_part.copy()
            x_prev_part = x_part.copy()
            Ax_part_temp[:] = dot(A_part, x_part)
            Ax_part[:] = zeros(M_part, dtype=float64)
            MPI.Prequest.Start(requests[0])
            MPI.Request.Wait(requests[0], status=None)
            #comm_row.Allreduce([Ax_part_temp, M_part, MPI.DOUBLE], 
            #                   [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM)

            b_part = b_delta_part - Ax_part
            gr_part_temp[:] = dot(A_part.T, b_part)
            gr_part[:] = zeros(N_part, dtype=float64)
            MPI.Prequest.Start(requests[1])
            MPI.Request.Wait(requests[1], status=None)
            #comm_col.Allreduce([gr_part_temp, N_part, MPI.DOUBLE], 
            #                   [gr_part, N_part, MPI.DOUBLE], op=MPI.SUM)

            demp_part = x_part - c_part
            x_part = x_part + mu*demp_part + w*gr_part
        s = s + 1
        
        Ax_part_temp[:] = dot(A_part, x_part)
        Ax_part[:] = zeros(M_part, dtype=float64)
        MPI.Prequest.Start(requests[0])
        MPI.Request.Wait(requests[0], status=None)
        #comm_row.Allreduce([Ax_part_temp, M_part, MPI.DOUBLE], 
        #                   [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM)

        res_temp = sum((Ax_part - b_delta_part)**2)
        res = array(0, dtype=float64)
        comm_col.Allreduce([res_temp, 1, MPI.DOUBLE], 
                           [res, 1, MPI.DOUBLE], op=MPI.SUM)

        res = res/M
        if res <= delta2 :
            criterion = True
            
    return x_part, s

time_start = empty(1, dtype=float64)
elapsed_time = empty(1, dtype=float64)
total_time = empty(1, dtype=float64)

comm.Barrier()

time_start[0] = MPI.Wtime()


x_part, s = brakhage_nu_method(A_part, b_delta_part, x_part, delta2,comm_row, comm_col, M, N, nu)

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
