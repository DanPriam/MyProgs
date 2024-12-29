Prog2PyCorr, Prog3_mod and Prog_Plot were parts of my bachelor's thesis and are dedicated to a numerical solution of an inverse coefficient problem for an initial-boundary problem for Burgers' type equation. 
This problem can be solved by minimizing the residual functional of a problem which can be done by means of gradient descent method. 
For this method to be implemented one needs to calculate a gradient of the functional and there are two approaches to this problem.
The name of the first method is "first differentiate, then discretisize" and as a part of this approach you first need to find an analytical formula of a functional gradient by means of functional analysis and then discretisize it and use a desrete formula.
The name of the the second is "first discretisize, then differentiate" and there you first discretisize the target functional and treat it as a funtion of finite number of variables. Then you differentiate it just as a funtion.
My thesis aim was to compare efficiency of these 2 approaches in application two a problem considered and thus Prog2PyCorr is dedicated to a first approach, Prog3_mod - to a second one and Prog_Plot is an auxillary program for plotting the results.

Then there are many different versions of parallel realisations of Brakhage's nu-method. It is a method iterative regularization.  
brakhage_parallel_3req is a parallel realisation that uses the MPI-3 standart. 
brakhage_parallel uses the most recent MPI-4 standart.
brakhage_parallel_cd uses MPI-4 and CUDA.
There is also a fortran realisation of MPI-3 version.

ARn_consecutive and ARn_parallel are respectively concecutive and parallel realisitions of Accelerated Regularization of the order n.
