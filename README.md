hybrid_gemm
===========

The Idea is to use the power of GPU for the computational work. Mainly Two BLAS libraries
are used by the PDGEMM Solver. At the CPU level (between nodes) we use ScaLAPACK
(Scalable Linear Algebra PACKage) which is portable to any system supporting MPI or PVM.
At the GPU level, we use Magma (Matrix Algebra on GPU and Multicore Architecture).
Magma is a linear algebra accelerated library for GPUs.

The total Time of execution for the GEMM call in Magma is measured. The size of the
matrices and the number of nodes are varied and performance are analyzed on Tődi. The
time for ScaLAPACK on 1 node is taken as the time for serial code execution on Tődi.
