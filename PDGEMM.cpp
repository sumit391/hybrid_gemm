#include <iostream>
#include <string>
#include <cassert>
#include <chrono>
#include "include/matrix.hpp"
#include <iostream>
#include <string>
#include <cassert>
#include <chrono>
//#include "matrix.hpp"
#include <stdio.h>
#include <cuda.h>
#include "magma.h"
#include "magma_lapack.h"

#ifndef HAVE_CUBLAS
#define HAVE_CUBLAS
#endif

#ifndef ADD_
#define ADD_
#endif

#define magma_ssetvector(n, hx_src, incx, dy_dst, incy)   magma_ssetvector_internal( n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )
#define magma_sgetvector(n, dx_src, incx, hy_dst, incy)   magma_sgetvector_internal( n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

extern "C"
{
	void Cblacs_pinfo(int&, int&);
	void Cblacs_get(int, int, int&);
	void Cblacs_gridinit(int&, char const*, int, int);
	void Cblacs_gridinfo(int, int&, int&, int&, int&);
	void Cblacs_barrier(int , char const*);
	void Cblacs_gridexit(int);
	void Cblacs_exit(int);

	int numroc_(int const& n, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
	int indxg2p_(int const& glob, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
 	int indxl2g_(int const& loc, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
    	void descinit_( int *desc, int const& m, int const& n, int const& mb, int const& nb, int const& irsrc, int const& icsrc, int const& ictxt, int const& lld, int& info);
	
    	void pdgemm_( char const *transa, char const *transb, int const& M, int const& N, int const& K, double const& ALPHA,  double * A, int const& IA, int const& JA, int * DESCA, double * B, int const& IB, int const& JB, int * DESCB, double const& BETA, double * C, int const& IC, int const& JC, int * DESCC );      
	void magma_sgemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, float alpha, magmaFloat_const_ptr dA, magma_int_t ldda,     magmaFloat_const_ptr dB, magma_int_t lddb, float beta, magmaFloat_ptr dC, magma_int_t lddc);
    	//magma_init();
    	magma_err_t magma_init(void); 	
    	//magma_smalloc_pinned(&a,n);
    	static magma_err_t magma_smalloc_pinned (float** ptrPtr, size_t n); 	
    	//magma_smalloc(&a_gpu,n);
    	static magma_err_t magma_smalloc (magmaFloat_ptr* ptrPtr, size_t n);
    	//lapackf77_slarnv(&ione, ISEED, &mk1, a);    
    	void lapackf77_slarnv (const magma_int_t* idist, magma_int_t* iseed, const magma_int_t* n, float* x);

}
typedef hpcse::matrix<double,hpcse::column_major> matrix_type;
int main(int argc, char **argv)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start1, end1;
	start1 = std::chrono::high_resolution_clock::now();
	int rank=-1;
    int nprocs=-1;
    int info=0;
    Cblacs_pinfo(rank,nprocs);
	//	assert(argc==3);
    int nprow=atoi(argv[1]);
    int npcol=atoi(argv[2]);
    int ctxt=0;
    Cblacs_get(0,0,ctxt);
    Cblacs_gridinit(ctxt,"C",nprow,npcol);
    int myrow = -1;
    int mycol = -1;
    Cblacs_gridinfo(ctxt,nprow,npcol,myrow,mycol);
    /*if (myrow>=0) 
	{	
		std::cout << "Rank " << rank << " has coordinates " << myrow << " " << mycol << "\n";
	}
    	else
	{	
		std::cout << "Rank " << rank << " is not used \n";
	}*/

    int n1=atoi(argv[3]);//1024;
    //int n2=atoi(argv[4]);//704;
    int nb=atoi(argv[5]);
   	//  int np = numroc_(n1,nb,myrow,0,nprow);  // computes number of rows/cols of a distributed matrix owned by the prcess indicated by myrow
    int nq = numroc_(n1,nb,mycol,0,npcol);
    magma_init();
	magma_timestr_t start , end;
	float gpu_time ;
	magma_int_t m = n1; 
	magma_int_t n = nq; 
	magma_int_t k = n1; 
	magma_int_t mk=m*k; 
	magma_int_t kn=k*n; 
	magma_int_t mn=m*n; 
	float *a; 
	float *b; 
	float *c; 
	float *d_a; 
	float *d_b; 
	float *d_c; 
	float alpha = MAGMA_S_MAKE ( 1.0 , 0.0 ); 
	float beta = MAGMA_S_MAKE ( 0.0 , 0.0 ); 
	magma_int_t ione = 1;
	magma_int_t ISEED [4] = { 0 ,1 ,2 ,3 }; 
	magma_err_t err;
	// allocate matrices on the host
	err = magma_smalloc_pinned ( &a , mk ); 
	err = magma_smalloc_pinned ( &b , kn ); 
	err = magma_smalloc_pinned ( &c , mn ); 
	// allocate matrix and vectors on the device
	err = magma_smalloc ( &d_a , mk ); 
	err = magma_smalloc ( &d_b , kn ); 
	err = magma_smalloc ( &d_c , mn ); 
	lapackf77_slarnv (& ione ,ISEED ,&mk ,a); 
	lapackf77_slarnv (& ione ,ISEED ,&kn ,b);
	lapackf77_slarnv (& ione ,ISEED ,&mn ,c); 
	
	magma_ssetmatrix ( m, k, a, m, d_a , m ); 
	magma_ssetmatrix ( k, n, b, k, d_b , k ); 
	magma_ssetmatrix ( m, n, c, m, d_c , m ); 
	start = get_current_time ();
	magma_sgemm(MagmaNoTrans,MagmaNoTrans,m,n,k,alpha,d_a,m,d_b,k,beta,d_c,m);
	end = get_current_time ();
	gpu_time = GetTimerValue (start ,end )/1e3;
	std::cout << std::endl << " magma_sgemm time  on rank ::" << rank << " :" << gpu_time ;
	magma_sgetmatrix ( m, n, d_c , m, c, m ); // copy d_c -> c
	//std::cout << std::endl << " after magma_sgemm :" << std::endl;
	magma_free_pinned (a); 
	magma_free_pinned (b); 
    magma_free_pinned (c); 
	magma_free (d_a ); 
	magma_free (d_b ); 
	magma_free (d_c ); 
	magma_finalize (); 
	Cblacs_exit(0);
    end1 = std::chrono::high_resolution_clock::now();
	double t1 = std::chrono::duration<double>(end1-start1).count();
	std::cout << std::endl << "Total time : " << t1;
	return 0;
}
