#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cublas_v2.h>


// Reference : https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A,int nr_mat)
{
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A*nr_mat);
}

// using cublas matrix multiplication
 void gpu_blas_mmul(cublasHandle_t &handle,const float *A, const float *B, float *C ,const int m, const int k, const int n,int nr_mat) {
    
	int P = nr_mat ;
	
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	 //create cuda stream
	cudaStream_t stream[100];
	for (int i = 0; i < P; i ++)
	{
    cudaStreamCreate(&stream[i]);
	}
	
	for (int iN = 0; iN < P; ++iN)
	{
	const float *const d_tmpIn = A + iN*m*k;
	//const float *const d_tmpInB = B + iN*k*n;
	float *const d_tmpOut = C + iN *m*n;
	cublasSetStream(handle,stream[iN]); //cublas set stream
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_tmpIn, lda, B, ldb, beta, d_tmpOut, ldc);

	}
	
	
 }



int main(void)
{
clock_t tic = clock();

//C(m,k) = A(m,n)*B(n,k)

	float   *h_A ,*d_A,*h_B,*d_B,*h_C,*d_C; // intialize variable
	int M = 500 ;
	int N = 500 ; 
	int K = 400;
	int P = 100;
	int Q = 1;
	
	
	int ASize = M*N*P;
	int BSize = N*K*Q;
	int CSize = M*K*P;
	
	// Memory allocation
	h_A = (float *)malloc(sizeof(float)*ASize) ;
	h_B = (float *)malloc(sizeof(float)*BSize) ;
	h_C = (float *)malloc(sizeof(float)*CSize) ;
	
	cudaMalloc((void**)&d_A, sizeof(float)*ASize) ;
	cudaMalloc((void**)&d_B, sizeof(float)*BSize) ;
	cudaMalloc((void**)&d_C, sizeof(float)*CSize) ;

	
	// sert intiall value 0 of matrix
	memset(h_A, 0, sizeof(float)*ASize) ;
	memset(h_B, 0, sizeof(float)*BSize) ;
	memset(h_C, 0, sizeof(float)*CSize) ;
	
    cudaMemset(d_A, 0, sizeof(float)*ASize) ;
	cudaMemset(d_B, 0, sizeof(float)*BSize) ;
	cudaMemset(d_C, 0, sizeof(float)*CSize) ;
	
	//generate random value
	GPU_fill_rand(d_A, M, N,P ) ;
	cudaMemcpy(h_A, d_A, sizeof(float)*ASize, cudaMemcpyDeviceToHost) ;
	
	GPU_fill_rand(d_B, N, K,Q ) ;
	cudaMemcpy(h_B, d_B, sizeof(float)*BSize, cudaMemcpyDeviceToHost) ;
	
	
	
	
	

	// Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
	
	
	
	
	
	
	
	//call mulitplication function
	gpu_blas_mmul(handle,d_A, d_B, d_C, M, N, K,P);
	cudaMemcpy(h_C,d_C,sizeof(float)*CSize,cudaMemcpyDeviceToHost);
	
	
	
	// Destroy the handle
    cublasDestroy(handle);
	
	
	/*//print matrix
	
	printf("==========A MAtrix================\n");
	for(int p=0;p<P;++p)
	{
	for(int i=0;i<M;++i)
	{
	for(int j=0;j<N;++j)
	{
	printf("%f ",h_A[p*N*M+j*M+i]);
	
	}
	printf("\n");
	}
	printf("\n");
	}
	
	
	
	printf("========B MAtrix========\n");
	for(int p=0;p<Q;++p)
	{
	for(int i=0;i<N;++i)
	{
	for(int j=0;j<K;++j)
	{
	printf("%f ",h_B[p*N*K+j*N+i]);
	}
	printf("\n");
	}
	printf("\n");
	}
	
	printf("========C MAtrix==========\n");
	for(int p=0;p<P;++p)
	{
	for(int i=0;i<M;i++)
	{
	for(int j=0;j<K;j++)
	{
	printf("%f ",h_C[p*M*K+i+j*M]);
	}
	printf("\n");
	}
	printf("\n");
	}*/

	clock_t toc = clock();
	printf("Cuda Runnig Time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

}

	
	
	
	
	
	
	