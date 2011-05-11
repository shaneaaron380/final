
#include "mat_mult_gpu.h"
#include "sys/time.h"
//#include "cuda.h"
#include "cuPrintf.cu"

#define SMEM_CACHE_SZ 32 
//#define THREADS_PER_BLOCK 512 

__global__ void MatMultKernel(const Matrix A, const Matrix B, Matrix C, const float alpha, const int N)
{
  int l = 0;
  //int j = (gridDim.x-1)*512 + threadIdx.x;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  //int j = blockIdx.y * blockDim.y + threadIdx.y;
  //int ixn = 0;
  float S;

  if (j < N) {
    //cuPrintf("%d,%d : %d,%d : %d,%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
    //if ( (j % 20) == 0) cuPrintf("%d,%d,%d\n", j, gridDim.x, threadIdx.x);
    //for (int i = 0; i < N*N; i+=N) {
    for (int i = 0; i < N; i++) {
			//ixn = i * N;
			S = alpha*B.els[i*N+j]; //S = B[i][j];
      //cuPrintf("i=%d,j=%d, S=%f\n", i, j, S);
      for (int k = 0; k < i; k++) {
        //S -= A.els[i*n+k] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
        S -= A.els[l++] * C.els[k*N+j]; //S -= A[i][k] * C[k][j];
        //cuPrintf("i=%d,j=%d,k=%d, S=%f, A=%f, C=%f\n", i, j, k, S, A.els[i*n+k], C.els[k*n+j]);
      }
      C.els[i*N+j] = S; //C[i][j] = S;
    }
  }
}

__global__ void MatMultKernelShared(const Matrix A, const Matrix B, Matrix C, const float alpha, const int N)
{
  int l = 0, k = 0;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx = threadIdx.x;
	float S;

  __shared__ float As[SMEM_CACHE_SZ];

  //if (j == 0) cuPrintf("N=%d\n", N);
  //if (t_idx == 241) cuPrintf("j=%d, t_idx=%d\n", j, t_idx);
  //Init cache to zero
  //if (t_idx < SMEM_CACHE_SZ) As[t_idx] = 0;
 
  if (j < N) {
		//cuPrintf("j = %d, t_idx = %d\n", j, t_idx);
    //cuPrintf("%d,%d : %d,%d : %d,%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
   	C.els[j] = alpha*B.els[j];
    //if (t_idx == 241) cuPrintf("j=%d, C.els[j]=%f, B.els[j]=%f\n", j, C.els[j], B.els[j]);
		for (int i = 1; i < N; i++) {
      
			S = alpha*B.els[i*N+j]; //S = B[i][j];
      //if (t_idx == 0) cuPrintf("i=%d, j=%d, Sinit=%f\n", i, j, S);
			k = 0;
			while ((k + 10) <= i) {
        __syncthreads();
				if (t_idx < 10) As[t_idx] = A.els[l+t_idx];
      	__syncthreads();
        //if (t_idx == 0) cuPrintf("4: As[%d] = A.els[%d] = %f\n", t_idx, l+t_idx, As[t_idx]);
        
        //cuPrintf("A+1=%f, C+1=%f\n", As[k+1], C.els[(k+1)*N+j]);
				S -=  (As[0]  * C.els[k*N+j]) + \
							(As[1]  * C.els[(k+1)*N+j]) + \
							(As[2]  * C.els[(k+2)*N+j]) + \
							(As[3]  * C.els[(k+3)*N+j]) + \
							(As[4]  * C.els[(k+4)*N+j]) + \
							(As[5]  * C.els[(k+5)*N+j]) + \
							(As[6]  * C.els[(k+6)*N+j]) + \
							(As[7]  * C.els[(k+7)*N+j]) + \
							(As[8]  * C.els[(k+8)*N+j]) + \
							(As[9]  * C.els[(k+9)*N+j]);
        
        //if(t_idx == 0) cuPrintf("10: i=%d,j=%d,k=%d,As[0]=%f,A.els[l]=%f,C.els[%d]=%f,S=%f\n", i, j, k, As[0], A.els[l+t_idx], k*N+j, C.els[k*N+j], S);
        k+=10;
     	  l+=10;
      }
			while ((k + 4) <= i) {
        __syncthreads();
				if (t_idx < 4) As[t_idx] = A.els[l+t_idx];
      	__syncthreads();
        //if (t_idx == 0) cuPrintf("4: As[%d] = A.els[%d] = %f\n", t_idx, l+t_idx, As[t_idx]);
        
        //cuPrintf("A+1=%f, C+1=%f\n", As[k+1], C.els[(k+1)*N+j]);
				S -=  (As[0]  * C.els[k*N+j]) + \
							(As[1]  * C.els[(k+1)*N+j]) + \
							(As[2]  * C.els[(k+2)*N+j]) + \
							(As[3]  * C.els[(k+3)*N+j]); 
        
        //if(t_idx == 0) cuPrintf("4: i=%d,j=%d,k=%d,As[0]=%f,A.els[l]=%f,C.els[%d]=%f,S=%f\n", i, j, k, As[0], A.els[l+t_idx], k*N+j, C.els[k*N+j], S);
        k+=4;
     	  l+=4;
      }
      while (k < i)  {
        __syncthreads();
      	if (t_idx == 0) As[0] = A.els[l];
        __syncthreads();
        S -= As[0] * C.els[k*N+j]; //S -= A[i][k] * C[k][j];
        //if(t_idx == 0) cuPrintf("1: i=%d,j=%d,k=%d,As[0]=%f,A.els[l]=%f,C.els[%d]=%f,S=%f\n", i, j, k, As[0], A.els[l], k*N+j, C.els[k*N+j], S);
     	  k++;
     	  l++;
      }
      C.els[i*N+j] = S; //C[i][j] = S;
      //if ((i*N+j) == 241) cuPrintf("i=%d, j=%d, C.els[%d] = %f\n", i, j, i*N+j, C.els[i*N+j]);
    }
  }
}

void MatMultGPU(const Matrix A, const Matrix B, Matrix C, const float alpha)
{
	Matrix d_A, d_B, d_C;
  const int n = A.width;
	cudaError_t cudaMallocReturnStatus;
	struct timeval timerValues;
	double start_time, end_time;
	double before_kernel, after_kernel;
	timerclear(&timerValues);	

  cudaPrintfInit();

	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t asize = ((A.width * A.height - A.width)/2) * sizeof(float);
	cudaMalloc((void**)&d_A.els, asize);
	cudaMallocReturnStatus = cudaMalloc((void**)&d_A.els, asize);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("ERROR: Couldn't allocate Matrix A on GPU, exiting\n"); exit(0);
	}
  TruncateMatrix(A);

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size_t size = B.width * B.height * sizeof(float);
	cudaMalloc((void**)&d_B.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("ERROR: Couldn't allocate Matrix B on GPU, exiting\n"); exit(0);
	}

	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void**)&d_C.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("ERROR: Couldn't allocate Matrix C on GPU, exiting\n"); exit(0);
	}

  int threadsPerBlock = n > 512 ? 512 : n;
  int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;
  printf("grids=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
	
	//Get start time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get start time of day\n");
	
	start_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	
	cudaMemcpy(d_A.els, A.els, asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice);
	
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get before kernel time of day\n");
	
	before_kernel = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	
	//MatMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, n);
	//Static shared memory
	MatMultKernelShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, n);
	//Dynamic shared memory
	//MatMultKernelShared<<<blocksPerGrid, threadsPerBlock, sizeof(float)*(n-1)>>>(d_A, d_B, d_C, alpha, n);
	cudaThreadSynchronize();	
	
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get after kernel time of day\n");
	
	after_kernel = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	
	cudaMemcpy(C.els, d_C.els, size, cudaMemcpyDeviceToHost);
  
	//Get end time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get end time of day\n");
	
	end_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	//printf("End secs: %ld, End usecs: %ld, Total Time: %f\n", timerValues.tv_sec, timerValues.tv_usec, end_time-start_time);
	printf("Total Time: %f\n", end_time-start_time);
	printf("Kernel Time: %f\n", after_kernel-before_kernel);
	printf("Transfer Time: %f\n", (end_time-after_kernel)+(before_kernel-start_time));
  
  cudaPrintfDisplay(stdout,true);
  cudaPrintfEnd();

	cudaFree(d_A.els);
	cudaFree(d_B.els);
	cudaFree(d_C.els);
}


