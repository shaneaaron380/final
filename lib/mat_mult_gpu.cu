
#include "mat_mult_gpu.h"
#include "sys/time.h"
//#include "cuda.h"
#include "cuPrintf.cu"

#define A_SM_CACHE_SZ 16 
#define THREADS_PER_BLOCK 512 

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
  int l = 0;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx = threadIdx.x;
  //int j = blockIdx.y * blockDim.y + threadIdx.y;
  //int M = N % A_SM_CACHE_SZ ? N/A_SM_CACHE_SZ+1 : N/A_SM_CACHE_SZ;
	//int M = N > A_SM_CACHE_SZ ? A_SM_CACHE_SZ : N;
	float S;


  __shared__ float As[A_SM_CACHE_SZ];
 
  if (j < N) {
		cuPrintf("j = %d, t_idx = %d\n", j, t_idx);
    //cuPrintf("%d,%d : %d,%d : %d,%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
   	C.els[j] = B.els[j]; 
		for (int i = 1; i < N; i++) {
      
			S = alpha*B.els[i*N+j]; //S = B[i][j];
      //cuPrintf("i=%d,j=%d, S=%f\n", i, j, S);
			for (int k = 0; k < i; k+=A_SM_CACHE_SZ) {
				__syncthreads();
				if (t_idx < A_SM_CACHE_SZ) { 
      		if (t_idx < (i-k)) As[t_idx] = A.els[l+t_idx];
      		else As[t_idx] = 0;
				}
      	//if (j == 0) cuPrintf("j=%d, i=%d, As[p]=%f\n", j, i, As[p]);
      	__syncthreads();
        //S -= A.els[i*N+k] * C.els[k*N+j]; //S -= A[i][k] * C[k][j];
        //S -= A.els[l] * C.els[k*N+j]; //S -= A[i][k] * C[k][j];
				S -=	As[k] 		* C.els[k*N+j] 			+ \
							As[k+1] 	* C.els[(k+1)*N+j] 	+ \
							As[k+2] 	* C.els[(k+2)*N+j] 	+ \
							As[k+3] 	* C.els[(k+3)*N+j] 	+ \
							As[k+4] 	* C.els[(k+4)*N+j] 	+ \
							As[k+5] 	* C.els[(k+5)*N+j] 	+ \
							As[k+6] 	* C.els[(k+6)*N+j] 	+ \
							As[k+7] 	* C.els[(k+7)*N+j] 	+ \
							As[k+8] 	* C.els[(k+8)*N+j] 	+ \
							As[k+9] 	* C.els[(k+9)*N+j] 	+ \
							As[k+10] 	* C.els[(k+10)*N+j] + \
							As[k+11] 	* C.els[(k+11)*N+j] + \
							As[k+12] 	* C.els[(k+12)*N+j] + \
							As[k+13] 	* C.els[(k+13)*N+j] + \
							As[k+14] 	* C.els[(k+14)*N+j] + \
							As[k+15] 	* C.els[(k+15)*N+j]; //S -= A[i][k] * C[k][j];
        //cuPrintf("i=%d,j=%d,k=%d, S=%f, A=%f, C=%f\n", i, j, k, S, As[k], C.els[k*N+j]);
      }
     	l += i;
      C.els[i*N+j] = S; //C[i][j] = S;
    }
  }
}

//__global__ void MatMultKernelShared(const Matrix A, const Matrix B, Matrix C, const float alpha, const int N)
//{
//  int l = 0;
//  int j = blockIdx.x * blockDim.x + threadIdx.x;
//  //int j = blockIdx.y * blockDim.y + threadIdx.y;
//  int M = N % A_SM_CACHE_SZ ? N/A_SM_CACHE_SZ+1 : N/A_SM_CACHE_SZ;
//	//int M = N > A_SM_CACHE_SZ ? A_SM_CACHE_SZ : N;
//	float S;
//
//  __shared__ float As[A_SM_CACHE_SZ];
// 
//  if (j < N) {
//    //cuPrintf("%d,%d : %d,%d : %d,%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
//    for (int i = 0; i < N; i++) {
//			int row1 = i*N;
//      S = alpha*B.els[row1+j]; //S = B[i][j];
//      __syncthreads();
//      if (threadIdx.x < i) As[threadIdx.x] = A.els[l+threadIdx.x];
//      else As[threadIdx.x] = 0;
//      //As[THREADS_PER_BLOCK+threadIdx.x] = A.els[l+THREADS_PER_BLOCK+threadIdx.x];
//      //if (j == 0) cuPrintf("j=%d, i=%d, As[p]=%f\n", j, i, As[p]);
//      __syncthreads();
//      //cuPrintf("i=%d,j=%d, S=%f\n", i, j, S);
//			for (int k = 0; k < i; k+=4) {
//        //S -= A.els[i*N+k] * C.els[k*N+j]; //S -= A[i][k] * C[k][j];
//        //S -= A.els[l] * C.els[k*N+j]; //S -= A[i][k] * C[k][j];
//				S -= As[k] * C.els[k*N+j] + As[k+1] * C.els[(k+1)*N+j] + As[k+2] * C.els[(k+2)*N+j] + As[k+3] * C.els[(k+3)*N+j]; //S -= A[i][k] * C[k][j];
//        //cuPrintf("i=%d,j=%d,k=%d, S=%f, A=%f, C=%f\n", i, j, k, S, As[k], C.els[k*N+j]);
//      }
//     	l += i;
//      C.els[row1+j] = S; //C[i][j] = S;
//    }
//  }
//}

//__global__ void MatMultKernelShared(const Matrix A, const Matrix B, Matrix C, const float alpha, const int n)
//{
//  int l = 0;
//  int j = blockIdx.x * blockDim.x + threadIdx.x;
//  //int j = blockIdx.y * blockDim.y + threadIdx.y;
//  int m = n%512 ? n%512+1 : n%512;
//	float S;
//
//  extern __shared__ float As[];
// 
//  if (j < n) {
//    //cuPrintf("%d,%d : %d,%d : %d,%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
//    for (int i = 0; i < n; i++) {
//      
//      __syncthreads();
//			for (int o = 0; o < m; o++) {
//      	int p = o*512+threadIdx.x;
//				if (p < i) { 
//        	As[p] = A.els[l+p];
//        	//cuPrintf("j=%d, i=%d,l+j=%d, As[l+j]=%f\n", j, i,l+j, As[j]);
//      	}
//				else break;
//			}
//      __syncthreads();
//      
//      S = alpha*B.els[i*n+j]; //S = B[i][j];
//      //cuPrintf("i=%d,j=%d, S=%f\n", i, j, S);
//      for (int k = 0; k < i; k++) {
//        //S -= A.els[i*n+k] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
//        //S -= A.els[l] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
//        S -= As[k] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
//        //cuPrintf("i=%d,j=%d,k=%d, S=%f, A=%f, C=%f\n", i, j, k, S, As[k], C.els[k*n+j]);
//      }
//      l += i;
//      C.els[i*n+j] = S; //C[i][j] = S;
//    }
//  }
//
//}

// this is now in matrix.h since it's shared
//void TruncateMatrix(Matrix A) {
//
//  int k = 0;
//  int n = A.width;
//  //int size = (n*n-n)/2;
//
//  for (int i = 0; i < n; i++) {
//    for (int j = 0; j < n; j++) {
//      if (i == j) continue;
//      if (j < i) {
//        //assert(k<size);
//        A.els[k] = A.els[i*n+j];
//        k++;
//      }
//    }
//  }
//}

// matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMultGPU(const Matrix A, const Matrix B, Matrix C, const float alpha)
{
	Matrix d_A, d_B, d_C;

  // Initialize 
  //cuInit(0); 
  
  // Get number of devices supporting CUDA 
  //int deviceCount = 0; 
  //cuDeviceGetCount(&deviceCount); 
  //if (deviceCount == 0) { 
  //  printf("There is no device supporting CUDA.\n"); exit (0); 
  //}
  //printf("deviceCount=%d\n", deviceCount);
  const int n = A.width;
	cudaError_t cudaMallocReturnStatus;
	struct timeval timerValues;
	double start_time, end_time;
	double before_kernel, after_kernel;
	timerclear(&timerValues);	

  cudaPrintfInit();

	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	//size_t size = A.width * A.height * sizeof(float);
	size_t asize = ((A.width * A.height - A.width)/2) * sizeof(float);
	cudaMalloc((void**)&d_A.els, asize);
	cudaMallocReturnStatus = cudaMalloc((void**)&d_A.els, asize);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("ERROR: Couldn't allocate Matrix A on GPU, exiting\n"); exit(0);
	}
  TruncateMatrix(A);
  //for (int i = 0; i < 3; i++) printf("A[%d] = %f\n", i, A.els[i]);
	//cudaMemcpy(d_A.els, A.els, size, cudaMemcpyHostToDevice);

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size_t size = B.width * B.height * sizeof(float);
	cudaMalloc((void**)&d_B.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("ERROR: Couldn't allocate Matrix B on GPU, exiting\n"); exit(0);
	}
	//cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice);

	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void**)&d_C.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("ERROR: Couldn't allocate Matrix C on GPU, exiting\n"); exit(0);
	}

  int threadsPerBlock = 512;
  int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;
	
	//Get start time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get start time of day\n");
	
	//if (timerisset(&timerValues)) 
	start_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	//printf("Start secs: %ld, Start usecs: %ld, Time: %f\n", timerValues.tv_sec, timerValues.tv_usec, start_time);
	
	cudaMemcpy(d_A.els, A.els, asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice);
	
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get before kernel time of day\n");
	
	before_kernel = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	
  //printf("grids=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
	MatMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, n);
	//MatMultKernelShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, n);
	//MatMultKernelShared<<<blocksPerGrid, threadsPerBlock, sizeof(float)*(n-1)>>>(d_A, d_B, d_C, alpha, n);
	//MatMultKernel<<<1, 3, sizeof(float)*(n-1)>>>(d_A, d_B, d_C, alpha, n);
	cudaThreadSynchronize();	
	
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get after kernel time of day\n");
	
	after_kernel = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	
	cudaMemcpy(C.els, d_C.els, size, cudaMemcpyDeviceToHost);
  
	//Get end time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get end time of day\n");
	
	//if (timerisset(&timerValues)) 
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


