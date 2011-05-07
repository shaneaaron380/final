
#include "mat_mult_gpu.h"
#include "cuPrintf.cu"
//#include "cuda.h"

__global__ void MatMultKernel(const Matrix A, const Matrix B, Matrix C, int n)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  //int j = blockIdx.y * blockDim.y + threadIdx.y;
  float S;
 
  cuPrintf("%d,%d : %d,%d : %d,%d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);

  if (j < n) {
    for (int i = 0; i < n; i++) {
      S = B.els[i*n+j]; //S = B[i][j];
      //cuPrintf("i=%d,j=%d, S=%f\n", i, j, S);
      for (int k = 0; k < i; k++) {
        S -= A.els[i*n+k] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
        //cuPrintf("i=%d,j=%d,k=%d, S=%f, A=%f, C=%f\n", i, j, k, S, A.els[i*n+k], C.els[k*n+j]);
      }
      C.els[i*n+j] = S; //C[i][j] = S;
    }
  }

}

// matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMultGPU(const Matrix A, const Matrix B, Matrix C)
{
	Matrix d_A, d_B, d_C;

  // Initialize 
  //cuInit(0); 
  //
  //// Get number of devices supporting CUDA 
  //int deviceCount = 0; 
  //cuDeviceGetCount(&deviceCount); 
  //if (deviceCount == 0) { 
  //  printf("There is no device supporting CUDA.\n"); exit (0); 
  //}
  int n = A.width;
	cudaError_t cudaMallocReturnStatus;

  cudaPrintfInit();

	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**)&d_A.els, size);
	cudaMallocReturnStatus = cudaMalloc((void**)&d_A.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("Couldn't allocate A on CPU, exiting\n"); exit(0);
	}
	cudaMemcpy(d_A.els, A.els, size, cudaMemcpyHostToDevice);

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**)&d_B.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("Couldn't allocate B on CPU, exiting\n"); exit(0);
	}
	cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice);

	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void**)&d_C.els, size);
	if (cudaMallocReturnStatus == cudaErrorMemoryAllocation) {
		printf("Couldn't allocate C on CPU, exiting\n"); exit(0);
	}

	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimBlock(A.width, A.width);
	//dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  //printf("dimBlock.x = %d, dimBlock.y = %d\n", dimBlock.x, dimBlock.y);
  int threadsPerBlock = 256;
  int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;
  printf("grids=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
	//MatMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, A.width);
	MatMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
  
  cudaPrintfDisplay(stdout,true);
  cudaPrintfEnd();

	cudaMemcpy(C.els, d_C.els, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A.els);
	cudaFree(d_B.els);
	cudaFree(d_C.els);
}


