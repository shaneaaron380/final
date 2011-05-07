
#include "mat_mult_gpu.h"
#include "cuPrintf.cu"

__global__ void MatMultKernel(const Matrix A, const Matrix B, Matrix C, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //int j = blockIdx.y * blockDim.y + threadIdx.y;
  float S;

  cuPrintf("Enter kernel with thread: %d\n", i);

  for (int j = 0; j < n; j++) {
    S = B.els[i*n+j]; //S = B[i][j];
    cuPrintf("i=%d,j=%d, S=%f\n", i, j, S);
    for (int k = 0; k < i; k++) {
      S -= A.els[i*n+k] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
      cuPrintf("k=%d, S=%f\n", k, S);
    }
    C.els[i*n+j] = S; //C[i][j] = S;
  }

}

// matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMultGPU(const Matrix A, const Matrix B, Matrix C)
{
	Matrix d_A, d_B, d_C;

  cudaPrintfInit();

	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**)&d_A.els, size);
	cudaMemcpy(d_A.els, A.els, size, cudaMemcpyHostToDevice);

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**)&d_B.els, size);
	cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice);

	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void**)&d_C.els, size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, A.width);
  
  cudaPrintfDisplay(stdout,true);
  cudaPrintfEnd();

	cudaMemcpy(C.els, d_C.els, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A.els);
	cudaFree(d_B.els);
	cudaFree(d_C.els);
}


