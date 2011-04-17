#include "mat_mult_from_doc.h"

__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.els[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
	A.els[row * A.stride + col] = value;
}

// return a BLOCK_SIZE x BLOCK_SIZE sub-matrix of A that is 'row' sub matrices
// from the top and 'col' sub matrices from the left
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.els = &A.els[row * A.stride * BLOCK_SIZE + col * BLOCK_SIZE];

	return Asub;
}

__global__ void MatMultKernel(const Matrix, const Matrix, Matrix);

// matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMult(const Matrix A, const Matrix B, Matrix C)
{
	Matrix d_A, d_B, d_C;

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
	MatMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(C.els, d_C.els, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A.els);
	cudaFree(d_B.els);
	cudaFree(d_C.els);
}

__global__ void MatMultKernel(const Matrix A, const Matrix B, Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	float Cvalue = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		__syncthreads();
	}

	SetElement(Csub, row, col, Cvalue);
}

