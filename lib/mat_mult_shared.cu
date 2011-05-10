#include "mat_mult_shared.h"
#include "sys/time.h"
#include "cuPrintf.cuh"

__global__ void MatMultKernelS(const Matrix A, const Matrix B, Matrix C, const float alpha, int n)
{
	int l = 0;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float S;

	if (j < n) {
		for (int i = 0; i < n; i++) {
			S = alpha*B.els[i*n+j];
			for (int k = 0; k < i; k++) {
				S -= A.els[l] * C.els[k*n+j]; //S -= A[i][k] * C[k][j];
				l++;
			}
			C.els[i*n+j] = S;
		}
	}
}


int MatMultShared(const Matrix A, const Matrix B, Matrix C, const float alpha)
{
	Matrix d_A, d_B, d_C;

	const int n = A.width;
	struct timeval timerValues;
	double start_time, end_time;
	timerclear(&timerValues);	

	cudaPrintfInit();

	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t asize = ((A.width * A.height - A.width)/2) * sizeof(float);
	cudaMalloc((void**)&d_A.els, asize);
	if (cudaMalloc((void**) &d_A.els, asize) != cudaSuccess)
		RET_ERROR("could not allocate matrix A on device");
	TruncateMatrix(A);

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size_t size = B.width * B.height * sizeof(float);
	if (cudaMalloc((void**) &d_B.els, size) != cudaSuccess)
		RET_ERROR("could not allocate matrix A on device");

	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	if (cudaMalloc((void**) &d_C.els, size) != cudaSuccess)
		RET_ERROR("could not allocate matrix A on device");

	int threadsPerBlock = 512;
	int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;

	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get start time of day\n");

	start_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;

	cudaMemcpy(d_A.els, A.els, asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice);

	MatMultKernelS<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, n);

	cudaMemcpy(C.els, d_C.els, size, cudaMemcpyDeviceToHost);
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get end time of day\n");

	end_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	printf("Total Time: %f\n", end_time-start_time);

	cudaPrintfDisplay(stdout,true);
	cudaPrintfEnd();

	cudaFree(d_A.els);
	cudaFree(d_B.els);
	cudaFree(d_C.els);

	return SUCCESS;
}



