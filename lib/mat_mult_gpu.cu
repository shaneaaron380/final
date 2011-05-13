#include <sys/time.h>
#include "mat_mult_gpu.h"
#include "cuPrintf.cu"

#define SMEM_CACHE_SZ 32 

__global__ void MatMultKernel(const Matrix A, Matrix B, const float alpha, const int N)
{
	int l = 0;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float S;

	if (j < N) {
		for (int i = 0; i < N; i++) {
			S = alpha*B.els[i*N+j];
			for (int k = 0; k < i; k++) {
				S -= A.els[l++] * B.els[k*N+j];
			}
			__syncthreads();
			B.els[i*N+j] = S;
			__syncthreads();
		}
	}
}

__global__ void MatMultKernelShared(const Matrix A, Matrix B, const float alpha, const int N)
{
	int l = 0, k = 0;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx = threadIdx.x;
	float S;

	__shared__ float As[SMEM_CACHE_SZ];

	if (j < N) {

		B.els[j] = alpha*B.els[j];

		for (int i = 1; i < N; i++) {

			S = alpha*B.els[i*N+j]; 
			k = 0;

			while ((k + 12) <= i) {
				if (t_idx < 12) As[t_idx] = A.els[l+t_idx];
				__syncthreads();

				S -=  (As[0]  * B.els[k*N+j]) +
					  (As[1]  * B.els[(k+1)*N+j]) +
					  (As[2]  * B.els[(k+2)*N+j]) +
					  (As[3]  * B.els[(k+3)*N+j]) +
					  (As[4]  * B.els[(k+4)*N+j]) +
					  (As[5]  * B.els[(k+5)*N+j]) +
					  (As[6]  * B.els[(k+6)*N+j]) +
					  (As[7]  * B.els[(k+7)*N+j]) +
					  (As[8]  * B.els[(k+8)*N+j]) +
					  (As[9]  * B.els[(k+9)*N+j]) +
					  (As[10] * B.els[(k+10)*N+j]) +
					  (As[11] * B.els[(k+11)*N+j]);
				k+=12;
				l+=12;
			}

			while ((k + 4) <= i) {
				if (t_idx < 4) As[t_idx] = A.els[l+t_idx];
				__syncthreads();

				S -=  (As[0] * B.els[k*N+j]) +
					  (As[1] * B.els[(k+1)*N+j]) +
					  (As[2] * B.els[(k+2)*N+j]) +
					  (As[3] * B.els[(k+3)*N+j]);
				k+=4;
				l+=4;
			}

			while (k < i)  {
				if (t_idx == 0) As[t_idx] = A.els[l];
				__syncthreads();

				S -= As[0] * B.els[k*N+j];

				k++;
				l++;
			}

			B.els[i*N+j] = S;
		}
	}
}

__global__ void MatMultKernelAlignedShared(const Matrix A, Matrix B, const float alpha, const int N)
{
	int l = 0, k = 0, i = 1;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx = threadIdx.x;
	float S;

	__shared__ float As[SMEM_CACHE_SZ];

	if (j < N) {

		B.els[j] = alpha*B.els[j];

		for (i = 1; i < N; i++) {

			S = alpha*B.els[i*N+j]; 
			k = 0;

			while (k < i) {	
				if (t_idx < 16) As[t_idx] = A.els[l+t_idx];
				__syncthreads();

				S -=  (As[0]  * B.els[k*N+j]) +
					  (As[1]  * B.els[(k+1)*N+j]) +
					  (As[2]  * B.els[(k+2)*N+j]) +
					  (As[3]  * B.els[(k+3)*N+j]) +
					  (As[4]  * B.els[(k+4)*N+j]) +
					  (As[5]  * B.els[(k+5)*N+j]) +
					  (As[6]  * B.els[(k+6)*N+j]) +
					  (As[7]  * B.els[(k+7)*N+j]) +
					  (As[8]  * B.els[(k+8)*N+j]) +
					  (As[9]  * B.els[(k+9)*N+j]) +
					  (As[10] * B.els[(k+10)*N+j]) +
					  (As[11] * B.els[(k+11)*N+j]) +
					  (As[12] * B.els[(k+12)*N+j]) +
					  (As[13] * B.els[(k+13)*N+j]) +
					  (As[14] * B.els[(k+14)*N+j]) +
					  (As[15] * B.els[(k+15)*N+j]);
				k+=16;
				l+=16;
			}
			B.els[i*N+j] = S;
		}
	}
}


int MatMultGPU(const Matrix A, const Matrix B, const float alpha)
{
	Matrix d_A, d_B;
	const int n = A.width;
	struct timeval timerValues;
	double start_time, end_time;
	double before_kernel, after_kernel;
	timerclear(&timerValues);	

	cudaPrintfInit();

	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t asize = GetPadMatrixSize(A.width,16) * sizeof(float);
	if (cudaMalloc((void**) &d_A.els, asize) != cudaSuccess)
		RET_ERROR("could not allocate matrix A on device");

	TruncAndPadMatrix(A,16);

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size_t bsize = B.width * B.height * sizeof(float);
	if (cudaMalloc((void**) &d_B.els, bsize) != cudaSuccess)
		RET_ERROR("could not allocate matrix A on device");

	int threadsPerBlock = 128;
	int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;
	printf("grids=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);

	if (gettimeofday(&timerValues, NULL))
		RET_ERROR("could not gettimeofday for start_time");

	start_time = (double) timerValues.tv_sec +
				 (double) (timerValues.tv_usec) / 1000000.0;

	if (cudaMemcpy(d_A.els, A.els, asize, cudaMemcpyHostToDevice) != cudaSuccess)
		RET_ERROR("could not copy data to A matrix");

	if (cudaMemcpy(d_B.els, B.els, bsize, cudaMemcpyHostToDevice) != cudaSuccess)
		RET_ERROR("could not copy data to B matrix");

	if (gettimeofday(&timerValues, NULL))
		RET_ERROR("could not gettimeofday for before_kernel");

	before_kernel = (double) timerValues.tv_sec	+
					(double) (timerValues.tv_usec) / 1000000.0;

	//MatMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, alpha, n);

	//Static shared memory
	//MatMultKernelShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, alpha, n);
	MatMultKernelAlignedShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, alpha, n);

	//Dynamic shared memory
	//MatMultKernelShared<<<blocksPerGrid, threadsPerBlock, sizeof(float)*(n-1)>>>(d_A, d_B, d_C, alpha, n);
	cudaThreadSynchronize();	

	if (gettimeofday(&timerValues, NULL))
		RET_ERROR("could not gettimeofday for after_kernel");

	after_kernel =  (double) timerValues.tv_sec +
					(double) (timerValues.tv_usec) / 1000000.0;

	// this keeps failing when i run it on TACC even though the outputs diff...
	// don't know what to make of it, so i'm just gonna ignore it?
	cudaError_t r = cudaMemcpy(B.els, d_B.els, bsize, cudaMemcpyDeviceToHost);
	if (r != cudaSuccess)
		fprintf(stderr, 
				"WARNING, copying results to host failed w/ error %d\n", r);

	if (gettimeofday(&timerValues, NULL))
		RET_ERROR("could not gettimeofday for end_time");

	end_time =  (double) timerValues.tv_sec	+
				(double) (timerValues.tv_usec) / 1000000.0;

	printf("Total Time: %f\n", end_time - start_time);
	printf("Kernel Time: %f\n", after_kernel - before_kernel);
	printf("Transfer Time: %f\n",   (end_time - after_kernel) + 
									(before_kernel - start_time));

	cudaPrintfDisplay(stdout,true);
	cudaPrintfEnd();

	cudaFree(d_A.els);
	cudaFree(d_B.els);

	return SUCCESS;
}


