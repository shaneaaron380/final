#include "mat_mult_shared.h"
#include "sys/time.h"
#include "cuPrintf.cu"

#define COLS_IN_SHMEM 4

__global__ void MatMultKernelS(const Matrix A, const Matrix B, Matrix C, const float alpha, int n)
{
	int l = 0;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float S;

	/*cuPrintf("(%d %d %d), (%d %d %d), (%d %d %d)\n", */
	/*        blockIdx.x,*/
	/*        blockIdx.y,*/
	/*        blockIdx.z,*/
	/*        blockDim.x,*/
	/*        blockDim.y,*/
	/*        blockDim.z,*/
	/*        threadIdx.x,*/
	/*        threadIdx.y,*/
	/*        threadIdx.z);*/

	__shared__ float As[COLS_IN_SHMEM];
	/*cuPrintf("%d\n", j);*/

	if (j < n) {
		for (int i = 0; i < n; i++) {
			S = alpha*B.els[i*n+j];
			for (int k = 0; k < i; k++) {

				if (k % COLS_IN_SHMEM == 0 && l + j % COLS_IN_SHMEM < n * (n + 1) / 2) {
					As[j%COLS_IN_SHMEM] = A.els[l + j % COLS_IN_SHMEM];
				} else {
					/*cuPrintf("%d %d %d\n", l, j % COLS_IN_SHMEM, n * (n + 1) / 2);*/
				}
					if (A.els[l + j%COLS_IN_SHMEM] >= 4) {
						cuPrintf("%d < %d\n", j%COLS_IN_SHMEM, l + j%COLS_IN_SHMEM);
					}
				__syncthreads();

				/*S -= A.els[l] * C.els[k*n+j];*/
				/*S -= As[k%COLS_IN_SHMEM] * C.els[k*n+j];*/
				S -= As[l%COLS_IN_SHMEM] * C.els[k*n+j];
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

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			fprintf(stderr, "%f ", A.els[i * n + j]);
		fprintf(stderr, "\n");
	}

	TruncateMatrix(A);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			fprintf(stderr, "%f ", A.els[i * n + j]);
		fprintf(stderr, "\n");
	}

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

	/*int threadsPerBlock = 512;*/
	/*int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;*/
	int blocksPerGrid = (n+COLS_IN_SHMEM-1)/COLS_IN_SHMEM;

	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get start time of day\n");

	start_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;

	if (cudaMemcpy(d_A.els, A.els, asize, cudaMemcpyHostToDevice) != cudaSuccess)
		RET_ERROR("could not copy A matrix to device");
	if (cudaMemcpy(d_B.els, B.els, size, cudaMemcpyHostToDevice) != cudaSuccess)
		RET_ERROR("could not copy B matrix to device");

	MatMultKernelS<<<blocksPerGrid, COLS_IN_SHMEM>>>(d_A, d_B, d_C, alpha, n);

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



