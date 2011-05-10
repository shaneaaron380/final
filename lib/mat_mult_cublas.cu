#include "mat_mult_cublas.h"
#include "sys/time.h"

/*******************************************************************************
 * 
 * Solve the following:
 * 
 *	A * X = alpha * B
 *	    X = alpha * A^(-1) * B
 * 
 ******************************************************************************/

#define N 2

int MatMultCublas(const Matrix A, Matrix B, const float alpha)
{
	int r;
	struct timeval timerValues;
	double start_time, end_time;
	double before_kernel, after_kernel;
	timerclear(&timerValues);	
	
	if (cublasInit() != CUBLAS_STATUS_SUCCESS)
		RET_ERROR("cublasInit failed");

	Matrix d_A, d_B;

	r = cublasAlloc(A.width*A.height, sizeof(A.els[0]), (void**) &d_A.els);
	if (r == CUBLAS_STATUS_INVALID_VALUE) {
		RET_ERROR("failed to allocate space for A b/c of an invalid value");
	} else if (r == CUBLAS_STATUS_ALLOC_FAILED) {
		RET_ERROR("failed to allocate space for A b/c of alloc failed");
	} else if (r != CUBLAS_STATUS_SUCCESS) {
		RET_ERROR("failed to allocate space for A");
	}

	r = cublasAlloc(B.width*B.height, sizeof(B.els[0]), (void**) &d_B.els);
	if (r == CUBLAS_STATUS_INVALID_VALUE) {
		RET_ERROR("failed to allocate space for B b/c of an invalid value");
	} else if (r == CUBLAS_STATUS_ALLOC_FAILED) {
		RET_ERROR("failed to allocate space for B b/c of alloc failed");
	} else if (r != CUBLAS_STATUS_SUCCESS) {
		RET_ERROR("failed to allocate space for B");
	}
	
	//Get start time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get start time of day\n");
	
	//if (timerisset(&timerValues)) 
	start_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	//printf("Start secs: %ld, Start usecs: %ld, Time: %f\n", timerValues.tv_sec, timerValues.tv_usec, start_time);

	r = cudaMemcpy(d_A.els, A.els, A.width*A.height*sizeof(float),
			cudaMemcpyHostToDevice);
	if (r != cudaSuccess)
		RET_ERROR("could not copy data to d_A");
	r = cudaMemcpy(d_B.els, B.els, B.width*B.height*sizeof(float),
			cudaMemcpyHostToDevice);
	if (r != cudaSuccess)
		RET_ERROR("could not copy data to d_B");
	
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get before kernel time of day\n");
	
	before_kernel = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;

	cublasStrsm('l',		/* side: a is on the left side of B (and this X) */
				'l',		/* uplo: lower triangular */
				'n',		/* transa: don't transpose */
				'u',		/* diag: unit diagonal */
				B.height,	/* m: number of rows in B, and since 'l', it's also
							   the order of A */
				B.width,	/* n: number of columns in B */
				alpha,		/* alpha: alpha scalar */
				d_A.els,	/* a: 'A' matrix */
				A.height,	/* lda -- ??? */
				d_B.els,	/* b: 'B' matrix */
				B.height	/* ldb -- ??? */);
	
	cudaThreadSynchronize();	
	
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get after kernel time of day\n");
	
	after_kernel = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	
	r = cudaMemcpy(B.els, d_B.els, B.height*B.width*sizeof(float),
			cudaMemcpyDeviceToHost);
	if (r != cudaSuccess)
		RET_ERROR("could not copy data from d_B");
	
	//Get end time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get end time of day\n");
	
	//if (timerisset(&timerValues)) 
	end_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	//printf("End secs: %ld, End usecs: %ld, Total Time: %f\n", timerValues.tv_sec, timerValues.tv_usec, end_time-start_time);
	printf("Total Time: %f\n", end_time-start_time);
	printf("Kernel Time: %f\n", after_kernel-before_kernel);
	printf("Transfer Time: %f\n", (end_time-after_kernel)+(before_kernel-start_time));

	int e = cublasGetError();
	if (e == CUBLAS_STATUS_NOT_INITIALIZED) {
		RET_ERROR("CUBLAS_STATUS_NOT_INITIALIZED")
	} else if (e == CUBLAS_STATUS_INVALID_VALUE) {
		RET_ERROR("CUBLAS_STATUS_INVALID_VALUE");
	} else if (e == CUBLAS_STATUS_EXECUTION_FAILED) {
		RET_ERROR("CUBLAS_STATUS_EXECUTION_FAILED");
	}

	return SUCCESS;
}
