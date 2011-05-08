#include "mat_mult_cublas.h"

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

	r = cudaMemcpy(d_A.els, A.els, A.width*A.height*sizeof(float),
			cudaMemcpyHostToDevice);
	if (r != cudaSuccess)
		RET_ERROR("could not copy data to d_A");
	r = cudaMemcpy(d_B.els, B.els, B.width*B.height*sizeof(float),
			cudaMemcpyHostToDevice);
	if (r != cudaSuccess)
		RET_ERROR("could not copy data to d_B");

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

	r = cudaMemcpy(B.els, d_B.els, B.height*B.width*sizeof(float),
			cudaMemcpyDeviceToHost);
	if (r != cudaSuccess)
		RET_ERROR("could not copy data from d_B");

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
