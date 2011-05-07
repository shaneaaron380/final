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

int MatMultCublas(const Matrix A, Matrix B)
{
	if (cublasInit() != CUBLAS_STATUS_SUCCESS)
		RET_ERROR("cublasInit failed");

	Matrix d_A, d_B;

	cublasAlloc(A.width*A.height, sizeof(float), (void**) &d_A.els);
	cublasAlloc(B.width*B.height, sizeof(float), (void**) &d_B.els);

	cudaMemcpy(d_A.els, A.els, A.width*A.height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.els, B.els, B.width*B.height*sizeof(float), cudaMemcpyHostToDevice);

	cublasStrsm('l',		/* side: a is on the left side of B (and this X) */
				'l',		/* uplo: lower triangular */
				'n',		/* transa: don't transpose */
				'u',		/* diag: unit diagonal */
				B.height,	/* m: number of rows in B, and since 'l', it's also
							   the order of A */
				B.width,	/* n: number of columns in B */
				1.0,		/* alpha: alpha scalar */
				d_A.els,	/* a: 'A' matrix */
				A.height,	/* lda -- ??? */
				d_B.els,	/* b: 'B' matrix */
				B.height	/* ldb -- ??? */);

	cudaMemcpy(B.els, d_B.els, B.height*B.width*sizeof(float), cudaMemcpyDeviceToHost);

	int e = cublasGetError();
	if (e == CUBLAS_STATUS_NOT_INITIALIZED) {
		fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");
	} else if (e == CUBLAS_STATUS_INVALID_VALUE) { 
		fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");
	} else if (e == CUBLAS_STATUS_EXECUTION_FAILED) {
		fprintf(stderr, "CUBLAS_STATUS_EXECUTION_FAILED\n");
	}

	return 0;
}
