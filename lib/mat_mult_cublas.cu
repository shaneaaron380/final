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
	cublasStatus status = CUBLAS_STATUS_SUCCESS;

	/*float A[N][N] =   { { 3.0, -1.0 },*/
	/*                    { 0.0, -2.0 } },*/

	/*      B[N][N] =   { { 1.0, 1.0 },*/
	/*                    { 1.0, 1.0 } };*/

#if 0
	float As[7][5] =  { { 3.0, -1.0,  2.0,  2.0,  1.0 },
						{ 0.0, -2.0,  4.0, -1.0,  3.0 },
						{ 0.0,  0.0, -3.0,  0.0,  2.0 },
						{ 0.0,  0.0,  0.0,  4.0, -2.0 },
						{ 0.0,  0.0,  0.0,  0.0,  1.0 },
						{ 0.0,  0.0,  0.0,  0.0,  0.0 },
						{ 0.0,  0.0,  0.0,  0.0,  0.0 } };

	float Bs[6][3] =  { {   6.0, 10.0,  -2.0 },
						{ -16.0, -1.0,   6.0 },
						{  -2.0,  1.0,  -4.0 },
						{  14.0,  0.0, -14.0 },
						{  -1.0,  2.0,   1.0 },
						{   0.0,  0.0,   0.0 } };
#else
	float As[5][7] =  { {  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
						{ -1.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
						{  2.0,  4.0, -3.0,  0.0,  0.0,  0.0,  0.0 },
						{  2.0, -1.0,  0.0,  4.0,  0.0,  0.0,  0.0 },
						{  1.0,  3.0,  2.0, -2.0,  1.0,  0.0,  0.0 } };

	float Bs[3][6] =  { {   6.0, -16.0,  -2.0,  14.0,  -1.0,   0.0 },
						{  10.0,  -1.0,   1.0,   0.0,   2.0,   0.0 },
						{  -2.0,   6.0,  -4.0, -14.0,   1.0,   0.0 } };
#endif

	if (cublasInit() != CUBLAS_STATUS_SUCCESS)
		RET_ERROR("cublasInit failed");

	Matrix d_A, d_B;

	cublasAlloc(5*7, sizeof(float), (void**) &d_A.els);
	cublasAlloc(3*6, sizeof(float), (void**) &d_B.els);

	cudaMemcpy(d_A.els, As, 5*7*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_B.els, Bs, 6*3*sizeof(float), cudaMemcpyHostToDevice);

	cublasStrsm('l',		/* side: a is on the left side of B (and this X) */
				'u',		/* uplo: upper triangular */
				'n',		/* transa: don't transpose */
				'n',		/* diag: NOT unit diagonal */
				5,			/* m: number of rows in B, and since 'l', it's also
							   the order of A */
				3,			/* n: number of columns in B */
				1.0,		/* alpha: alpha scalar */
				d_A.els,	/* a: 'A' matrix */
				7,			/* lda -- ??? */
				d_B.els,	/* b: 'B' matrix */
				6			/* ldb -- ??? */);

	cudaMemcpy(Bs, d_B.els, 6*3*sizeof(float), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < 3; ++i) {*/
	/*    for (int j = 0; j < 6; ++j) {*/
	/*        fprintf(stderr, "%5.1f ", Bs[i][j]);*/
	/*    }*/
	/*    fprintf(stderr, "\n");*/
	/*}*/
	/*fprintf(stderr, "\n");*/

	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 3; ++j) {
			fprintf(stderr, "%5.1lf ", ((float *) Bs)[6 * j + i]);
		}
		fprintf(stderr, "\n");
	}

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
