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
	/*float A[N][N] =   { { 3.0, -1.0 },*/
	/*                    { 0.0, -2.0 } },*/

	/*      B[N][N] =   { { 1.0, 1.0 },*/
	/*                    { 1.0, 1.0 } };*/

	/*float A[7][5] =   { { 3.0, -1.0,  2.0,  2.0,  1.0 },*/
	/*                    { 0.0, -2.0,  4.0, -1.0,  3.0 },*/
	/*                    { 0.0,  0.0, -3.0,  0.0,  2.0 },*/
	/*                    { 0.0,  0.0,  0.0,  4.0, -2.0 },*/
	/*                    { 0.0,  0.0,  0.0,  0.0,  1.0 },*/
	/*                    { 0.0,  0.0,  0.0,  0.0,  0.0 },*/
	/*                    { 0.0,  0.0,  0.0,  0.0,  0.0 } },*/

	/*float A[5][7] =   { {  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },*/
	/*                    { -1.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0 },*/
	/*                    {  2.0,  4.0, -3.0,  0.0,  0.0,  0.0,  0.0 },*/
	/*                    {  2.0, -1.0,  0.0,  4.0,  0.0,  0.0,  0.0 },*/
	/*                    {  1.0,  3.0,  2.0, -2.0,  1.0,  0.0,  0.0 } },*/

		  /*B[6][3] =   { {   6.0, 10.0,  -2.0 },*/
		  /*              { -16.0, -1.0,   6.0 },*/
		  /*              {  -2.0,  1.0,  -4.0 },*/
		  /*              {  14.0,  0.0, -14.0 },*/
		  /*              {  -1.0,  2.0,   1.0 },*/
		  /*              {   0.0,  0.0,   0.0 } };*/

		  /*B[3][6] =   { {   6.0, -16.0,  -2.0,  14.0,  -1.0,   0.0 },*/
		  /*              {  10.0,  -1.0,   1.0,   0.0,   2.0,   0.0 },*/
		  /*              {  -2.0,   6.0,  -4.0, -14.0,   1.0,   0.0 } };*/

	Matrix d_A, d_B;

	d_A.height = A.height;
	d_A.width = A.width;
	int A_size = d_A.height * d_A.width * sizeof(float);
	cudaMalloc( (void**) &d_A.els, A_size);
	cudaMemcpy(d_A.els, A.els, A_size, cudaMemcpyHostToDevice);

	d_B.height = B.height;
	d_B.width = B.width;
	int B_size = d_B.height * d_B.width * sizeof(float);
	cudaMalloc((void**)&d_B.els, B_size);
	cudaMemcpy(d_B.els, B.els, B_size, cudaMemcpyHostToDevice);

	cublasStrsm('l',		/* side: a is on the left side of B (and this X) */
				'u',		/* uplo: upper triangular */
				'n',		/* transa: don't transpose */
				'n',		/* diag: NOT unit diagonal */
				5,			/* m: number of rows in B, and since 'l', it's also
							   the order of A */
				3,			/* n: number of columns in B */
				1.0,		/* alpha: alpha scalar */
				/*(float*) A,	[> a: 'A' matrix <]*/
				d_A.els,	/* a: 'A' matrix */
				5,			/* lda -- ??? */
				/*(float*) B,	[> b: 'B' matrix <]*/
				d_B.els,	/* b: 'B' matrix */
				5			/* ldb -- ??? */);

	cudaMemcpy(B.els, d_B.els, B_size, cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < B.height; ++i) {*/
	/*    for (int j = 0; j < B.width; ++j) {*/

	/*        printf("%5.1f ", B.els[i * B.height + j]);*/
	/*    }*/
	/*    printf("\n");*/
	/*}*/

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
