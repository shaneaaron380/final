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

	if (cublasInit() != CUBLAS_STATUS_SUCCESS)
		RET_ERROR("cublasInit failed");

	Matrix d_A, d_B;

#if 0
	printf("using static, NON-transposed matrices\n");

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
#elif 0
	printf("using static, transposed matrices\n");

	float As[5][7] =  { {  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
						{ -1.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
						{  2.0,  4.0, -3.0,  0.0,  0.0,  0.0,  0.0 },
						{  2.0, -1.0,  0.0,  4.0,  0.0,  0.0,  0.0 },
						{  1.0,  3.0,  2.0, -2.0,  1.0,  0.0,  0.0 } };

	float Bs[3][6] =  { {   6.0, -16.0,  -2.0,  14.0,  -1.0,   0.0 },
						{  10.0,  -1.0,   1.0,   0.0,   2.0,   0.0 },
						{  -2.0,   6.0,  -4.0, -14.0,   1.0,   0.0 } };

	cublasAlloc(5*7, sizeof(float), (void**) &d_A.els);
	cublasAlloc(3*6, sizeof(float), (void**) &d_B.els);

	cudaMemcpy(d_A.els, As, 5*7*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.els, Bs, 6*3*sizeof(float), cudaMemcpyHostToDevice);

#else
	printf("using static, transposed matrices without padding\n");

	float As[5][5] =  { {  3.0,  0.0,  0.0,  0.0,  0.0},
						{ -1.0, -2.0,  0.0,  0.0,  0.0},
						{  2.0,  4.0, -3.0,  0.0,  0.0},
						{  2.0, -1.0,  0.0,  4.0,  0.0},
						{  1.0,  3.0,  2.0, -2.0,  1.0} };

	float Bs[3][5] =  { {   6.0, -16.0,  -2.0,  14.0,  -1.0},
						{  10.0,  -1.0,   1.0,   0.0,   2.0},
						{  -2.0,   6.0,  -4.0, -14.0,   1.0} };

	cublasAlloc(5*5, sizeof(float), (void**) &d_A.els);
	cublasAlloc(3*5, sizeof(float), (void**) &d_B.els);

	/*fprintf(stderr, "------------------------------ before mult:\n");*/
	/*for (int i = 0; i < 5; ++i) {*/
	/*    for (int j = 0; j < 3; ++j) {*/
	/*        fprintf(stderr, "%5.1lf ", ((float *) Bs)[5 * j + i]);*/
	/*    }*/
	/*    fprintf(stderr, "\n");*/
	/*}*/
	/*fprintf(stderr, "------------------------------ B matrix\n");*/
	/*for (int i = 0; i < B.height; ++i) {*/
	/*    for (int j = 0; j < B.width; ++j) {*/
	/*        fprintf(stderr, "%5.1lf ", ((float *) B.els)[B.height * j + i]);*/
	/*    }*/
	/*    fprintf(stderr, "\n");*/
	/*}*/
	/*fprintf(stderr, "------------------------------\n");*/

	cudaMemcpy(d_A.els, As, 5*5*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.els, Bs, 5*3*sizeof(float), cudaMemcpyHostToDevice);
	/*cudaMemcpy(d_B.els, B.els, B.width*B.height*sizeof(float), cudaMemcpyHostToDevice);*/

#endif

	cublasStrsm('l',		/* side: a is on the left side of B (and this X) */
				'u',		/* uplo: upper triangular */
				'n',		/* transa: don't transpose */
				'n',		/* diag: NOT unit diagonal */
				5,			/* m: number of rows in B, and since 'l', it's also
							   the order of A */
				3,			/* n: number of columns in B */
				1.0,		/* alpha: alpha scalar */
				d_A.els,	/* a: 'A' matrix */
				5,			/* lda -- ??? */
				d_B.els,	/* b: 'B' matrix */
				5			/* ldb -- ??? */);

	cudaMemcpy(Bs, d_B.els, 5*3*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 3; ++j) {
			fprintf(stderr, "%5.1lf ", ((float *) Bs)[5 * j + i]);
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
