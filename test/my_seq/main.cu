
#include "mat_mult_seq.h"

void Usage(int retVal, char *argv0)
{
	fprintf(retVal == 0? stdout : stderr,
			"USAGE: %s <matrix A input file> <matrix B input file> "
			"<output matrix file> <alpha> <S,G,C> [o]\n", argv0);

	exit(retVal);
}

int GetInputs(int argc, char *argv[], Matrix *a, Matrix *b, float *alpha, char
		*which, int *useOldFormat)
{
	*useOldFormat = 0;

	if (argc < 6)
		RET_ERROR("must have at least 5 cmd line args");

	if (argc > 6 && strncmp(argv[6], "o", 2) == 0)
		*useOldFormat = 1;

	if (useOldFormat) {
		if (MatrixFromFile(argv[1], a, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromFile(argv[2], b, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix B");
	} else {
		if (MatrixFromCOOFile(argv[1], a, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromCOOFile(argv[2], b, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix B");
	}

	*alpha = strtof(argv[3], (char**)NULL);

	*which = argv[4][0];

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A, B, X;
	float alpha;
	char which;
	int useOldFormat;

	if (GetInputs(argc, argv, &A, &B, &alpha, &which, &useOldFormat) != SUCCESS)
		Usage(1, argv[0]);

	X.width = A.width;
	X.height = A.height;
	if (! (X.els = (float*) malloc(X.width * X.height * sizeof(X.els[0]))))
		RET_ERROR("could not allocate memory for elements from test seq main");

	MatMultSeq(&A, &B, &X, alpha); 

	if (useOldFormat) {
		if (MatrixToFile(argv[5], &X, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	} else {
		if (MatrixToCOOFile(argv[5], &X, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	}

	free(A.els);
	free(B.els);
	free(X.els);

	//double ** A = (double **) malloc(n*sizeof (double *) );
	//double ** B = (double **) malloc(n*sizeof (double *) );
	//double ** X = (double **) malloc(n*sizeof (double *) );
	//
	//for (i = 0; i < n; i++ )
	//  A[i] = (double *) malloc(n*sizeof(double));

	//for (i = 0; i < n; i++ )
	//  B[i] = (double *) malloc(n*sizeof(double));
	//
	//for (i = 0; i < n; i++ )
	//  X[i] = (double *) malloc(n*sizeof(double));

	//printf("A=\n");
	//for (i = 0; i < n; i++) {
	//  for (j = 0; j < n; j++) {
	//    if (i == j)
	//      A[i][j] = 1;
	//    if (i > j)
	//      A[i][j] = 0;
	//    if (j < i)
	//      A[i][j] = 5;
	//    printf("%f ", A[i][j]);
	//  }
	//  printf("\n");
	//}
	//printf("\nB=\n");
	//for (i = 0; i < n; i++) {
	//  for (j = 0; j < n; j++) {
	//    B[i][j] = 3;
	//    printf("%f ", B[i][j]);
	//  }
	//  printf("\n");
	//}
	//printf("\n");


	//printf("\nX=\n");
	//for (i = 0; i < n; i++) {
	//  for (j = 0; j < n; j++) {
	//    printf("%f ", X[i][j]);
	//  }
	//  printf("\n");
	//}

	//for (i = 0; i < n; i++) {
	//  free(A[i]);
	//  free(B[i]);
	//  free(X[i]);
	//}
	//free(A);
	//free(B);
	//free(X);

	return 0;
}
