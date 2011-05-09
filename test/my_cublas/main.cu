#include <stdio.h>

#include "mat_mult_cublas.h"

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
		RET_ERROR("must have 5 or 6 cmd line args");

	if (argc > 6 && strncmp(argv[6], "o", 2) == 0)
		*useOldFormat = 1;

	if (*useOldFormat) {
		if (MatrixFromFile(argv[1], a, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromFile(argv[2], b, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix B");
	} else {
		if (MatrixFromCOOFile(argv[1], a, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromCOOFile(argv[2], b, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix B");
	}

	*alpha = strtof(argv[3], (char**)NULL);

	*which = argv[4][0];

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A, B;
	float alpha;
	char which;
	int useOldFormat;

	if (GetInputs(argc, argv, &A, &B, &alpha, &which, &useOldFormat) != SUCCESS)
		Usage(1, argv[0]);

	if (MatMultCublas(A, B, alpha) != SUCCESS)
		RET_ERROR("MatMultCublas failed");

	if (useOldFormat) {
		if (MatrixToFile(argv[5], &B, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	} else {
		if (MatrixToCOOFile(argv[5], &B, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	}

	return 0;
}

