#include <stdio.h>

#include "mat_mult_cublas.h"

void Usage(int retVal, char *argv0)
{
	fprintf(retVal == 0? stdout : stderr,
			"USAGE: %s <matrix A input file> <matrix B input file> "
			"[<output matrix file>] <alpha> <S,G,C>\n", argv0);

	exit(retVal);
}

int GetInputs(int argc, char *argv[], Matrix *a, Matrix *b, float *alpha, char
		*which)
{
	if (argc < 5)
		RET_ERROR("must have 4 or 5 cmd line args");

	if (MatrixFromFile(argv[1], a, MATRIX_FILE_TRANSPOSE) != SUCCESS)
		RET_ERROR("could not read matrix A");

	if (MatrixFromFile(argv[2], b, MATRIX_FILE_TRANSPOSE) != SUCCESS)
		RET_ERROR("could not read matrix B");

	*alpha = strtof(argv[3], (char**)NULL);

	*which = argv[4][0];

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A, B;
	float alpha;
	char which;

	if (GetInputs(argc, argv, &A, &B, &alpha, &which) != SUCCESS)
		Usage(1, argv[0]);

	MatMultCublas(A, B);

	if (argc >= 6) {
		if (MatrixToFile(argv[5], &B, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	} else {
		if (MatrixToFile("-", &B, MATRIX_FILE_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	}

	return 0;
}

