#include <stdio.h>

#include "mat_mult_cublas.h"

void Usage(int retVal, char *argv0)
{
	fprintf(retVal == 0? stdout : stderr,
			"USAGE: %s <matrix A input file> <matrix B input file> "
			"<output matrix file> <alpha> <S,G,C>\n", argv0);

	exit(retVal);
}

int GetInputs(int argc, char *argv[], Matrix *a, Matrix *b, float *alpha, char
		*which)
{
	if (argc != 6)
		RET_ERROR("must have 5 cmd line args");

	if (MatrixFromFile_T(argv[1], a) != SUCCESS)
		RET_ERROR("could not read matrix A");

	if (MatrixFromFile_T(argv[2], b) != SUCCESS)
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

	if (MatrixToFile(argv[5], &B) != SUCCESS)
		RET_ERROR("could not write result matrix to %s", argv[5]);

	return 0;
}

