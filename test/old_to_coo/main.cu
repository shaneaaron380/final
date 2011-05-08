#include <stdio.h>

#include "mat_mult_cublas.h"

void Usage(int retVal, char *argv0)
{
	fprintf(retVal == 0? stdout : stderr,
			"USAGE: %s <matrix A input file> [t]\n", argv0);

	exit(retVal);
}

int GetInputs(int argc, char *argv[], Matrix *a, int *trans)
{
	*trans = MATRIX_FILE_NO_TRANSPOSE;

	if (argc < 2)
		RET_ERROR("must have at least 2 cmd line args");

	if (argc > 2 && strncmp(argv[2], "t", 2) == 0)
		*trans = MATRIX_FILE_TRANSPOSE;

	if (MatrixFromFile(argv[1], a, *trans) != SUCCESS)
		RET_ERROR("could not read matrix");

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A;
	int trans;

	if (GetInputs(argc, argv, &A, &trans) != SUCCESS)
		Usage(1, argv[0]);

	if (MatrixToCOOFile("-", &A, trans) != SUCCESS)
		RET_ERROR("could not write result matrix to stdout");

	return 0;
}

