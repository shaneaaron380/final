#include <stdio.h>

//#include "mat_mult_from_doc.h"
#include "mat_mult_cublas.h"
#include "mat_mult_gpu.h"
#include "mat_mult_seq.h"

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

	if (MatrixFromFile(argv[1], a) != SUCCESS)
		RET_ERROR("could not read matrix A");

	if (MatrixFromFile(argv[2], b) != SUCCESS)
		RET_ERROR("could not read matrix B");

	*alpha = strtof(argv[3], (char**)NULL);

	*which = argv[4][0];

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A, B, C;
	float alpha;
	char which;

	if (GetInputs(argc, argv, &A, &B, &alpha, &which) != SUCCESS)
		Usage(1, argv[0]);

	C.height = A.height;
	C.width = B.width;
	C.stride = C.width;
	if (! (C.els = (float*) malloc(A.height * B.width * sizeof(C.els[0]))))
		RET_ERROR("could not allocate space for results matrix");

	MatMultGPU(A, B, C);

	if (MatrixToFile(argv[5], &C) != SUCCESS)
		RET_ERROR("could not write result matrix to %s", argv[5]);

	return 0;
}
