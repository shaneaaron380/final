#include <stdio.h>

//#include "mat_mult_from_doc.h"
#include "mat_mult_cublas.h"
#include "mat_mult_gpu.h"
#include "mat_mult_seq.h"
#include "mat_mult_shared.h"

void Usage(int retVal, char *argv0)
{
	fprintf(retVal == 0? stdout : stderr,
			"USAGE: %s <matrix A input file> <matrix B input file> "
			"<alpha> <S,G,C> <output matrix file> [o]\n", argv0);

	exit(retVal);
}

int GetInputs(int argc, char *argv[], Matrix *a, Matrix *b, float *alpha, char
		*which, int *useOldFormat)
{
	int trans = MATRIX_FILE_NO_TRANSPOSE;

	*which = argv[4][0];
	if (*which == 'C')
		trans = MATRIX_FILE_TRANSPOSE;

	if (argc < 6)
		RET_ERROR("must have at least 5 cmd line args");

	*useOldFormat = 0;
	if (argc > 6 && strncmp(argv[6], "o", 2) == 0)
		*useOldFormat = 1;

	if (*useOldFormat) {
		if (MatrixFromFile(argv[1], a, trans) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromFile(argv[2], b, trans) != SUCCESS)
			RET_ERROR("could not read matrix B");
	} else {
		if (MatrixFromCOOFile(argv[1], a, trans) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromCOOFile(argv[2], b, trans) != SUCCESS)
			RET_ERROR("could not read matrix B");
	}

	*alpha = strtof(argv[3], (char**)NULL);

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A, B, C;
	Matrix *out;
	float alpha;
	char which;
	int useOldFormat;
	int trans = MATRIX_FILE_NO_TRANSPOSE;

	if (GetInputs(argc, argv, &A, &B, &alpha, &which, &useOldFormat) != SUCCESS)
		Usage(1, argv[0]);

	if (which == 'C') {
		printf("Using CUBLAS implementation\n");
		out = &B;
		trans = MATRIX_FILE_TRANSPOSE;
		if (MatMultCublas(A, B, alpha) != SUCCESS)
			RET_ERROR("MatMultCublas failed");

	} else {
		if (which == 'G') {
			printf("Using GPU implementation\n");
			out = &B;
			if (MatMultGPU(A, B, alpha) != SUCCESS)
				RET_ERROR("MatMultGPU failed");

		} else {
			out = &C;
			C.height = A.height;
			C.width = B.width;
			C.stride = C.width;
			cudaError_t r;
			r = cudaMallocHost(&C.els, C.width * C.height * sizeof(C.els[0]));
			if (r != cudaSuccess) 
				if (! (C.els = (float*) malloc(C.width*C.height*sizeof(C.els[0]))))
					RET_ERROR("couldn't allocate host mem for C matrix");

			printf("Using sequential implementation\n");
			if (MatMultSeq(&A, &B, &C, alpha) != SUCCESS)
				RET_ERROR("MatMultSeq failed");
		}
	}

	if (useOldFormat) {
		if (MatrixToFile(argv[5], out, trans) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	} else {
		if (MatrixToCOOFile(argv[5], out, trans) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	}

	return 0;
}
