#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "common.h"

typedef struct {
	int width;
	int height;
	int stride;
	float *els;
} Matrix;

#define MATRIX_FILE_TRANSPOSE 1
#define MATRIX_FILE_NO_TRANSPOSE 0

int MatrixFromFile(char const* const filename, Matrix *m, int trans);
int MatrixToFile(char const* const filename, Matrix const* const m, int trans);

int MatrixFromCOOFile(char const* const filename, Matrix *m, int trans);
int MatrixToCOOFile(char const* const filename, Matrix const* const m, int trans);

void TruncateMatrix(Matrix A);
void TruncAndPadMatrix(Matrix A, int alignment);
int GetPadMatrixSize(int N, int alignment);

#endif
