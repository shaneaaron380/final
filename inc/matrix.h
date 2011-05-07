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
#define MATRIX_FILE_NO_TRANSPOSE 1

int MatrixFromFile(char const* const filename, Matrix *m, int trans);
int MatrixToFile(char const* const filename, Matrix const* const m, int trans);

#endif
