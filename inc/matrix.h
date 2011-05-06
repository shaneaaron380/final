#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "common.h"

typedef struct {
	int width;
	int height;
	int stride;
	float *els;
} Matrix;

int MatrixFromFile(char const* const filename, Matrix *m);
int MatrixToFile(char const* const filename, Matrix const* const m);
int MatrixFromFile_T(char const* const filename, Matrix *m);
int MatrixToFile_T(char const* const filename, Matrix const* const m);

#endif
