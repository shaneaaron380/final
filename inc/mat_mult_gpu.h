#ifndef __MAT_MULT_GPU_H__
#define __MAT_MULT_GPU_H__

#include "matrix.h"

#define BLOCK_SIZE 16

void MatMultGPU(const Matrix A, const Matrix B, Matrix C);

#endif

