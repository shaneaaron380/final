#ifndef __MAT_MULT_CUBLAS_H__
#define __MAT_MULT_CUBLAS_H__

#include <stdio.h>
#include <cublas.h>

#include "matrix.h"

int MatMultCublas(const Matrix A, Matrix B, const float alpha);

#endif
