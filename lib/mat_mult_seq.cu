
#include "mat_mult_seq.h"

void MatMultSeq(Matrix *A, Matrix *B, Matrix *X, float alpha) {

  int i, j, k;
  float S;
  int n = A->width;

  //initiallize known values of X[O]s based on B[0]s
  //we can do this because we know the diagonal (i==j) are all 1s in A
  for (i = 0; i < n; i++)
    X->els[i] = alpha*B->els[i];

  //calculate rest of X[1..n] values
  for (i = 1; i < n; i++) {
    for (j = 0; j < n; j++) {
      S = alpha*B->els[i*B->width+j];
      printf("i=%d,j=%d, S=%f\n", i, j, S);
      for (k = 0; k < i; k++) {
        S -= A->els[i*A->width+k] * X->els[k*X->width+j];
        printf("k=%d, S=%f\n", k, S);
      }
      X->els[i*X->width+j] = S; 
    }
  }
  //apply alpha
  //for (i = 0; i < n; i++)
  //  for (j = 0; j < n; j++)
  //    B[i][j] = alpha*B[i][j];

  //initiallize known values of X[O]s based on B[0]s
  //we can do this because we know the diagonal (i==j) are all 1s in A
  //for (i = 0; i < n; i++)
  //  X[0][i] = alpha*B[0][i];

  ////calculate rest of X[1..n] values
  //for (i = 1; i < n; i++) {
  //  for (j = 0; j < n; j++) {
  //    S = alpha*B[i][j];
  //    //printf("i=%d,j=%d, S=%f\n", i, j, S);
  //    for (k = 0; k < i; k++) {
  //      S -= A[i][k] * X[k][j];
  //      //printf("k=%d, S=%f\n", k, S);
  //    }
  //    X[i][j] = S; 
  //  }
  //}
  
}

