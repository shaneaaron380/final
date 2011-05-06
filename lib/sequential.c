
#include<stdio.h>
#include<stdlib.h>

void tri_solve_seq(double** A, double** B, double** X, double alpha, unsigned int n) {

  int i, j, k;
  double S;

  //apply alpha
  //for (i = 0; i < n; i++)
  //  for (j = 0; j < n; j++)
  //    B[i][j] = alpha*B[i][j];

  //initiallize known values of X[O]s based on B[0]s
  //we can do this because we know the diagonal (i==j) are all 1s in A
  for (i = 0; i < n; i++)
    X[0][i] = alpha*B[0][i];

  //calculate rest of X[1..n] values
  for (i = 1; i < n; i++) {
    for (j = 0; j < n; j++) {
      S = alpha*B[i][j];
      //printf("i=%d,j=%d, S=%f\n", i, j, S);
      for (k = 0; k < i; k++) {
        S -= A[i][k] * X[k][j];
        //printf("k=%d, S=%f\n", k, S);
      }
      X[i][j] = S; 
    }
  }
  
}

//int main ( ) {
//
//  int i, j;
//  int n = 3;
//  double alpha = 1;
//  
//  double ** A = (double **) malloc(n*sizeof (double *) );
//  double ** B = (double **) malloc(n*sizeof (double *) );
//  double ** X = (double **) malloc(n*sizeof (double *) );
//  
//  for (i = 0; i < n; i++ )
//    A[i] = (double *) malloc(n*sizeof(double));
//
//  for (i = 0; i < n; i++ )
//    B[i] = (double *) malloc(n*sizeof(double));
//  
//  for (i = 0; i < n; i++ )
//    X[i] = (double *) malloc(n*sizeof(double));
// 
//  printf("A=\n");
//  for (i = 0; i < n; i++) {
//    for (j = 0; j < n; j++) {
//      if (i == j)
//        A[i][j] = 1;
//      if (i > j)
//        A[i][j] = 0;
//      if (j < i)
//        A[i][j] = 5;
//      printf("%f ", A[i][j]);
//    }
//    printf("\n");
//  }
//  printf("\nB=\n");
//  for (i = 0; i < n; i++) {
//    for (j = 0; j < n; j++) {
//      B[i][j] = 3;
//      printf("%f ", B[i][j]);
//    }
//    printf("\n");
//  }
//  printf("\n");
//
//  tri_solve_seq(A, B, X, alpha, n); 
//
//  printf("\nX=\n");
//  for (i = 0; i < n; i++) {
//    for (j = 0; j < n; j++) {
//      printf("%f ", X[i][j]);
//    }
//    printf("\n");
//  }
//
//  for (i = 0; i < n; i++) {
//    free(A[i]);
//    free(B[i]);
//    free(X[i]);
//  }
//  free(A);
//  free(B);
//  free(X);
//
//  return 0;
//}
