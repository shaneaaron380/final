#include <sys/time.h>
#include "mat_mult_seq.h"

int MatMultSeq(Matrix *A, Matrix *B, Matrix *X, float alpha)
{
	int i, j, k;
	float S;
	int n = A->width;
	struct timeval timerValues;
	double start_time, end_time;
	timerclear(&timerValues);	

	if (gettimeofday(&timerValues, NULL))
		RET_ERROR("could not gettimeofday for start_time");

	start_time = (double) timerValues.tv_sec +
				 (double) (timerValues.tv_usec) / 1000000.0;

	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {
			S = alpha*B->els[i*n+j];
			for (k = 0; k < i; k++) {
				S -= A->els[i*n+k] * X->els[k*n+j];
			}
			X->els[i*n+j] = S;
		}
	}

	if (gettimeofday(&timerValues, NULL))
		RET_ERROR("could not gettimeofday for end_time");

	end_time =  (double) timerValues.tv_sec	+
				(double) (timerValues.tv_usec) / 1000000.0;

	printf("Total Time: %f\n", end_time - start_time);

	return SUCCESS;
}

