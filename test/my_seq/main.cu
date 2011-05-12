
#include "mat_mult_seq.h"
#include "sys/time.h"

void Usage(int retVal, char *argv0)
{
	fprintf(retVal == 0? stdout : stderr,
			"USAGE: %s <matrix A input file> <matrix B input file> "
			"<output matrix file> <alpha> <S,G,C> [o]\n", argv0);

	exit(retVal);
}

int GetInputs(int argc, char *argv[], Matrix *a, Matrix *b, float *alpha, char
		*which, int *useOldFormat)
{
	*useOldFormat = 0;

	if (argc < 6)
		RET_ERROR("must have at least 5 cmd line args");

	if (argc > 6 && strncmp(argv[6], "o", 2) == 0)
		*useOldFormat = 1;

	if (*useOldFormat) {
		if (MatrixFromFile(argv[1], a, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromFile(argv[2], b, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix B");
	} else {
		if (MatrixFromCOOFile(argv[1], a, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix A");

		if (MatrixFromCOOFile(argv[2], b, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not read matrix B");
	}

	*alpha = strtof(argv[3], (char**)NULL);

	*which = argv[4][0];

	return SUCCESS;
}

int main(int argc, char *argv[])
{
	Matrix A, B, X;
	float alpha;
	char which;
	int useOldFormat;
	struct timeval timerValues;
	double start_time, end_time;
	timerclear(&timerValues);	

	if (GetInputs(argc, argv, &A, &B, &alpha, &which, &useOldFormat) != SUCCESS)
		Usage(1, argv[0]);

	X.width = A.width;
	X.height = A.height;
	if (! (X.els = (float*) malloc(X.width * X.height * sizeof(X.els[0]))))
		RET_ERROR("could not allocate memory for elements from test seq main");
	
  //Get start time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get start time of day\n");
	
	//if (timerisset(&timerValues)) 
	start_time = (double) timerValues.tv_sec	+ (double) (timerValues.tv_usec)/1000000;
	//printf("Start secs: %ld, Start usecs: %ld, Time: %f\n", timerValues.tv_sec, timerValues.tv_usec, start_time);

	MatMultSeq(&A, &B, &X, alpha); 

	//Get end time
	if (gettimeofday(&timerValues, NULL))
		printf("WARNING: Counldn't get end time of day\n");
	
	//if (timerisset(&timerValues)) 
	end_time = (double) timerValues.tv_sec + (double) (timerValues.tv_usec)/1000000;
	//printf("End secs: %ld, End usecs: %d, Total Time: %f\n", \
			timerValues.tv_sec, timerValues.tv_usec, end_time-start_time);
	printf("Total Time: %f\n", end_time-start_time);
	
  if (useOldFormat) {
		if (MatrixToFile(argv[5], &X, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	} else {
		if (MatrixToCOOFile(argv[5], &X, MATRIX_FILE_NO_TRANSPOSE) != SUCCESS)
			RET_ERROR("could not write result matrix to %s", argv[5]);
	}

	/*free(A.els);*/
	/*free(B.els);*/
	/*free(X.els);*/
	
  return 0;
}
