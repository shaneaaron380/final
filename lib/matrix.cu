#include "matrix.h"

int MatrixFromFile(char const* const filename, Matrix *m, int trans)
{
	FILE *f;

	if (! (f = fopen(filename, "r")))
		RET_ERROR("could not open %s", filename);

	if (fscanf(f, "%d %d\n", &m->width, &m->height) != 2)
		RET_ERROR("could not read width and height from %s", filename);

	if (! (m->els = (float*) malloc(m->width * m->height * sizeof(m->els[0]))))
		RET_ERROR("could not allocate memory for elements from %s", filename);

	if (trans == MATRIX_FILE_TRANSPOSE) {
		for (int i = 0; i < m->height; ++i)
			for (int j = 0; j < m->width; ++j)
				if (fscanf(f, "%f", &m->els[i + j * m->height]) != 1)
					RET_ERROR("could not read element (%d, %d)", i, j);
	} else {
		for (int i = 0; i < m->height; ++i)
			for (int j = 0; j < m->width; ++j)
				if (fscanf(f, "%f", &m->els[i * m->width + j]) != 1)
					RET_ERROR("could not read element (%d, %d)", i, j);
	}

	fclose(f);

	return SUCCESS;
}

int MatrixToFile(char const* const filename, Matrix const* const m, int trans)
{
	FILE *f;

	if (strncmp(filename, "-", 2) == 0)
		f = stdout;
	else
		if (! (f = fopen(filename, "w")))
			RET_ERROR("could not open %s for writing", filename);

	fprintf(f, "%d %d\n", m->width, m->height);

	if (trans == MATRIX_FILE_TRANSPOSE) {
		for (int i = 0; i < m->height; ++i) {
			for (int j = 0; j < m->width; ++j)
				fprintf(f, "%f ", m->els[j * m->height + i]);
			fprintf(f, "\n");
		}

	} else {
		for (int i = 0; i < m->width; ++i) {
			for (int j = 0; j < m->height; ++j)
				fprintf(f, "%f ", m->els[i * m->width + j]);
			fprintf(f, "\n");
		}
	}

	fclose(f);

	return SUCCESS;
}

int MatrixFromCOOFile(char const* const filename, Matrix *m, int trans)
{
	FILE *f;
	float temp;
	int i,j;

	if (! (f = fopen(filename, "r")))
		RET_ERROR("could not open %s", filename);

	if (fscanf(f, "%d %d\n", &m->height, &m->width) != 2)
		RET_ERROR("could not read width and height from %s", filename);

	if (! (m->els = (float*) malloc(m->height * m->width * sizeof(m->els[0]))))
		RET_ERROR("could not allocate memory for elements from %s", filename);

	// zero out everything since we don't have any guarantees about the matrix
	bzero(m->els, m->height * m->width * sizeof(m->els[0]));

	if (trans == MATRIX_FILE_TRANSPOSE)
		while (fscanf(f, "%d %d %f", &i, &j, &temp) == 3)
			m->els[i + j * m->height] = temp;
	else
		while (fscanf(f, "%d %d %f", &i, &j, &temp) == 3)
			m->els[i * m->width + j] = temp;

	fclose(f);

	return SUCCESS;
}

int MatrixToCOOFile(char const* const filename, Matrix const* const m, int trans)
{
	FILE *f;

	if (strncmp(filename, "-", 2) == 0)
		f = stdout;
	else
		if (! (f = fopen(filename, "w")))
			RET_ERROR("could not open %s for writing", filename);

	fprintf(f, "%d %d\n", m->height, m->width);

	if (trans == MATRIX_FILE_TRANSPOSE) {
		for (int i = 0; i < m->height; ++i)
			for (int j = 0; j < m->width; ++j)
				if (m->els[j * m->height + i] != 0)
					fprintf(f, "%d %d %f\n", i, j, m->els[j * m->height + i]);

	} else {
		for (int i = 0; i < m->width; ++i)
			for (int j = 0; j < m->height; ++j)
				if (m->els[i * m->width + j] != 0)
					fprintf(f, "%d %d %f\n", i, j, m->els[i * m->width + j]);
	}

	fclose(f);

	return SUCCESS;
}
