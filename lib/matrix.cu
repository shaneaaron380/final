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

/*int MatrixFromFile_T(char const* const filename, Matrix *m)*/
/*{*/
/*    FILE *f;*/

/*    if (! (f = fopen(filename, "r")))*/
/*        RET_ERROR("could not open %s", filename);*/

/*    if (fscanf(f, "%d %d\n", &m->width, &m->height) != 2)*/
/*        RET_ERROR("could not read width and height from %s", filename);*/

/*    if (! (m->els = (float*) malloc(m->width * m->height * sizeof(m->els[0]))))*/
/*        RET_ERROR("could not allocate memory for elements from %s", filename);*/


/*    fclose(f);*/

/*    return SUCCESS;*/
/*}*/

/*int MatrixToFile_T(char const* const filename, Matrix const* const m)*/
/*{*/
/*    FILE *f;*/

/*    if (! (f = fopen(filename, "w")))*/
/*        RET_ERROR("could not open %s for writing", filename);*/

/*    fprintf(f, "%d %d\n", m->width, m->height);*/

/*    for (int i = 0; i < m->height; ++i) {*/
/*        for (int j = 0; j < m->width; ++j)*/
/*            fprintf(f, "%f ", m->els[j * m->height + i]);*/
/*        fprintf(f, "\n");*/
/*    }*/

/*    fclose(f);*/

/*    return SUCCESS;*/
/*}*/

