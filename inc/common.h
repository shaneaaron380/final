#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>

#define SUCCESS 0
#define ERROR -1

#define RET_ERROR(...) \
{ \
	fprintf(stderr, "ERROR: "); \
	fprintf(stderr, __VA_ARGS__); \
	fprintf(stderr, "\n"); \
	return ERROR; \
}

#endif
