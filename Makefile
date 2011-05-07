NVCC = nvcc
FLAGS = -Iinc
LIBS = -Llib
TARGET := bin/$(shell basename `pwd`)
SHELL := $(shell which bash)

ifneq ($(TACC_CUDA_INC),)
	override FLAGS += -I$(TACC_CUDA_INC)
else
	override FLAGS += -I/usr/local/cuda/include
endif

ifneq ($(TACC_CUDA_LIB),)
	override LIBS += -L$(TACC_CUDA_LIB) -lcublas
else
	override LIBS += -L/usr/local/cuda/lib -lcublas
	DYLD_LIBRARY_PATH := /usr/local/cuda/lib
	export DYLD_LIBRARY_PATH
endif

ifeq ($(DEBUG), 1)
	override FLAGS += -O0 -g -D DEBUG
endif

OBJ_DIR := obj

all: $(TARGET)

INPUTS := $(shell bin/inputs.py -d)
# re-make inputs dependencies every time - i need a better way to do this...
INPUTS_DUMMY := $(shell bin/inputs.py -D > test_inputs.D)
-include test_inputs.D
inputs: $(INPUTS)
	@echo -n "" # dummy command just so make doesn't whine

################################################################################
# main application
################################################################################

$(TARGET): obj/main.o obj/matrix.o obj/mat_mult_gpu.o obj/mat_mult_cublas.o obj/mat_mult_seq.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/main.o obj/matrix.o obj/mat_mult_gpu.o obj/mat_mult_cublas.o obj/mat_mult_seq.o

obj/main.o: src/main.cu inc/matrix.h inc/mat_mult_gpu.h inc/mat_mult_cublas.h inc/mat_mult_seq.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

run: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_3_tri.txt \
		inputs/test_input_3_threes.txt 1.0 G \
		obj/test_input_3_tri.txt.gpu.out

run1: $(TARGET) $(INPUTS)
	./$(TARGET) inputs/test_input_64_inc.txt inputs/test_input_64_inc.txt 1.0 G obj/test_input_64_inc.txt.gpu.out

run2: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_100000000_tri.txt 1.0 C \
		inputs/test_input_100000000_ones.txt  \
		obj/test_input_100000000_tri.txt.out

################################################################################
# libraries
################################################################################

obj/matrix.o: lib/matrix.cu inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/mat_mult_from_doc.o: lib/mat_mult_from_doc.cu inc/mat_mult_from_doc.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/mat_mult_cublas.o: lib/mat_mult_cublas.cu inc/mat_mult_cublas.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/mat_mult_seq.o: lib/mat_mult_seq.cu inc/mat_mult_seq.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/mat_mult_gpu.o: lib/mat_mult_gpu.cu inc/mat_mult_gpu.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

################################################################################
# testing cublas
################################################################################

cublas: bin/my_cublas $(INPUTS)
	bin/my_cublas \
		inputs/test_cublas_A_unit_low.txt \
		inputs/test_cublas_B.txt \
		1.0 C \
		obj/test_cublas_A_unit_low.txt.out

bin/my_cublas: obj/my_cublas.o obj/mat_mult_cublas.o obj/matrix.o | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/my_cublas.o obj/mat_mult_cublas.o obj/matrix.o

obj/my_cublas.o: test/my_cublas/main.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<


################################################################################
# testing sequential 
################################################################################

sequential: bin/my_seq $(INPUTS)
#	$(SHELL) -c "DYLD_LIBRARY_PATH=/usr/local/cuda/lib bin/my_seq \
		inputs/test_input_1024_tri.txt \
		inputs/test_input_1024_ones.txt \
		1.0 S \
		obj/test_input_1024_tri.txt.seq.out"
	bin/my_seq \
		inputs/test_input_100000000_tri.txt \
		inputs/test_input_100000000_ones.txt \
		1.0 S \
		obj/test_input_100000000_tri.txt.seq.out

sequential2: bin/my_seq $(INPUTS)
	./bin/my_seq inputs/test_cublas_A_unit_low.txt inputs/test_cublas_B.txt 1.0 S obj/test_cublas_A_unit_low.txt.out

bin/my_seq: obj/my_seq.o obj/mat_mult_seq.o obj/matrix.o | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/my_seq.o obj/mat_mult_seq.o obj/matrix.o

obj/my_seq.o: test/my_seq/main.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<


################################################################################
# housekeeping
################################################################################

.PHONY: clean inputs

clean:
	-rm -rf $(OBJ_DIR) $(TARGET) Session.vim bin/my_cublas bin/my_seq

CTAGS_DIRS = $(shell \
			 for d in `echo $(FLAGS) | sed 's/-I//g'`; do \
				 [ -d $$d ] && echo "$$d/*"; \
			 done)
tags: src/* lib/* $(CTAGS_DIRS)
	[ -f tags ] && rm tags || true
	ctags --langmap=C:.c.cu src/* lib/* $(CTAGS_DIRS)

