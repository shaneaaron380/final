NVCC = nvcc
FLAGS = -Iinc
LIBS = -Llib
TARGET := bin/$(shell basename `pwd`)
SHELL := $(shell which bash)

ifneq ($(TACC_CUDA_INC),)
	override FLAGS += -I$(TACC_CUDA_INC)
else
	override FLAGS += -I/usr/local/cuda/inc
endif

ifneq ($(TACC_CUDA_LIB),)
	override FLAGS += -L$(TACC_CUDA_LIB)
else
	override FLAGS += -L/usr/local/cuda/lib -lcublas
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

$(TARGET): obj/main.o obj/matrix.o obj/mat_mult_from_doc.o obj/mat_mult_cublas.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/main.o obj/matrix.o obj/mat_mult_from_doc.o obj/mat_mult_cublas.o

obj/main.o: src/main.cu inc/matrix.h inc/mat_mult_from_doc.h inc/mat_mult_cublas.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

run: $(TARGET) $(INPUTS)
	$(SHELL) -c "DYLD_LIBRARY_PATH=/usr/local/cuda/lib ./$(TARGET) \
		inputs/test_input_64_inc.txt \
		inputs/test_input_64_inc.txt 1.0 C \
		obj/test_input_64_inc.txt.out"

run2: $(TARGET) $(INPUTS)
	$(SHELL) -c "DYLD_LIBRARY_PATH=/usr/local/cuda/lib ./$(TARGET) \
		inputs/test_input_100000000_tri.txt 1.0 C \
		inputs/test_input_100000000_ones.txt  \
		obj/test_input_100000000_tri.txt.out"

################################################################################
# libraries
################################################################################

obj/matrix.o: lib/matrix.cu inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/mat_mult_from_doc.o: lib/mat_mult_from_doc.cu inc/mat_mult_from_doc.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/mat_mult_cublas.o: lib/mat_mult_cublas.cu inc/mat_mult_cublas.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<


################################################################################
# testing cublas
################################################################################

cublas: bin/my_cublas
	$(SHELL) -c "DYLD_LIBRARY_PATH=/usr/local/cuda/lib bin/my_cublas \
		inputs/test_input_100000000_tri.txt \
		inputs/test_input_100000000_ones.txt \
		1.0 C \
		obj/test_input_100000000_tri.txt.cuda.out"

bin/my_cublas: obj/my_cublas.o obj/mat_mult_cublas.o obj/matrix.o | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/my_cublas.o obj/mat_mult_cublas.o obj/matrix.o

obj/my_cublas.o: test/my_cublas/main.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<


################################################################################
# housekeeping
################################################################################

.PHONY: clean inputs

clean:
	-rm -rf $(OBJ_DIR) $(TARGET) Session.vim bin/my_cublas

tags: src/* inc/* lib/*
	[ -f tags ] && rm tags || true
	ctags src/* inc/* lib/*

