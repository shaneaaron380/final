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
	override LIBS += -L$(TACC_CUDA_LIB) -lcublas -lcudart
else
	override LIBS += -L/usr/local/cuda/lib -lcublas -lcuda
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

$(TARGET): obj/main.o obj/matrix.o obj/mat_mult_gpu.o obj/mat_mult_cublas.o obj/mat_mult_seq.o obj/mat_mult_shared.o obj/cuPrintf.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/main.o obj/matrix.o obj/mat_mult_gpu.o obj/mat_mult_cublas.o obj/mat_mult_seq.o obj/mat_mult_shared.o obj/cuPrintf.o

obj/main.o: src/main.cu inc/matrix.h inc/mat_mult_gpu.h inc/mat_mult_cublas.h inc/mat_mult_seq.h inc/mat_mult_shared.h inc/cuPrintf.cuh | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

run: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_1024_tri.txt \
		inputs/test_input_1024_ones.txt 1.0 G \
		obj/test_input_1024_tri.txt.gpu.out o

run_all:
	make gpu
	make cublas
	make seq

run1: $(TARGET) $(INPUTS)
	./$(TARGET) inputs/test_input_64_inc.txt inputs/test_input_64_inc.txt 1.0 G obj/test_input_64_inc.txt.gpu.out o

run2: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_100000000_tri.txt \
		inputs/test_input_100000000_ones.txt 1.0 G \
		obj/test_input_100000000_tri.txt.out o

run3: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_90000_tri.txt \
		inputs/test_input_90000_ones.txt 1.0 G \
		obj/test_input_90000_tri.txt.out o

run4: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_25000000_tri.txt \
		inputs/test_input_25000000_ones.txt 1.0 G \
		obj/test_input_25000000_tri.txt.out o

run5: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_4000000_tri.txt \
		inputs/test_input_4000000_ones.txt 1.0 G \
		obj/test_input_4000000_tri.txt.out o

run6: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_1000000_tri.txt \
		inputs/test_input_1000000_ones.txt 1.0 G \
		obj/test_input_1000000_tri.txt.out o

gpu:
	make gpu1
	make gpu2
	make gpu3

gpu1: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_1a.txt \
		inputs/test_input_1b.txt 1.0 G \
		obj/test_input_1.txt.gpu.out

gpu2: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_2a.txt \
		inputs/test_input_2b.txt 1.0 G \
		obj/test_input_2.txt.gpu.out

gpu3: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/test_input_3a.txt \
		inputs/test_input_3b.txt 1.0 G \
		obj/test_input_3.txt.gpu.out

simple: $(TARGET) $(INPUTS)
	./$(TARGET) \
		inputs/simple_A.txt \
		inputs/simple_B.txt 1.0 G \
		obj/simple_A.txt.gpu.out
	bin/diff_coo_matrices.py inputs/simple_A_golden.txt obj/simple_A.txt.gpu.out

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

obj/mat_mult_shared.o: lib/mat_mult_shared.cu inc/mat_mult_shared.h inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

obj/cuPrintf.o: lib/cuPrintf.cu inc/cuPrintf.cuh | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

################################################################################
# testing cublas
################################################################################

#cublas: bin/my_cublas $(INPUTS)
#    bin/my_cublas \
#        inputs/test_input_1024_tri.txt \
#        inputs/test_input_1024_ones.txt \
#        1.0 C \
#        obj/test_input_1024_tri.txt.cublas.out o
cublas:
	make cublas1
	make cublas2
	make cublas3

cublassmall: bin/my_cublas $(INPUTS)
	bin/my_cublas \
		inputs/test_cublas_A_unit_low.txt \
		inputs/test_cublas_B.txt \
		1.0 C \
		obj/test_cublas_A_unit_low.txt.out o

cublastest: cublassmall cublas
	bin/golden_mat_mult.py \
		inputs/test_cublas_A_unit_low.txt \
		inputs/test_cublas_B.txt > \
		obj/test_cublas_A_unit_low.txt.golden
	bin/diff_matrices.py \
		obj/test_cublas_A_unit_low.txt.out \
		obj/test_cublas_A_unit_low.txt.golden
	bin/golden_mat_mult.py \
		inputs/test_input_1024_tri.txt \
		inputs/test_input_1024_ones.txt > \
		obj/test_input_1024_tri.txt.golden
	bin/diff_matrices.py \
		obj/test_input_1024_tri.txt.cublas.out \
		obj/test_input_1024_tri.txt.golden


cublaslarge: bin/my_cublas $(INPUTS)
	bin/my_cublas \
		inputs/test_input_100000000_tri.txt \
		inputs/test_input_100000000_ones.txt \
		1.0 C \
		obj/test_input_100000000_tri.txt.cublas.out o

cublas1: bin/my_cublas $(INPUTS)
	bin/my_cublas \
		inputs/test_input_1a.txt \
		inputs/test_input_1b.txt \
		1.0 C \
		obj/test_input_1.txt.cublas.out

cublas2: bin/my_cublas $(INPUTS)
	bin/my_cublas \
		inputs/test_input_2a.txt \
		inputs/test_input_2b.txt \
		1.0 C \
		obj/test_input_2.txt.cublas.out

cublas3: bin/my_cublas $(INPUTS)
	bin/my_cublas \
		inputs/test_input_3a.txt \
		inputs/test_input_3b.txt \
		1.0 C \
		obj/test_input_3.txt.cublas.out

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
		obj/test_input_100000000_tri.txt.seq.out o

sequentialsmall: bin/my_seq $(INPUTS)
	./bin/my_seq inputs/test_cublas_A_unit_low.txt inputs/test_cublas_B.txt 1.0 S obj/test_cublas_A_unit_low.txt.out o

seq:
	make seq1
	make seq2
	make seq3

seq1: bin/my_seq $(INPUTS)
	bin/my_seq \
		inputs/test_input_1a.txt \
		inputs/test_input_1b.txt \
		1.0 C \
		obj/test_input_1.txt.seq.out

seq2: bin/my_seq $(INPUTS)
	bin/my_seq \
		inputs/test_input_2a.txt \
		inputs/test_input_2b.txt \
		1.0 C \
		obj/test_input_2.txt.seq.out

seq3: bin/my_seq $(INPUTS)
	bin/my_seq \
		inputs/test_input_3a.txt \
		inputs/test_input_3b.txt \
		1.0 C \
		obj/test_input_3.txt.seq.out

bin/my_seq: obj/my_seq.o obj/mat_mult_seq.o obj/matrix.o | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/my_seq.o obj/mat_mult_seq.o obj/matrix.o

obj/my_seq.o: test/my_seq/main.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<


################################################################################
# testing matrix i/o
################################################################################

conversions: bin/old_to_coo bin/coo_to_old $(INPUTS)
	bin/old_to_coo inputs/test_cublas_A.txt > obj/test_cublas_A.txt.coo
	bin/coo_to_old obj/test_cublas_A.txt.coo > obj/test_cublas_A.txt.coo.txt
	bin/diff_matrices.py inputs/test_cublas_A.txt obj/test_cublas_A.txt.coo.txt
	@
	bin/old_to_coo inputs/test_cublas_A.txt t > obj/test_cublas_A.txt.coo
	bin/coo_to_old obj/test_cublas_A.txt.coo t > obj/test_cublas_A.txt.coo.txt
	bin/diff_matrices.py inputs/test_cublas_A.txt obj/test_cublas_A.txt.coo.txt
	@
	bin/coo_to_old inputs/test_input_1a.txt > obj/test_input_1a.txt.old
	bin/old_to_coo obj/test_input_1a.txt.old > obj/test_input_1a.txt.old.txt
	bin/diff_coo_matrices.py inputs/test_input_1a.txt obj/test_input_1a.txt.old.txt
	@
	bin/coo_to_old inputs/test_input_1a.txt t > obj/test_input_1a.txt.old
	bin/old_to_coo obj/test_input_1a.txt.old > obj/test_input_1a.txt.old.txt
	bin/diff_coo_matrices.py inputs/test_input_1a.txt obj/test_input_1a.txt.old.txt

bin/old_to_coo: obj/old_to_coo.o obj/matrix.o | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/old_to_coo.o obj/matrix.o

obj/old_to_coo.o: test/old_to_coo/main.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

bin/coo_to_old: obj/coo_to_old.o obj/matrix.o | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -o $@ obj/coo_to_old.o obj/matrix.o

obj/coo_to_old.o: test/coo_to_old/main.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBS) -c -o $@ $<

################################################################################
# housekeeping
################################################################################

.PHONY: clean inputs cublas cublassmall

clean:
	-rm -rf $(OBJ_DIR) $(TARGET) Session.vim bin/my_cublas bin/my_seq

CTAGS_DIRS = $(shell \
			 for d in `echo $(FLAGS) | sed 's/-I//g'`; do \
				 [ -d $$d ] && echo "$$d/*"; \
			 done)
tags: src/* lib/* $(CTAGS_DIRS)
	[ -f tags ] && rm tags || true
	ctags --langmap=C:.c.cu src/* lib/* $(CTAGS_DIRS)

