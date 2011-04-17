NVCC = nvcc
FLAGS = -I/usr/local/cuda/inc -Iinc
LIBRARIES = -L/usr/local/cuda/lib -Llib
TARGET := bin/$(shell basename `pwd`)
SHELL := $(shell which bash)

ifeq ($(DEBUG), 1)
	override FLAGS += -O0 -g -D DEBUG
endif

OBJ_DIR := obj


################################################################################
# main application
################################################################################

all: $(TARGET)

$(TARGET): obj/main.o obj/matrix.o obj/mat_mult_from_doc.o
	$(NVCC) $(FLAGS) $(LIBRARIES) -o $@ obj/main.o obj/matrix.o obj/mat_mult_from_doc.o

obj/main.o: src/main.cu inc/matrix.h inc/mat_mult_from_doc.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

run: $(TARGET)
	$(SHELL) -c "DYLD_LIBRARY_PATH=/usr/local/cuda/lib ./$(TARGET) \
		obj/test_input_incremental.mat obj/test_input_incremental.mat 1.0 C \
		obj/test_output_incremental.mat"


################################################################################
# libraries
################################################################################

obj/matrix.o: lib/matrix.cu inc/matrix.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBRARIES) -c -o $@ $<

obj/mat_mult_from_doc.o: lib/mat_mult_from_doc.cu inc/mat_mult_from_doc.h | $(OBJ_DIR)
	$(NVCC) $(FLAGS) $(LIBRARIES) -c -o $@ $<

################################################################################
# housekeeping
################################################################################

.PHONY: clean inputs

clean:
	-rm -rf $(OBJ_DIR) $(mARGET) Session.vim

inputs:
	bin/make_test_inputs.py

tags: src/* inc/* lib/*
	[ -f tags ] && rm tags || true
	ctags src/* inc/* lib/*

