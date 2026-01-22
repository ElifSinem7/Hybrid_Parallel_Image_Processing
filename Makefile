# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -fopenmp -I./include
NVCCFLAGS = -O3 -arch=sm_75 -Xcompiler -fopenmp -I./include

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin

# Targets
TARGETS = $(BIN_DIR)/sequential $(BIN_DIR)/openmp $(BIN_DIR)/cuda $(BIN_DIR)/hybrid

# Object files
IMAGE_UTILS_OBJ = $(SRC_DIR)/image_utils.o

all: directories $(TARGETS)

directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p results

# Image utilities
$(IMAGE_UTILS_OBJ): $(SRC_DIR)/image_utils.cpp $(INC_DIR)/image_utils.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Sequential
$(BIN_DIR)/sequential: $(SRC_DIR)/sequential.cpp $(IMAGE_UTILS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

# OpenMP
$(BIN_DIR)/openmp: $(SRC_DIR)/openmp.cpp $(IMAGE_UTILS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

# CUDA
$(BIN_DIR)/cuda: $(SRC_DIR)/cuda.cu $(SRC_DIR)/kernels.cu $(IMAGE_UTILS_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Hybrid
$(BIN_DIR)/hybrid: $(SRC_DIR)/hybrid.cu $(SRC_DIR)/kernels.cu $(IMAGE_UTILS_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -rf $(BIN_DIR) $(SRC_DIR)/*.o results/*

.PHONY: all clean directories
