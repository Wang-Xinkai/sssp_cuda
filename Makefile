CC=nvcc
# HIP_PLATFORM=$(shell hipconfig --compiler)

# ifeq (${HIP_PLATFORM}, nvcc)
# 	HIPCC_FLAGS = -gencode=arch=compute_20,code=sm_20 
# endif
# ifeq (${HIP_PLATFORM}, hcc)
# 	HIPCC_FLAGS = -Wno-deprecated-register
# endif

FLAG=-arch=sm_75 -G -g  -std=c++11  -I /home/pywang/env/cub

# SOURCE := utils.cu kernels.cu shortest_path.cu
# OBJS   := utils.o kernels.o shortest_path.o

all: shortest_path.o
	$(CC)  shortest_path.o  test.cpp  -o test $(FLAG) 
	
shortest_path.o: shortest_path.cu 
	$(CC) shortest_path.cu -c -o shortest_path.o $(FLAG) 
# scan.o: scan.cu
# 	$(CC) scan.cu -c -o scan.o $(FLAG)
# utils.o: utils.cu
# 	$(CC) utils.cu -o utils.o -c $(FLAG) 
# kernels.o: kernels.cu
# 	$(CC) kernels.cu -o kernels.o -c $(FLAG) 


	# $(CC) shortest_path.o test.cpp -o test.o $(FLAG) 

clean:
	rm -f *.o $(EXE)