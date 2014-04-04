#include "linear_decoder.h"

int main()
{
	int visibleSize = 8*8*3; // Size of the input vectors (8x8 RGB images patches)
	int hiddenSize = 64;
	int trainSize = 100; // Size of the training set 

	float* d_data;
	float* h_data = new(float[visibleSize * trainSize]);

	// Load the training set in host memory
	load("data/patches.dat", h_data, trainSize*visibleSize);

	// Allocate memory in device for the training data and labels
	CUDA_SAFE_CALL(cudaMalloc(&d_data, visibleSize * trainSize * sizeof(float)));

	// Copy training set and label to device memory
	CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, visibleSize * trainSize * sizeof(float), cudaMemcpyHostToDevice));

	return 0;
}