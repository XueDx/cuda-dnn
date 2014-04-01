#include "stacked_autoencoder.h"

int main()
{
	int visibleSize = 28*28; // Size of the input vectors
	int hiddenSizeL1 = 196; // Hidden size of the first autoencoder layer
	int hiddenSizeL2 = 196; // Hidden size of the second autoencoder layer
	int numClasses = 10; // Number of different class labels

	int trainSize = 5000; // Size of the training set 
	int testSize = 10000; // Size of the test set

	float* d_data;
	float* d_label;
	float* d_testData;
	float* d_testLabel;
	float* h_data = new(float[visibleSize * trainSize]);
	float* h_label = new(float[trainSize]);
	float* h_testData = new(float[visibleSize * testSize]);
	float* h_testLabel = new(float[testSize]);

	// Load the training set in host memory
	load("data/trainData.dat", h_data, trainSize*visibleSize);
	load("data/trainLabel.dat", h_label, trainSize);

	// Load the training labels in host memory
	load("data/testData.dat", h_testData, testSize*visibleSize);
	load("data/testLabel.dat", h_testLabel, testSize);

	// Allocate memory in device for the training data and labels
	CUDA_SAFE_CALL(cudaMalloc(&d_data, visibleSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_label, trainSize * sizeof(float)));
	
	// Copy training set and label to device memory
	CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, visibleSize * trainSize * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_label, h_label, trainSize * sizeof(float), cudaMemcpyHostToDevice));

	// Create two sparse autoencoders and one softmax layer
	SparseAutoencoder* sa1 = new SparseAutoencoder(visibleSize, hiddenSizeL1);
	SparseAutoencoder* sa2 = new SparseAutoencoder(hiddenSizeL1, hiddenSizeL2);
	Softmax* sm = new Softmax(hiddenSizeL2, numClasses);

	// Create a stacked autoencoder object
	StackAutoencoder* sa = new StackAutoencoder();
	
	// Add the autoencoders to the stack (the order is important)
	sa->addAutoencoder(sa1);
	sa->addAutoencoder(sa2);
	// Add the softmax layer to the stack (the order is important)
	sa->addSoftmax(sm);

	// Pre-train the network (layer-wise)
	PROFILE("pre-train",
	sa->train(d_data, d_label, trainSize, 2000);
	);

	// Allocate memory in device for the test data and labels
	CUDA_SAFE_CALL(cudaMalloc(&d_testData, visibleSize * testSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_testLabel, testSize * sizeof(float)));
	
	// Copy test set and label to device memory
	CUDA_SAFE_CALL(cudaMemcpy(d_testData, h_testData, visibleSize * testSize * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_testLabel, h_testLabel, testSize * sizeof(float), cudaMemcpyHostToDevice));
	
	// Calculate the accuracy of the sparse autoencoder before fine tunning
	sa->test(d_testData, d_testLabel, testSize);
	
	// Free the memory allocated for the test data
	CUDA_SAFE_CALL(cudaFree(d_testData));
	CUDA_SAFE_CALL(cudaFree(d_testLabel));

	// Train the network for fine tunning
	PROFILE("finetune",
	sa->fineTune(d_data, d_label, trainSize, 2000);
	);

	// Allocate memory in device for the test data and labels
	CUDA_SAFE_CALL(cudaMalloc(&d_testData, visibleSize * testSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_testLabel, testSize * sizeof(float)));
	
	// Copy test set and label to device memory
	CUDA_SAFE_CALL(cudaMemcpy(d_testData, h_testData, visibleSize * testSize * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_testLabel, h_testLabel, testSize * sizeof(float), cudaMemcpyHostToDevice));
	
	// Calculate the accuracy of the sparse autoencoder after fine tunning
	sa->test(d_testData, d_testLabel, testSize);
	
	// Clean the resources
	delete sa;
	CUDA_SAFE_CALL(cudaFree(d_testData));
	CUDA_SAFE_CALL(cudaFree(d_testLabel));
	CUDA_SAFE_CALL(cudaFree(d_data));
	CUDA_SAFE_CALL(cudaFree(d_label));
	
	CUDA_SAFE_CALL(cudaDeviceReset());
	return 0;
}