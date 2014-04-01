#ifndef SOFTMAX_H
#define SOFTMAX_H
 
#include "helper.cuh"

class Softmax
{
public:
    int numClasses;
    int inputSize;
    float* d_theta;
 
    Softmax(int input, int classes) : numClasses(classes), inputSize(input) {
		CUDA_SAFE_CALL(cudaMalloc(&d_theta, numClasses * inputSize * sizeof(float)));
	}

	~Softmax() {
		CUDA_SAFE_CALL(cudaFree(d_theta));
	}

	// Train using data and label as training set
	void train(float* data, float* label, int trainSize, int maxIter);

	// Compute the accuracy of the network on testData
	void test(float* data, float* label, int testSize);
};

void Softmax::train(float* d_data, float* d_label, int trainSize, int maxIter){
	curandState* devStates;
	cublasHandle_t handle;
	float* d_thetagrad;
	float* d_groundtruth;
	float* d_M;
	float* d_aux_tx1;
	float* d_aux_ix1;
	float* d_AUX_nxt;
	float* d_AUX_nxi;
	float* d_ones_tx1;
	float sum;
	float cost = 0.0f;
	float prev_cost = 0.0f;
	float alpha = 1.5f; //learning rate
	float one_over_trainSize = 1.0f/(float)trainSize; // needed in some methods

	CUDA_SAFE_CALL(cudaMalloc(&d_thetagrad, numClasses * inputSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_groundtruth, numClasses * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_M, numClasses * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_AUX_nxt, numClasses * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_AUX_nxi, numClasses * inputSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_tx1, trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_ix1, inputSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ones_tx1, trainSize * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&devStates, numClasses * inputSize * sizeof( curandState )));
	CUBLAS_SAFE_CALL(cublasCreate(&handle));

	// Random initialize parameters
	setup_kernel<<<ceilf(numClasses * inputSize/(float)NTHREADS), NTHREADS>>>( devStates, 0 );
	initialize_normal<<<ceilf(numClasses * inputSize/(float)NTHREADS), NTHREADS>>>(d_theta, numClasses, inputSize , devStates);
	initialize_groundtruth<<<ceilf(numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_groundtruth, d_label, numClasses, trainSize);
	initialize_float<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_ones_tx1, trainSize, 1.0f);

	//Gradient Descent
	for(int iteration = 1; iteration <= maxIter; iteration++){
		/******************************************************************************
		****** Compute P (prediction matrix)*******************************************
		*******************************************************************************/
		// d_M = d_theta * d_data (numClasses x trainSize)
		mMul(handle, numClasses, trainSize, inputSize, d_theta, d_data, d_M);

		// d_aux_tx1 = max(d_m) (trainSize)
		columnMax<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, numClasses, trainSize);

		// d_M = d_M - repmat(d_aux_tx1, trainSize, 1)
		mSubMax(handle, numClasses, trainSize, d_M, d_aux_tx1, d_ones_tx1);

		// d_M = exp(d_M)
		Exp<<<ceilf(numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_M,d_M,numClasses*trainSize);

		// d_aux_tx1 = sum(d_M) (trainSize: sum of columns)
		columnSum<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, numClasses, trainSize);

		// d_M = bsxfun(@rdivide, d_M, d_aux_tx1) (Divide column j of d_M by d_aux_tx1(j)) This is the prediction matrix P
		rdivide<<<ceilf(numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_M,d_aux_tx1,numClasses, trainSize);

		/******************************************************************************
		****** Cost *******************************************************************
		*******************************************************************************/
		prev_cost = cost;
	
		Log<<<ceilf(numClasses*trainSize/(float)NTHREADS), NTHREADS>>>(d_M,d_AUX_nxt, numClasses*trainSize);
		mHad<<<ceilf(numClasses*trainSize/(float)NTHREADS), NTHREADS>>>(d_AUX_nxt,d_groundtruth, d_AUX_nxt, numClasses, trainSize);
		columnSum<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_AUX_nxt, d_aux_tx1, numClasses, trainSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, trainSize, d_aux_tx1, 1, &sum));
		cost = sum/trainSize;

		Square<<<ceilf(numClasses*inputSize/(float)NTHREADS), NTHREADS>>>(d_theta,d_AUX_nxi, numClasses*inputSize);
		columnSum<<<ceilf(inputSize/(float)NTHREADS), NTHREADS>>>(d_AUX_nxi, d_aux_ix1, numClasses, inputSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, inputSize, d_aux_ix1, 1, &sum));
		cost += LAMBDA*sum/2.0f;
			
		std::cout <<  iteration << ": " << cost << std::endl;

		if(fabs(cost - prev_cost) < EPS_COST)
			break;
		// Dynamic learning rate
		if(cost > prev_cost)
			alpha = alpha*0.5f;
		else
			if(alpha < 0.5f/(0.1f + iteration/maxIter))
				alpha = alpha*1.01f;

		/******************************************************************************
		****** Gradient ***************************************************************
		*******************************************************************************/
		// d_M = d_groundtruth - d_M
		mSub<<<ceilf(numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_groundtruth, d_M, d_M,numClasses, trainSize);

		// d_thetagrad = d_M * d_data
		mMulTR(handle, numClasses, inputSize, trainSize,d_M,d_data,d_thetagrad);

		// d_thetagrad = -(1/trainSize)*d_thetagrad + LAMBDA * d_theta
		mAdd<<<ceilf(numClasses * inputSize/(float)NTHREADS), NTHREADS>>>(d_theta, d_thetagrad, numClasses, inputSize, LAMBDA, -one_over_trainSize);
		
		/******************************************************************************
		****** Update Theta************************************************************
		*******************************************************************************/
		// d_theta = d_theta - alpha*d_thetagrad
		mAdd<<<ceilf(numClasses * inputSize/(float)NTHREADS), NTHREADS>>>(d_thetagrad,d_theta, numClasses, inputSize, -alpha, 1.0f);
	}
	
	// Release GPU resources
	CUBLAS_SAFE_CALL(cublasDestroy(handle));
	CUDA_SAFE_CALL(cudaFree(devStates));
	CUDA_SAFE_CALL(cudaFree(d_thetagrad)); 
	CUDA_SAFE_CALL(cudaFree(d_groundtruth)); 
	CUDA_SAFE_CALL(cudaFree(d_M)); 
	CUDA_SAFE_CALL(cudaFree(d_AUX_nxt)); 
	CUDA_SAFE_CALL(cudaFree(d_AUX_nxi)); 
	CUDA_SAFE_CALL(cudaFree(d_aux_tx1)); 
	CUDA_SAFE_CALL(cudaFree(d_aux_ix1)); 
	CUDA_SAFE_CALL(cudaFree(d_ones_tx1)); 
}

void Softmax::test(float* d_data, float* d_label, int testSize){
	float* d_M;
	float* d_aux_tx1;
	float* d_ones_tx1;
	cublasHandle_t handle;

	CUDA_SAFE_CALL(cudaMalloc(&d_M, numClasses * testSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_tx1, testSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ones_tx1, testSize * sizeof(float)));
	CUBLAS_SAFE_CALL(cublasCreate(&handle));

	// Initialize the vector with ones used to copy vectors to matrices
	initialize_float<<<ceilf(testSize/(float)NTHREADS), NTHREADS>>>(d_ones_tx1, testSize, 1.0f);

	// d_M = d_theta * d_data (numClasses x testSize)
	mMul(handle, numClasses, testSize, inputSize, d_theta, d_data, d_M);

	// d_aux_tx1 = max(d_m) (testSize)
	columnMax<<<ceilf(testSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, numClasses, testSize);

	// d_M = d_M - repmat(d_aux_tx1, testSize, 1)
	mSubMax(handle, numClasses, testSize, d_M, d_aux_tx1, d_ones_tx1);

	// d_M = exp(d_M)
	Exp<<<ceilf(numClasses * testSize/(float)NTHREADS), NTHREADS>>>(d_M,d_M,numClasses*testSize);

	// d_aux_tx1 = sum(d_M) (testSize: sum of columns)
	columnSum<<<ceilf(testSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, numClasses, testSize);

	// d_M = bsxfun(@rdivide, d_M, d_aux_tx1) (Divide column j of d_M by d_aux_tx1(j)) This is the prediction matrix P
	rdivide<<<ceilf(numClasses * testSize/(float)NTHREADS), NTHREADS>>>(d_M,d_aux_tx1,numClasses, testSize);

	// d_aux_tx1 = max(d_M), contains the max probable class 
	columnMaxIndex<<<ceilf(testSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, numClasses, testSize);

	float *d_acc;
	float acc = 0.0f;
	CUDA_SAFE_CALL(cudaMalloc(&d_acc, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(d_acc, &acc, sizeof(float), cudaMemcpyHostToDevice));
	accuracy<<<1, 1>>>(d_label, d_aux_tx1, testSize, d_acc);
	CUDA_SAFE_CALL(cudaMemcpy(&acc, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
	//show_device("P", d_aux_tx1, testSize, 1);
	//show_device("label", d_label, testSize, 1);

	std::cout << "Accuracy: " << acc/testSize << std::endl;
	
	CUDA_SAFE_CALL(cudaFree(d_acc));
	CUBLAS_SAFE_CALL(cublasDestroy(handle));
	CUDA_SAFE_CALL(cudaFree(d_M));
	CUDA_SAFE_CALL(cudaFree(d_aux_tx1));
	CUDA_SAFE_CALL(cudaFree(d_ones_tx1));
}

#endif