#ifndef STACKAUTOENCODER_H
#define STACKAUTOENCODER_H
 
#include "helper.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "sparse_autoencoder.h"
#include "softmax.h"

// Maximum number of stacked autoencoders
#define STACK_DEPTH 2

// Neural Network containing many autoencoders hidden layers and one softmax output layer.
class StackAutoencoder
{
public:
	int numAEs;
	Softmax* sm;
	SparseAutoencoder* sa[STACK_DEPTH];
  
	StackAutoencoder(): numAEs(0) {}
	~StackAutoencoder() {
		delete sm;
		for(int i = 0; i < numAEs; i++){
			delete sa[i];
		}
	}

	// Add one autoencoder layer (from bottom to top)
	void addAutoencoder(SparseAutoencoder* ae);

	// Add softmax layer (after adding the autoencoder layers)
	void addSoftmax(Softmax* sm_model);

	// Pre-train the neural network usign data and label as training set
	void train(float* data, float* label, int trainSize, int maxIter);

	// Fine tune the neural network usign data and label as training set
	void fineTune(float* data, float* label, int trainSize, int maxIter);

	// Compute the accuracy of the network on testData
	void test(float* testData, float* testLabel, int testSize);

	void predict(float* d_predData, float* d_predLabel, int predSize);
};

void StackAutoencoder::addAutoencoder(SparseAutoencoder* ae){
		if(numAEs >= STACK_DEPTH)
			ERROR_REPORT("Exceeded maximum number of autoencoders layers.");

		if(numAEs > 1)
			if(ae->visibleSize != sa[numAEs-1]->hiddenSize)
				ERROR_REPORT("Autoencoders dimensions do not match.");

		sa[numAEs++] = ae;
	}

void StackAutoencoder::addSoftmax(Softmax* sm_model){
		if(numAEs == 0)
			ERROR_REPORT("Cannot add softmax layer without autoencoders layers.");
		if(sm_model->inputSize != sa[numAEs-1]->hiddenSize)
			ERROR_REPORT("Softmax dimension do not match top autoencoder.");
		sm = sm_model;
	}

void StackAutoencoder::train(float* d_data, float* d_label, int trainSize, int maxIter){
	float* saFeature = d_data;
	for(int i = 0; i < numAEs; i++){
		sa[i]->train(saFeature, trainSize, maxIter);
		saFeature = sa[i]->feature(saFeature, trainSize);
	}
	sm->train(saFeature, d_label, trainSize, maxIter*10);
}

void StackAutoencoder::fineTune(float* d_data, float* d_label, int trainSize, int maxIter){
	cublasHandle_t handle;
	float* d_thetagrad;
	float* d_groundtruth;
	float* d_M;
	float* d_aux_tx1;
	float* d_aux_ix1;
	float* d_AUX_nxt;
	float* d_AUX_nxi;
	float* d_ones_tx1;
	float* d_ones_nx1;
	float sum;
	float cost = 0.0f;
	float prev_cost = 0.0f;
	float alpha = 1.0f; //learning rate
	float one_over_trainSize = 1.0f/trainSize; // needed in some methods

	float* z[STACK_DEPTH+1];
	float* a[STACK_DEPTH+1];
	float* d[STACK_DEPTH+1];
	float* Wgrad[STACK_DEPTH];
	float* bgrad[STACK_DEPTH];

	CUDA_SAFE_CALL(cudaMalloc(&d_thetagrad, sm->numClasses * sm->inputSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_groundtruth, sm->numClasses * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_M, sm->numClasses * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_AUX_nxt, sm->numClasses * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_AUX_nxi, sm->numClasses * sm->inputSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_tx1, trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_ix1, sm->inputSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ones_tx1, trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ones_nx1, sm->numClasses * sizeof(float)));
	CUBLAS_SAFE_CALL(cublasCreate(&handle));
	for(int i = 0; i < numAEs; i++){
		CUDA_SAFE_CALL(cudaMalloc(&z[i], sa[i]->hiddenSize * trainSize * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&a[i], sa[i]->hiddenSize * trainSize * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&d[i], sa[i]->hiddenSize * trainSize * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&Wgrad[i], sa[i]->hiddenSize * sa[i]->visibleSize * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&bgrad[i], sa[i]->hiddenSize * sizeof(float)));
	}
	CUDA_SAFE_CALL(cudaMalloc(&z[numAEs], sm->inputSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&a[numAEs], sm->inputSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d[numAEs], sm->inputSize * trainSize * sizeof(float)));

	// Initialization
	initialize_groundtruth<<<ceilf(sm->numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_groundtruth, d_label, sm->numClasses, trainSize);
	initialize_float<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_ones_tx1, trainSize, 1.0f);
	initialize_float<<<ceilf(sm->numClasses/(float)NTHREADS), NTHREADS>>>(d_ones_nx1, sm->numClasses, 1.0f);

	// Gradient Descent
	a[0] = d_data;
	for(int iteration = 1; iteration <= maxIter; iteration++){
		/******************************************************************************
		****** Feedforward ************************************************************
		*******************************************************************************/
		for(int i = 0; i < numAEs; i++){
			mMulSum   (handle, sa[i]->hiddenSize, trainSize, sa[i]->visibleSize, sa[i]->d_W1, a[i], z[i+1], sa[i]->d_b1, d_ones_tx1);
			Logistic<<<ceilf(sa[i]->hiddenSize*trainSize/(float)NTHREADS),NTHREADS>>>(z[i+1],a[i+1],sa[i]->hiddenSize*trainSize); // d_a2 <- Logistic(d_z2)

		}

		/******************************************************************************
		****** Softmax ****************************************************************
		*******************************************************************************/
		// d_M = d_theta * saFeature (numClasses x trainSize)
		mMul(handle, sm->numClasses, trainSize, sm->inputSize, sm->d_theta, a[numAEs], d_M);

		// d_aux_tx1 = max(d_m) (trainSize)
		columnMax<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, sm->numClasses, trainSize);

		// d_M = d_M - repmat(d_aux_tx1, trainSizem, 1)
		mSubMax(handle, sm->numClasses, trainSize, d_M, d_aux_tx1, d_ones_nx1);

		// d_M = exp(d_M)
		Exp<<<ceilf(sm->numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_M,d_M,sm->numClasses*trainSize);

		// d_aux_tx1 = sum(d_M) (trainSize: sum of columns)
		columnSum<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_M, d_aux_tx1, sm->numClasses, trainSize);

		// d_M = bsxfun(@rdivide, d_M, d_aux_tx1) (Divide column j of d_M by d_aux_tx1(j)) This is the prediction matrix P
		rdivide<<<ceilf(sm->numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_M,d_aux_tx1,sm->numClasses, trainSize);

		/******************************************************************************
		****** Cost *******************************************************************
		*******************************************************************************/
		prev_cost = cost;

		Log<<<ceilf(sm->numClasses*trainSize/(float)NTHREADS), NTHREADS>>>(d_M,d_AUX_nxt, sm->numClasses*trainSize);
		mHad<<<ceilf(sm->numClasses*trainSize/(float)NTHREADS), NTHREADS>>>(d_AUX_nxt,d_groundtruth, d_AUX_nxt, sm->numClasses, trainSize);
		columnSum<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_AUX_nxt, d_aux_tx1, sm->numClasses, trainSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, trainSize, d_aux_tx1, 1, &sum));
		cost = sum/trainSize;

		Square<<<ceilf(sm->numClasses*sm->inputSize/(float)NTHREADS), NTHREADS>>>(sm->d_theta,d_AUX_nxi, sm->numClasses*sm->inputSize);
		columnSum<<<ceilf(sm->inputSize/(float)NTHREADS), NTHREADS>>>(d_AUX_nxi, d_aux_ix1, sm->numClasses, sm->inputSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, sm->inputSize, d_aux_ix1, 1, &sum));
		cost += LAMBDA*sum/2.0f;
			
		std::cout <<  iteration << ": " << cost << std::endl;
		
		if(fabs(cost - prev_cost) < EPS_COST)
			break;
		// Dynamic learning rate
		if(cost > prev_cost)
			alpha = alpha*0.5f;
		else
			if(alpha < 0.5f/(0.1f + iteration/maxIter))
				alpha = alpha*1.1f;
		
		/******************************************************************************
		****** Gradients **************************************************************
		*******************************************************************************/
		// d_M = d_groundtruth - d_M
		mSub<<<ceilf(sm->numClasses * trainSize/(float)NTHREADS), NTHREADS>>>(d_groundtruth, d_M, d_M,sm->numClasses, trainSize);

		// d_thetagrad = d_M * saFeature'
		mMulTR(handle, sm->numClasses, sm->inputSize, trainSize, d_M, a[numAEs], d_thetagrad);

		// d_thetagrad = -(1/trainSize)*d_thetagrad + LAMBDA * d_theta
		mAdd<<<ceilf(sm->numClasses * sm->inputSize/(float)NTHREADS), NTHREADS>>>(sm->d_theta, d_thetagrad, sm->numClasses, sm->inputSize, LAMBDA, -one_over_trainSize);

		// Softmax layer delta
		minusLogisticGrad<<<ceilf(sm->inputSize * trainSize/(float)NTHREADS), NTHREADS>>>(a[numAEs], z[numAEs], sm->inputSize, trainSize);
		mMulTLnosum(handle,sm->inputSize, trainSize, sm->numClasses, sm->d_theta, d_M, d[numAEs]);
		mHad<<<ceilf(sm->inputSize * trainSize/(float)NTHREADS), NTHREADS>>>(d[numAEs], z[numAEs], d[numAEs], sm->inputSize, trainSize);

		// Hidden layer deltas
		for(int i = numAEs - 1; i > 0; i--){
			logisticGrad<<<ceilf(sa[i]->visibleSize * trainSize/(float)NTHREADS), NTHREADS>>>(a[i], z[i], sa[i]->visibleSize, trainSize);
			mMulTLnosum(handle,sa[i]->visibleSize, trainSize, sa[i]->hiddenSize, sa[i]->d_W1, d[i+1], d[i]);
			mHad<<<ceilf(sa[i]->visibleSize * trainSize/(float)NTHREADS), NTHREADS>>>(d[i], z[i], d[i], sa[i]->visibleSize, trainSize);
		}

		for(int i = numAEs - 1; i >= 0; i--){
			// W grad
			mMulTRf(handle,sa[i]->hiddenSize, sa[i]->visibleSize, trainSize, &one_over_trainSize, d[i+1], a[i], Wgrad[i]);

			// b grad
			mvMul(handle,sa[i]->hiddenSize, trainSize, d[i+1], d_ones_tx1, bgrad[i]);
			CUBLAS_SAFE_CALL(cublasSscal(handle, sa[i]->hiddenSize, &one_over_trainSize, bgrad[i], 1));
		}

		/******************************************************************************
		****** Update weights *********************************************************
		*******************************************************************************/
		// d_theta = d_theta - alpha*d_thetagrad
		mAdd<<<ceilf(sm->numClasses * sm->inputSize/(float)NTHREADS), NTHREADS>>>(d_thetagrad,sm->d_theta, sm->numClasses, sm->inputSize, -alpha, 1.0f);
		// w and b
		for(int i = 0; i < numAEs; i++){
			mAdd<<<ceilf(sa[i]->hiddenSize*sa[i]->visibleSize/(float)NTHREADS), NTHREADS>>>(Wgrad[i], sa[i]->d_W1, sa[i]->hiddenSize, sa[i]->visibleSize, -alpha, 1.0f);
			mAdd<<<ceilf(sa[i]->hiddenSize/(float)NTHREADS), NTHREADS>>>(bgrad[i], sa[i]->d_b1, sa[i]->hiddenSize, 1, -alpha, 1.0f);
		}
	}

	//Release resources
	CUBLAS_SAFE_CALL(cublasDestroy(handle));
	CUDA_SAFE_CALL(cudaFree(d_thetagrad)); 
	CUDA_SAFE_CALL(cudaFree(d_groundtruth));
	CUDA_SAFE_CALL(cudaFree(d_M)); 
	CUDA_SAFE_CALL(cudaFree(d_AUX_nxt)); 
	CUDA_SAFE_CALL(cudaFree(d_AUX_nxi));
	CUDA_SAFE_CALL(cudaFree(d_aux_tx1)); 
	CUDA_SAFE_CALL(cudaFree(d_aux_ix1)); 
	CUDA_SAFE_CALL(cudaFree(d_ones_tx1));
	CUDA_SAFE_CALL(cudaFree(d_ones_nx1));
	for(int i = 0; i < numAEs; i++){
		if(i != 0) // We don't want to release a[0] = d_data
			CUDA_SAFE_CALL(cudaFree(a[i])); 
		CUDA_SAFE_CALL(cudaFree(z[i])); 
		CUDA_SAFE_CALL(cudaFree(d[i])); 
		CUDA_SAFE_CALL(cudaFree(Wgrad[i])); 
		CUDA_SAFE_CALL(cudaFree(bgrad[i]));
	}
	CUDA_SAFE_CALL(cudaFree(a[numAEs])); 
	CUDA_SAFE_CALL(cudaFree(z[numAEs])); 
	CUDA_SAFE_CALL(cudaFree(d[numAEs])); 
}

void StackAutoencoder::test(float* testData, float* testLabel, int testSize){
	float* saFeature = testData;
	for(int i = 0; i < numAEs; i++)
		saFeature = sa[i]->feature(saFeature, testSize);
	sm->test(saFeature, testLabel, testSize);
}

void StackAutoencoder::predict(float* d_predData, float* d_predLabel, int predSize){
	float* saFeature = d_predData;
	for(int i = 0; i < numAEs; i++)
		saFeature = sa[i]->feature(saFeature, predSize);
	sm->predict(saFeature, d_predLabel, predSize);
}

#endif