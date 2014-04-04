#ifndef SPARSEAUTOENCODER_H
#define SPARSEAUTOENCODER_H
 
#include "helper.cuh"

class SparseAutoencoder
{
public:
    int hiddenSize;
    int visibleSize;
    float* d_W1;
	float* d_b1;
 
    SparseAutoencoder(int visible, int hidden) : hiddenSize(hidden), visibleSize(visible) {
		CUDA_SAFE_CALL(cudaMalloc(&d_W1, hiddenSize * visibleSize * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&d_b1, hiddenSize * sizeof(float)));
	}

	~SparseAutoencoder() {
		CUDA_SAFE_CALL(cudaFree(d_W1));
		CUDA_SAFE_CALL(cudaFree(d_b1));
	}

	void train(float* data, int trainSize, int maxIter);
	float* feature(float* data, int trainSize);
};

void SparseAutoencoder::train(float* d_data, int trainSize, int maxIter){
	curandState* devStates;
	cublasHandle_t handle;
	float* d_W2;
	float* d_b2;
	float* d_W1grad;
	float* d_W2grad;
	float* d_b1grad;
	float* d_b2grad;
	float* d_a2;
	float* d_z2;
	float* d_z2grad;
	float* d_d2;
	float* d_a3;
	float* d_z3;
	float* d_z3grad;
	float* d_d3;
	float* d_rho;
	float* d_rho_mod;
	float* d_ones_tx1;
	float* d_aux_tx1;
	float* d_aux_vx1;
	float* d_aux_hx1;
	float* d_AUX_vxh;
	
	float alpha = 0.1f; //learning rate
	float one_over_trainSize = 1.0f/trainSize; // needed in some methods

	float sum;
	float prev_cost = 0.0f;
	float cost = 0.0f;
	CUDA_SAFE_CALL(cudaMalloc(&d_W2, visibleSize * hiddenSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_b2, visibleSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_W1grad, hiddenSize * visibleSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_W2grad, visibleSize * hiddenSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_b1grad, hiddenSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_b2grad, visibleSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ones_tx1, trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_a2, hiddenSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_z2, hiddenSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_z2grad, hiddenSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_d2, hiddenSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_a3, visibleSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_z3, visibleSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_z3grad, visibleSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_d3, visibleSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_rho, hiddenSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_rho_mod, hiddenSize * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&devStates, visibleSize * hiddenSize*sizeof( curandState )));
	
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_tx1, trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_vx1, visibleSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_aux_hx1, hiddenSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_AUX_vxh, visibleSize * hiddenSize * sizeof(float)));

	CUBLAS_SAFE_CALL(cublasCreate(&handle));
	
	// Random initialize parameters
	setup_kernel<<<ceilf(visibleSize * hiddenSize/(float)NTHREADS), NTHREADS>>>( devStates, 0 );
	initialize_uniform<<<ceilf(visibleSize * hiddenSize/(float)NTHREADS), NTHREADS>>>(d_W1, hiddenSize, visibleSize, devStates);
	initialize_uniform<<<ceilf(visibleSize * hiddenSize/(float)NTHREADS), NTHREADS>>>(d_W2, visibleSize, hiddenSize, devStates);
	initialize_float<<<ceilf(hiddenSize/(float)NTHREADS), NTHREADS>>>(d_b1, hiddenSize, 0.0f);
	initialize_float<<<ceilf(visibleSize/(float)NTHREADS), NTHREADS>>>(d_b2, visibleSize, 0.0f);
	initialize_float<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_ones_tx1, trainSize, 1.0f);

	//Backpropagation
	for(int iteration = 1; iteration <= maxIter; iteration++){
		/******************************************************************************
		****** Feedforward ************************************************************
		*******************************************************************************/
		//Calculate z2 and a2
		mMulSum(handle, hiddenSize, trainSize,visibleSize, d_W1, d_data, d_z2,d_b1,d_ones_tx1);
		Logistic<<<ceilf(hiddenSize*trainSize/(float)NTHREADS),NTHREADS>>>(d_z2,d_a2,hiddenSize*trainSize); // d_a2 <- Logistic(d_z2)
		
		//Calculate z3 and a_3
		mMulSum   (handle, visibleSize, trainSize,hiddenSize, d_W2, d_a2, d_z3,d_b2,d_ones_tx1);
		Logistic<<<ceilf(visibleSize*trainSize/(float)NTHREADS),NTHREADS>>>(d_z3,d_a3,visibleSize*trainSize); // d_a3 <- Logistic(d_z3)

		//Calculate rho
		mvMul(handle,hiddenSize, trainSize,d_a2,d_ones_tx1,d_rho);
		CUBLAS_SAFE_CALL(cublasSscal(handle,hiddenSize, &one_over_trainSize, d_rho, 1));

		//Calculate d3
		mSub<<<ceilf(visibleSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_data,d_a3, d_d3, visibleSize, trainSize);
		minusLogisticGrad<<<ceilf(visibleSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_a3, d_z3grad, visibleSize, trainSize);
		mHad<<<ceilf(visibleSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_d3,d_z3grad, d_d3, visibleSize, trainSize);
			
		//Calculate d2
		modifyRho<<<ceilf(hiddenSize/(float)NTHREADS), NTHREADS>>>(d_rho, d_rho_mod, hiddenSize);
		copyVecMatrix<<<ceilf(hiddenSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_rho_mod,d_d2,hiddenSize,trainSize); //Calculate repmat(rho_mod,1,m) into d_d2;
		mMulTL(handle, hiddenSize, trainSize,visibleSize, d_W2, d_d3, d_d2);
		logisticGrad<<<ceilf(hiddenSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_a2, d_z2grad, hiddenSize, trainSize);
		mHad<<<ceilf(hiddenSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_d2,d_z2grad, d_d2, hiddenSize, trainSize);

		/******************************************************************************
		****** Gradients **************************************************************
		*******************************************************************************/
		//Calcule W1grad
		mMulTR(handle, hiddenSize, visibleSize,trainSize, d_d2, d_data, d_W1grad); //d_W1grad <- Delta_W1
		mAdd<<<ceilf(hiddenSize*visibleSize/(float)NTHREADS), NTHREADS>>>(d_W1, d_W1grad,hiddenSize,visibleSize,LAMBDA,one_over_trainSize);

		//Calcule W2grad
		mMulTR(handle, visibleSize, hiddenSize,trainSize, d_d3, d_a2, d_W2grad); //d_W1grad <- Delta_W1
		mAdd<<<ceilf(hiddenSize*visibleSize/(float)NTHREADS), NTHREADS>>>(d_W2, d_W2grad,visibleSize,hiddenSize,LAMBDA,one_over_trainSize);

		//Calcule b1grad
		mvMul(handle,hiddenSize, trainSize,d_d2,d_ones_tx1,d_b1grad); //Mult
		CUBLAS_SAFE_CALL(cublasSscal(handle,hiddenSize, &one_over_trainSize, d_b1grad, 1)); //Multiply by one over m

		//Calcule b2grad
		mvMul(handle,visibleSize, trainSize,d_d3,d_ones_tx1,d_b2grad);
		CUBLAS_SAFE_CALL(cublasSscal(handle,visibleSize, &one_over_trainSize, d_b2grad, 1));

		/******************************************************************************
		****** Cost *******************************************************************
		*******************************************************************************/
		prev_cost = cost;

		mSub<<<ceilf(visibleSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_data,d_a3, d_d3, visibleSize, trainSize);
		Square<<<ceilf(visibleSize*trainSize/(float)NTHREADS), NTHREADS>>>(d_d3,d_d3, visibleSize*trainSize);
		columnSum<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_d3, d_aux_tx1, visibleSize, trainSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, trainSize, d_aux_tx1, 1, &sum));
		cost = sum/(2*trainSize);
		
		Square<<<ceilf(visibleSize*hiddenSize/(float)NTHREADS), NTHREADS>>>(d_W2,d_AUX_vxh, visibleSize*hiddenSize);
		columnSum<<<ceilf(hiddenSize/(float)NTHREADS), NTHREADS>>>(d_AUX_vxh, d_aux_hx1, visibleSize, hiddenSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, hiddenSize, d_aux_hx1, 1, &sum));
		cost += LAMBDA*sum/2;

		Square<<<ceilf(visibleSize*hiddenSize/(float)NTHREADS), NTHREADS>>>(d_W1,d_AUX_vxh, visibleSize*hiddenSize);
		columnSum<<<ceilf(visibleSize/(float)NTHREADS), NTHREADS>>>(d_AUX_vxh, d_aux_vx1, hiddenSize, visibleSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, hiddenSize, d_aux_vx1, 1, &sum));
		cost += LAMBDA*sum/2;
		
		sparsityCost<<<ceilf(hiddenSize/(float)NTHREADS), NTHREADS>>>(d_rho, d_rho_mod, hiddenSize);
		CUBLAS_SAFE_CALL(cublasSasum(handle, hiddenSize, d_rho_mod, 1, &sum));
		cost += BETA*sum;
		
		std::cout <<  iteration << ": " << cost << std::endl;

		if(fabs(cost - prev_cost) < EPS_COST)
			break;
		// Dynamic learning rate (yes, need work)
		if(cost > prev_cost)
			alpha = alpha*0.5f;
		else
			if(alpha < 0.1f/(0.1f + iteration/maxIter))
				alpha = alpha*1.1f;

		/******************************************************************************
		****** Update the weights *****************************************************
		*******************************************************************************/
		//Update d_W1, d_W2, d_b1 and d_b2 with learning rate alpha
		mAdd<<<ceilf(hiddenSize*visibleSize/(float)NTHREADS), NTHREADS>>>(d_W1grad, d_W1,hiddenSize, visibleSize,-alpha,1.0f);
		mAdd<<<ceilf(hiddenSize*visibleSize/(float)NTHREADS), NTHREADS>>>(d_W2grad, d_W2,visibleSize, hiddenSize,-alpha,1.0f);
		mAdd<<<ceilf(hiddenSize/(float)NTHREADS), NTHREADS>>>(d_b1grad, d_b1,hiddenSize, 1,-alpha,1.0f);
		mAdd<<<ceilf(visibleSize/(float)NTHREADS), NTHREADS>>>(d_b2grad, d_b2,visibleSize, 1,-alpha,1.0f);	
	}

	//Release resources
	CUBLAS_SAFE_CALL(cublasDestroy(handle));
	CUDA_SAFE_CALL(cudaFree(devStates));
	CUDA_SAFE_CALL(cudaFree(d_z2));
	CUDA_SAFE_CALL(cudaFree(d_z2grad));
	CUDA_SAFE_CALL(cudaFree(d_a2));
	CUDA_SAFE_CALL(cudaFree(d_d2));
	CUDA_SAFE_CALL(cudaFree(d_z3));
	CUDA_SAFE_CALL(cudaFree(d_z3grad)); 
	CUDA_SAFE_CALL(cudaFree(d_a3));
	CUDA_SAFE_CALL(cudaFree(d_d3));
	CUDA_SAFE_CALL(cudaFree(d_rho));
	CUDA_SAFE_CALL(cudaFree(d_rho_mod));
	CUDA_SAFE_CALL(cudaFree(d_W1grad));
	CUDA_SAFE_CALL(cudaFree(d_W2grad));
	CUDA_SAFE_CALL(cudaFree(d_b1grad));
	CUDA_SAFE_CALL(cudaFree(d_b2grad));
	CUDA_SAFE_CALL(cudaFree(d_aux_tx1));
	CUDA_SAFE_CALL(cudaFree(d_aux_vx1));
	CUDA_SAFE_CALL(cudaFree(d_aux_hx1));
	CUDA_SAFE_CALL(cudaFree(d_AUX_vxh));
	CUDA_SAFE_CALL(cudaFree(d_W2));
	CUDA_SAFE_CALL(cudaFree(d_b2));
	CUDA_SAFE_CALL(cudaFree(d_ones_tx1));
}

float* SparseAutoencoder::feature(float* d_data, int trainSize){
	cublasHandle_t handle;
	float* d_a2;
	float* d_ones_tx1;

	// Allocate GPU resources
	CUDA_SAFE_CALL(cudaMalloc(&d_a2, hiddenSize * trainSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ones_tx1, trainSize * sizeof(float)));
	CUBLAS_SAFE_CALL(cublasCreate(&handle));
	
	// Compute activation
	initialize_float<<<ceilf(trainSize/(float)NTHREADS), NTHREADS>>>(d_ones_tx1, trainSize, 1.0f);

	mMulSum(handle, hiddenSize, trainSize,visibleSize, d_W1, d_data, d_a2,d_b1,d_ones_tx1);
	Logistic<<<ceilf(hiddenSize*trainSize/(float)NTHREADS),NTHREADS>>>(d_a2,d_a2,hiddenSize*trainSize); // d_a2 <- Logistic(d_z2)
		
	// Release GPU resources
	CUBLAS_SAFE_CALL(cublasDestroy(handle));
	CUDA_SAFE_CALL(cudaFree(d_ones_tx1));
	return d_a2;
}

#endif