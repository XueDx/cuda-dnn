#ifndef SPARSE_AUTOENCODER_HELPER_CUH
#define	SPARSE_AUTOENCODER_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>

/******************************************************************************
****** Constant parameters ****************************************************
*******************************************************************************/
#define NTHREADS 640

#define SPARSITY_PARAM 0.1f
#define BETA 3.0f
#define LAMBDA 0.003f // Weight decay parameter
#define EPS_COST 0.000001f // Training stops if |new_cost - old_cost| < EPS_COST

/******************************************************************************
****** Utilities **************************************************************
*******************************************************************************/
const char* cublasGetErrorString(cublasStatus_t status);

#define CUDA_SAFE_CALL(call) do { cudaError_t err = call; \
	if(err != cudaSuccess) { \
		printf("Error at %s:%d\nError: %s\n",__FILE__,__LINE__, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE);}} while(0)

#define CUBLAS_SAFE_CALL(call) do { cublasStatus_t err = call; \
	if(err != CUBLAS_STATUS_SUCCESS) { \
		printf("Cublas error at %s:%d\nError: %s\n",__FILE__,__LINE__, cublasGetErrorString(call)); \
		exit(EXIT_FAILURE);}} while(0)

#define ERROR_REPORT(description) do { \
	printf("Error at %s:%d\nDescription: %s\n",__FILE__,__LINE__, description); \
    exit(EXIT_FAILURE);} while(0)

#define PROFILE(text, call) do {  \
	long startTime = clock(); \
	call; \
	long finishTime = clock(); \
	std::cout<< text << ": " << (finishTime - startTime) / (double) CLOCKS_PER_SEC << " seconds" << std::endl; \
	} while(0)

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/******************************************************************************
****** My kernels <3 **********************************************************
*******************************************************************************/
// Initialize curand random number generator
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

// vec = normal random distribution
__global__ void initialize_normal(float* vec, int m, int n, curandState* globalState){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = globalState[id];
    globalState[id] = localState; 
	float r = 0.005;
	if(id < n*m)
		vec[id] = curand_normal( &localState )*r; //rand	
		globalState[id] = localState;
}

// vec = uniform random distribution
__global__ void initialize_uniform(float* vec, int m, int n, curandState* globalState){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = globalState[id];
    globalState[id] = localState; 
	float r = fdividef(sqrtf(6), sqrtf(n+m+1));
	if(id < n*m){
		vec[id] = curand_uniform( &localState )*2*r - r; //rand	
		globalState[id] = localState;
	}
}

// vec = f
__global__ void initialize_float(float* vec, int size, float f){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
		vec[id] = f;
}

// Calculate the groundtruth matrix used by softmax
// It's the matrix obtained by the replacement of each element of label' by a column vector.
// The column vector is the binary codification of the label.
__global__ void initialize_groundtruth(float* groundtruth, float* label, int numClasses, int trainSize){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < numClasses * trainSize){
		if(id % numClasses != label[id/numClasses])
			groundtruth[id] = 0.0f;
		else
			groundtruth[id] = 1.0f;
	}
}

// A = repmat(vec,1,m)
// Store in each column of A a copy of vec
__global__ void copyVecMatrix(float* vec, float* A, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m * n)
		A[id] = vec[id % m];	
}

// result = 1/(1 + exp(-m))
__global__ void Logistic(float* m, float* result, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	{
		result[id] = fdividef(1.0f, 1.0f + expf(-m[id]));
	}
}

// result = exp(m)
__global__ void Exp(float* m, float* result, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	{
		result[id] = expf(m[id]);
	}
}

// result = log(m)
__global__ void Log(float* m, float* result, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	{
		result[id] = logf(m[id]);
	}
}

// m = m .^ 2
__global__ void Square(float* m, float* result, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	{
		result[id] = m[id]*m[id];
	}
}

// mean = sumRows(A)/numCols(A)
// Calculate the mean value of each row of A and stores in mean.
__global__ void meanFeatures(float* mean, float* A, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m){
		mean[id] = 0.0f;
		for(int i = 0; i < n; i++){
			mean[id] += A[id + m*i];
		}
		mean[id] = mean[id]/n;	
	}
}

// rho_mod =  BETA*( -SPARSITY_PARAM ./ rho + (1-SPARSITY_PARAM) ./ (1-rho[id]))
__global__ void modifyRho(float* rho, float* rho_mod, int m){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m){
		rho_mod[id] = BETA*(fdividef(-SPARSITY_PARAM, rho[id]) + fdividef(1-SPARSITY_PARAM, 1-rho[id]));
	}
}

// rho_mod = SPARSITY_PARAM*log(SPARSITY_PARAM ./ rho) + (1-SPARSITY_PARAM)*log((1-SPARSITY_PARAM) ./ (1-rho))
__global__ void sparsityCost(float* rho, float* rho_mod, int m){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m){
		rho_mod[id] = SPARSITY_PARAM*logf(fdividef(SPARSITY_PARAM, rho[id])) + (1-SPARSITY_PARAM)*logf(fdividef(1-SPARSITY_PARAM, 1-rho[id]));
	}
}

// B = alpha*A + beta*B
__global__ void mAdd(float* A, float* B, int m, int n, float alpha, float beta){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m*n){
		B[id] = alpha*A[id] + beta*B[id];
	}
}

// C = A - B
__global__ void mSub(float* A, float* B, float* C, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m*n){
		C[id] = A[id] - B[id];
	}
}

// B = A .* (1-A)
__global__ void logisticGrad(float* A, float* B, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m*n){
		B[id] = A[id]*(1 - A[id]);
	}
}

// A = A/x (column m of A is divided by x[m])
__global__ void rdivide(float* A, float* x, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m*n){
		A[id] = A[id]/x[id/m];
	}
}

// B = -A .* (1-A)
__global__ void minusLogisticGrad(float* A, float* B, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m*n){
		B[id] = -A[id]*(1 - A[id]);
	}
}

// C = A .* B
// It's the Hadamard product of the two matrices
__global__ void mHad(float* A, float* B, float* C,int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < m*n){
		C[id] = A[id]*B[id];
	}
}

// A = A - f
// Subtract each element of A by f
__global__ void mfSub(float* A, int size, float* f){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < size)
		A[id] = A[id] - *f;
}

// max = max(x)
// Calculate the maximum element of x
__device__ void maxVec(float* x, int size, float *max){
	*max = FLT_MIN;
	for(int i = 0; i < size; i++)
		if(abs(x[i]) > abs(*max))
			*max = x[i];
}

// index = maxIndex(x)
// Calculate the index of the maximum element of x
__device__ void maxVecIndex(float* x, int size, float *index){
	float max = FLT_MIN;
	*index = 0.0;
	for(int i = 0; i < size; i++)
		if(abs(x[i]) > abs(max)){
			max = x[i];
			*index = i;
		}
}

// x = maxValueEachColumn(A)
// Calculate the maximum value of each column of A and store in x
__global__ void columnMax(float* A, float* x, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n)
		maxVec(&A[id*m], m, &x[id]);
}

// x = maxIndexEachColumn(A)
// Calculate the index of maximum value of each column of A and store in x
__global__ void columnMaxIndex(float* A, float* x, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n)
		maxVecIndex(&A[id*m], m, &x[id]);
}

// sum = sum(x)
// Sum all the elements of vector x
__device__ void sumVec(float* x, int size, float *sum){
	*sum = 0;
	for(int i = 0; i < size; i++)
		*sum += x[i];
}

// x = sumColumn(A)
// Calculate the sum of each column of A and stores in x
__global__ void columnSum(float* A, float* x, int m, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n)
		sumVec(&A[id*m], m, &x[id]);
}


// Compute the accuracy of prediction given label and stores the result in acc
__global__ void accuracy(float* label, float* prediction, int size, float* acc){
	for( int i = 0; i < size; i++)
		if(label[i] == prediction[i])
			*acc += 1.0f;
}

//void bsxfunMinusMax(cublasHandle_t handle, float* A, int m, int n){
//	int index = 0;
//
//	for(int j = 0; j < n; j++){
//		cublasIsamax(handle, m, &A[j*m], 1, &index);
//		index = index - 1;
//		mfSub<<<ceilf(m/(float)NTHREADS), NTHREADS>>>(&A[j*m], m, &A[j*m+index]);
//	}
//}

/******************************************************************************
****** Cublas wrappers ********************************************************
*******************************************************************************/
// C = A*B
// C = C + v * v_ones' (same in Matlab: C = C + repmat(v, 1, m))
void mMulSum(cublasHandle_t handle, int m, int n, int k,float* A, float* B,float* C,float* v, float* v_ones){
	float alpha = 1.0f;
	float beta = 0.0f;
	CUBLAS_SAFE_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k, &alpha,A,m,B,k,&beta,C,m)); // C = A*B
	CUBLAS_SAFE_CALL(cublasSger(handle,m, n,&alpha,v,1,v_ones,1,C,m)); // C <- v * v_ones' + C (same in Matlab: C = C + repmat(v, 1, n))
}

// A = A - v_max * v_ones'
// We use that to subtract the max value of each column (v_max) from the column.
void mSubMax(cublasHandle_t handle, int m, int n, float* A, float* v_max, float* v_ones){
	float alpha = -1.0f;
	CUBLAS_SAFE_CALL(cublasSger(handle,m, n,&alpha,v_ones,1,v_max,1,A,m)); // A <- A - v_max * v_ones' (same in Matlab: A = A - repmat(v_max, 1, n))
}

// y = A*x
void mvMul(cublasHandle_t handle, int m, int n,float* A, float* x,float* y){
	float alpha = 1.0f;
	float beta = 0.0f;
	CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1));
}

// C = A*B
void mMul(cublasHandle_t handle, int m, int n, int k,float* A, float* B,float* C){
	float alpha = 1.0f;
	float beta = 0.0f;
	CUBLAS_SAFE_CALL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k, &alpha,A,m,B,k,&beta,C,m));
}

//C = A'*B + C
void mMulTL(cublasHandle_t handle, int m, int n, int k,float* A, float* B,float* C){
	float alpha = 1.0f; 
	float beta = 1.0f;
	CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_T ,CUBLAS_OP_N,m,n,k, &alpha,A,k,B,k,&beta,C,m));
}

//C = A'*B
void mMulTLnosum(cublasHandle_t handle, int m, int n, int k,float* A, float* B,float* C){
	float alpha = 1.0f; 
	float beta = 0.0f;
	CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_T ,CUBLAS_OP_N,m,n,k, &alpha,A,k,B,k,&beta,C,m));
}

//C = A*B'
// m is number of rows of resulting matrix
// n is number of columns of resulting matrix
// k is supressed dimension
void mMulTR(cublasHandle_t handle, int m, int n, int k,float* A, float* B,float* C){
	float alpha = 1.0f; 
	float beta = 0.0f;
	CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N ,CUBLAS_OP_T,m,n,k, &alpha,A,m,B,n,&beta,C,m));
}

//C = alpha*A*B'
// m is number of rows of resulting matrix
// n is number of columns of resulting matrix
// k is supressed dimension
void mMulTRf(cublasHandle_t handle, int m, int n, int k, float* alpha, float* A, float* B,float* C){
	float beta = 0.0f;
	CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N ,CUBLAS_OP_T,m,n,k, alpha,A,m,B,n,&beta,C,m));
}

/******************************************************************************
****** Debug functions and save/load utils ************************************
*******************************************************************************/
// Print matrix A located in host memory
void show(char* name, float* A, int m, int n){
	std::cout << name << std::endl;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			std::cout << std::fixed << std::setw( 8 ) << std::setprecision( 6 ) << A[IDX2C(i,j,m)] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// Print matrix d_A located in device memory
void show_device(char* name, float* d_A, int m, int n){
	float* A = new(float[m * n]);
	cudaMemcpy(A, d_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << name << std::endl;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			std::cout << std::fixed << std::setw( 6 ) << std::setprecision( 4 ) << A[IDX2C(i,j,m)] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// Save matrix A located in host memory
void save(char* filename, float* A, int size){
	std::ofstream outdata;
	outdata.open(filename);
	if( !outdata ) { // file couldn't be opened
      ERROR_REPORT("Error: file could not be opened");
	}

	for (int i = 0; i< size; ++i)
      outdata << A[i] << std::endl;
   outdata.close();
}

// Save matrix d_A located in device memory
void save_device(char* filename, float* d_A, int size){
	float* A = new(float[size]);
	cudaMemcpy(A, d_A, size * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outdata;
	outdata.open(filename);
	if( !outdata ) { // file couldn't be opened
      ERROR_REPORT("Error: file could not be opened");
	}

	for (int i = 0; i< size; ++i)
      outdata << A[i] << std::endl;
   outdata.close();
}

// Load matrix A located in filename
void load(char* filename, float* x, int size){
	std::ifstream indata;

	indata.open(filename);
	if( !indata ) { // file couldn't be opened
      ERROR_REPORT("Error: file could not be opened");
	}

	for (int i = 0; i< size; ++i)
      indata >> x[i];
   indata.close();
}
const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

#endif	/* SPARSE_AUTOENCODER_HELPER_CUH */