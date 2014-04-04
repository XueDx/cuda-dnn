# CUDA Deep Neural Networks


This is an implementation of some Deep Neural Networks (DNN). We closely followed the [ULFDL Tutorial], but using C++/[CUDA] instead of Matlab or Octave.

Each neural network architecture is implemented in a separate class, some of them being composed of others. We already have working versions of the following architectures:

* Sparse autoencoder  (AE)
* Softmax regression  (SM)
* Stacked autoencoders (SAE)

## The Math

We give here, for reference, summarized information for each architecture. Actually, we give mainly the equations that we use in our code, so refer to the [ULFDL Tutorial] for complete explanations. Note that our equations may not look exactly like the ones there, as we will give vectorized versions working with batches of data simultaneously.  But first, some general notation:

| Symbol            | Description                                                 |
|:-----------------:|:------------------------------------------------------------|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20v%20%24%24)           | Data input size. The dimension of the feature vectors. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20t%20%24%24)           | Data train size. How many feature vectors to train. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BX%7D%20%24%24)  | Data matrix of dimensions ![equation](http://latex.codecogs.com/png.latex?%24%24%20v%20%5Ctimes%20t%20%24%24). Each column is a feature vector. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7By%7D%20%24%24)  | Label vector of dimension ![equation](http://latex.codecogs.com/png.latex?%24%24%20t%20%24%24). Element ![equation](http://latex.codecogs.com/png.latex?%24%24%20y%5E%7B%28i%29%7D%20%24%24) contains the label of feature vector ![equation](http://latex.codecogs.com/png.latex?%24%24%20x%5E%7B%28i%29%7D%20%24%24). | 
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7B1_m%7D%20%24%24)  | Vector of ones and dimension ![equation](http://latex.codecogs.com/png.latex?%24%24%20m%20%24%24). This is not the identity matrix ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bid_m%7D%24%24).|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7B1_%7Bmn%7D%7D%20%24%24)  | Matrix of ones and dimension ![equation](http://latex.codecogs.com/png.latex?%24%24%20m%20%5Ctimes%20n%24%24). This is not the identity matrix ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bid_%7Bmn%7D%7D%24%24).|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Clambda%20%24%24) | Weight decay parameter in the cost function. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Calpha%20%24%24) | Learning rate for gradient descent. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Coperatorname%7BS%7D%20%24%24)  | The sigmoid function. ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Coperatorname%7BS%7D%28x%29%20%3D%201/%281%2Be%5E%7B-x%7D%29%20%24%24) whatever ![equation](http://latex.codecogs.com/png.latex?%24%24x%24%24) may be (real or matrix).|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Coperatorname%7Bmax%7D%20%24%24)  | When applied to a matrix ![equation](http://latex.codecogs.com/png.latex?%24%24%20A%20%24%24), returns a vector with the maximum element of each column of ![equation](http://latex.codecogs.com/png.latex?%24%24%20A%20%24%24).|  
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Codot%20%24%24)  | Element-wise multiplication. The Hadamard product binary operator. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Coslash%20%24%24)  | Element-wise division. |

All vectors are considered as column matrices. 

You should notice that we try to give vectorized versions of each calculation. Sometimes we just need to sum all the elements of a matrix, but this operation can also be written in matrix form. In fact, given a matrix ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BA%7D%20%24%24) with dimensions ![equation](http://latex.codecogs.com/png.latex?%24%24%20m%20%5Ctimes%20n%20%24%24), we have:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Coperatorname%7Bsum%7D%28A%29%20%3A%3D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20A_%7Bij%7D%20%3D%20%5Cmathbf%7B1_m%7D%5ET%20%5Ccdot%20%5Cmathbf%7BA%7D%20%5Ccdot%20%5Cmathbf%7B1_n%7D%24%24).

In the code this may be implemented different, but this notation is useful.

### Sparse autoencoder
A sparse autoencoder is a neural network with a visible layer ![equation](http://latex.codecogs.com/png.latex?%24%24L_1%24%24), a hidden layer ![equation](http://latex.codecogs.com/png.latex?%24%24L_2%24%24) and an output layer ![equation](http://latex.codecogs.com/png.latex?%24%24L_3%24%24). It's purpose is the output the inputs the most faithful possible. This is not trivial, given that, in general, we have less neurons in the hidden layer than in the input layer.

We define the following:

| Symbol            | Description                                                 |
|:-----------------:|:------------------------------------------------------------|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20v%20%24%24)            | The dimension of the input vectors (and of the output too). ![equation](http://latex.codecogs.com/png.latex?%24%24L_1%24%24) size. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20h%20%24%24)            | The dimension of the hidden layer. ![equation](http://latex.codecogs.com/png.latex?%24%24L_2%24%24) size.  |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_1%7D%20%24%24) | Weight matrix of dimensions ![equation](http://latex.codecogs.com/png.latex?%24%24%20h%20%5Ctimes%20%20v%24%24). The weights between  ![equation](http://latex.codecogs.com/png.latex?%24%24L_1%24%24) and  ![equation](http://latex.codecogs.com/png.latex?%24%24L_2%24%24) . |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bb_1%7D%20%24%24) | Bias vector of dimension ![equation](http://latex.codecogs.com/png.latex?%24%24%20h%20%24%24). The bias of ![equation](http://latex.codecogs.com/png.latex?%24%24L_1%24%24) into ![equation](http://latex.codecogs.com/png.latex?%24%24L_2%24%24) . |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_2%7D%20%24%24) | Weight matrix of dimensions ![equation](http://latex.codecogs.com/png.latex?%24%24%20v%20%5Ctimes%20%20h%24%24). The weights between  ![equation](http://latex.codecogs.com/png.latex?%24%24L_2%24%24) and  ![equation](http://latex.codecogs.com/png.latex?%24%24L_3%24%24) . |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bb_2%7D%20%24%24) | Bias vector of dimension ![equation](http://latex.codecogs.com/png.latex?%24%24%20v%20%24%24). The bias of ![equation](http://latex.codecogs.com/png.latex?%24%24L_2%24%24) into ![equation](http://latex.codecogs.com/png.latex?%24%24L_3%24%24) . |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Crho%20%24%24) | Sparsity parameter. Controls the level of sparsity. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cbeta%20%24%24) | Weight of the sparsity penalty term in the cost function. |

* Initialize the ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW%7D%20%24%24) using a random uniform distribution and the bias ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bb%7D%20%24%24) to zeros.

To train the network, in each iteration we do:

* Compute the gradients:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bz_2%7D%20%3A%3D%20%20%5Cmathbf%7BW_1%7D%20%5Ccdot%20%5Cmathbf%7BX%7D%20%2B%20%5Cmathbf%7Bb_1%7D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%5ET%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Ba_2%7D%20%3A%3D%20%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Bz_2%7D%29%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bz_3%7D%20%3A%3D%20%20%5Cmathbf%7BW_2%7D%20%5Ccdot%20%5Cmathbf%7Ba_2%7D%20%2B%20%5Cmathbf%7Bb_2%7D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%5ET%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Ba_3%7D%20%3A%3D%20%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Bz_3%7D%29%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Chat%20%5Crho%7D%20%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%20%5Cmathbf%7Ba_2%7D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%5Ctext%7B%20%20%20%20%28it%27s%20just%20the%20mean%20of%20%7D%20%5Cmathbf%7Ba_2%7D%20%5Ctext%7B%29.%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cdelta_3%7D%20%3A%3D%20-%28%5Cmathbf%7BX%7D%20-%20%5Cmathbf%7Ba_3%7D%29%20%5Codot%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_3%7D%29%20%5Codot%20%28%5Cmathbf%7B1_%7Bvt%7D%7D%20-%20%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_3%7D%29%29%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cdelta_2%7D%20%3A%3D%20%20%28%5Cmathbf%7BW_2%7D%5ET%20%5Ccdot%20%5Cboldsymbol%7B%5Cdelta_3%7D%20%20%2B%20%20%5Cbeta%28-%5Crho%5Cmathbf%7B1_h%7D%20%5Coslash%20%5Cboldsymbol%7B%5Chat%20%5Crho%7D%20%2B%20%28%5Cmathbf%7B1_h%7D-%5Crho%5Cmathbf%7B1_h%7D%29%20%5Coslash%20%28%5Cmathbf%7B1_h%7D%20-%20%5Cboldsymbol%7B%5Chat%20%5Crho%7D%29%29%20%5Ccdot%20%5Cmathbf%7B1_t%7D%5ET%29%20%20%5Codot%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_2%7D%29%20%5Codot%20%28%5Cmathbf%7B1_%7Bht%7D%7D%20-%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_2%7D%29%29%20%24%24)
 
![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7BW_1%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%5Cboldsymbol%7B%5Cdelta_2%7D%20%5Ccdot%20%5Cmathbf%7BX%7D%5ET%20%2B%20%5Clambda%20%5Cmathbf%7BW_1%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7BW_2%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%5Cboldsymbol%7B%5Cdelta_3%7D%20%5Ccdot%20%5Cmathbf%7Ba_2%7D%5ET%20%2B%20%5Clambda%20%5Cmathbf%7BW_2%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7Bb_1%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%20%5Cboldsymbol%7B%5Cdelta_2%7D%20%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7Bb_2%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%20%5Cboldsymbol%7B%5Cdelta_3%7D%20%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%24%24)

* Compute the cost:

![equation](http://latex.codecogs.com/png.latex?%24%24%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%3A%3D%20%5Cfrac%7B1%7D%7B2t%7D%5Cmathbf%7B1_v%7D%5ET%20%5Ccdot%20%5B%28%5Cmathbf%7Ba_3%7D%20-%20%5Cmathbf%7BX%7D%29%20%5Codot%20%28%5Cmathbf%7Ba_3%7D%20-%20%5Cmathbf%7BX%7D%29%5D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2%7D%5B%5Cmathbf%7B1_h%7D%5ET%20%5Ccdot%20%28%5Cmathbf%7BW_1%7D%5Codot%20%5Cmathbf%7BW_1%7D%29%20%5Ccdot%20%5Cmathbf%7B1_v%7D%20%2B%20%5Cmathbf%7B1_v%7D%5ET%20%5Ccdot%20%28%5Cmathbf%7BW_2%7D%5Codot%20%5Cmathbf%7BW_2%7D%29%20%5Ccdot%20%5Cmathbf%7B1_h%7D%20%5D%20%2B%20%5Cbeta%5Cmathbf%7B1_h%7D%5ET%20%5Ccdot%20%5B%5Crho%20%5Clog%28%5Crho%5Cmathbf%7B1_h%7D%20%5Coslash%20%5Cboldsymbol%7B%5Chat%20%5Crho%7D%29%20%2B%20%281%20-%20%5Crho%29%5Clog%28%28%5Cmathbf%7B1_h%7D-%5Crho%5Cmathbf%7B1_h%7D%29%20%5Coslash%20%28%5Cmathbf%7B1_h%7D%20-%20%5Cboldsymbol%7B%5Chat%20%5Crho%7D%29%29%5D%20%24%24)

* Update the parameters:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_1%7D%20%3A%3D%20%5Cmathbf%7BW_1%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7BW_1%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bb_1%7D%20%3A%3D%20%5Cmathbf%7Bb_1%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7Bb_1%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_2%7D%20%3A%3D%20%5Cmathbf%7BW_2%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7BW_2%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bb_2%7D%20%3A%3D%20%5Cmathbf%7Bb_2%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%5Cmathbf%7Bb_2%7D%20J%28%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%2C%20%5Cmathbf%7BW_2%7D%2C%20%5Cmathbf%7Bb_2%7D%29%20%24%24)

### Softmax regression
Softmax regression is a generalization of logistic regression. It's used as the final layer in many neural networks. It receives as input a dataset ![equation](http://latex.codecogs.com/png.latex?%24%24X%24%24) with labels ![equation](http://latex.codecogs.com/png.latex?%24%24y%24%24), each label ![equation](http://latex.codecogs.com/png.latex?%24%24y%20%5Cin%20%7B0%2C1%2C2%2C%20%5Cdots%2C%20n%7D%24%24) belonging to one of a total of $n$ classes. It's purpose is to, given only the data, predict the class of each of its points.

We define the following:

| Symbol            | Description                                                 |
|:-----------------:|:------------------------------------------------------------|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20i%20%24%24)            | The dimension of the input vectors. |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20n%20%24%24)            | The number of classes.  |
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Ctheta%7D%20%24%24) | Parameters matrix of dimensions ![equation](http://latex.codecogs.com/png.latex?%24%24%20n%20%5Ctimes%20i%20%24%24).|
| ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7BG%7D%20%24%24) | Groundtruth matrix of dimensions ![equation](http://latex.codecogs.com/png.latex?%24%24%20n%20%5Ctimes%20t%20%24%24). Column ![equation](http://latex.codecogs.com/png.latex?%24%24%20i%20%24%24) contains a binary vector of dimension ![equation](http://latex.codecogs.com/png.latex?%24%24%20n%20%24%24) corresponding to the binary representation of label ![equation](http://latex.codecogs.com/png.latex?%24%24%20y%5E%7B%28i%29%7D%20%24%24) class.|

* Initialize ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Ctheta%7D%20%24%24) using a normal distribution.

To train the network, in each iteration we do:

* Compute the gradient:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BM%7D%20%3A%3D%20%5Cboldsymbol%7B%5Ctheta%7D%20%5Ccdot%20%5Cmathbf%7BX%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BM%7D%20%3A%3D%20%5Cmathbf%7BM%7D%20-%20%5Cmathbf%7B1_n%7D%20%5Ccdot%20%5B%5Coperatorname%7Bmax%7D%28%5Cmathbf%7BM%7D%29%5D%5ET%20%5Ctext%7B%20%20%20%20%28subtracts%20each%20column%20by%20its%20maximum%20element%29%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BP%7D%20%3A%3D%20%5Cexp%28%5Cmathbf%7BM%7D%29%20%5Coslash%20%5B%5Cmathbf%7B1_%7Bnn%7D%7D%20%5Ccdot%20%5Cexp%28%5Cmathbf%7BM%7D%29%5D%20%5Ctext%7B%20%20%20%20%28divide%20each%20column%20of%20%7D%5Cexp%28%5Cmathbf%7BM%7D%29%20%5Ctext%7B%20by%20its%20sum%29%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cboldsymbol%7B%5Ctheta%7D%7D%20J%28%5Cboldsymbol%7B%5Ctheta%7D%29%20%3A%3D%20%20-%5Cfrac%7B1%7D%7Bt%7D%20%28%5Cmathbf%7BG%7D%20-%20%5Cmathbf%7BP%7D%29%20%5Ccdot%20%5Cmathbf%7BX%7D%5ET%20%2B%20%5Clambda%20%5Ccdot%20%5Cboldsymbol%7B%5Ctheta%7D%24%24)

* Compute the cost:

![equation](http://latex.codecogs.com/png.latex?%24%24%20J%28%5Cboldsymbol%7B%5Ctheta%7D%29%20%3A%3D%20-%5Cfrac%7B1%7D%7Bt%7D%5Cmathbf%7B1_n%7D%5ET%20%5Ccdot%20%5B%5Cmathbf%7BG%7D%20%5Codot%20%5Clog%28%5Cmathbf%7BP%7D%29%5D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2%7D%5Cmathbf%7B1_n%7D%5ET%20%5Ccdot%20%28%5Cboldsymbol%7B%5Ctheta%7D%5Codot%20%5Cboldsymbol%7B%5Ctheta%7D%29%20%5Ccdot%20%5Cmathbf%7B1_i%7D%20%24%24)

* Update the parameters:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Ctheta%7D%20%3A%3D%20%5Cboldsymbol%7B%5Ctheta%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cboldsymbol%7B%5Ctheta%7D%7D%20J%28%5Cboldsymbol%7B%5Ctheta%7D%29%20%24%24)

To make predictions, we note that the matrix ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BP%7D%20%24%24) holds the conditional probabilities, so we just need to compute ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BP%7D_%7Bpred%7D%20%24%24) for the data ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BX%7D_%7Bpred%7D%20%24%24) we are predicting and take the class with maximum probability:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7By%7D_%7Bpred%7D%20%3A%3D%20%5Coperatorname%7Bmax%7D%28%5Cmathbf%7BP%7D_%7Bpred%7D%20%29%20%24%24)

### Stacked eutoencoders
In this architecture, we stack autoencoders, passing the hidden layer activation of one as the input to the next autoencoder, and so on, until a softmax layer, that outputs the prediction for the data passed as input to the first autoencoder. Each autoencoder is trained using the procedure above, the next one being trained after the previous one finished its training. After that first training is done, we then apply backpropagation to fine-tune the network as a whole.

Here we use the notation from both sparse autoencoders and softmax regression. We just have to be careful about the input from each layer and about which layer we are talking. We will use a superscript to label each matrix/vector with the corresponding sparse autoencoder layer. For example, ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_1%7D%5E%7B%28l%29%7D%20%24%24), means the matrix ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_1%7D%20%24%24) from sparse autoencoder layer ![equation](http://latex.codecogs.com/png.latex?%24%24%20l%20%5Cin%20%5C%7B0%2C%201%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%20%24%24), where ![equation](http://latex.codecogs.com/png.latex?%24%24%20n_l%20%24%24) is the number of autoencoders layers.

To pre-train the network:

* Train the first autoencoder layer with ![equation](http://latex.codecogs.com/png.latex?%24%24%20X%20%24%24) as input data.
* Train the ![equation](http://latex.codecogs.com/png.latex?%24%24%20lth%20%24%24) autoencoder layer with ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Ba_2%7D%5E%7B%28l-1%29%7D%20%24%24) as input data.
* Train the softmax layer with  ![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Ba_2%7D%5E%7B%28n_l%29%7D%20%24%24) as input data.

To fine-tune the network, in each iteration we do:

* Compute the gradients:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Ba_2%7D%5E%7B%280%29%7D%20%3A%3D%20X%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bz_2%7D%5E%7B%28l%2B1%29%7D%20%3A%3D%20%20%5Cmathbf%7BW_1%7D%5E%7B%28l%2B1%29%7D%20%5Ccdot%20%5Cmathbf%7Ba_2%7D%5E%7B%28l%29%7D%20%2B%20%5Cmathbf%7Bb_1%7D%5E%7B%28l%2B1%29%7D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%5ET%20%5Cquad%20l%20%5Cin%20%5C%7B0%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Ba_2%7D%5E%7B%28l%2B1%29%7D%20%3A%3D%20%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Bz_2%7D%5E%7B%28l%29%7D%29%20%5Cquad%20l%20%5Cin%20%5C%7B0%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BM%7D%20%3A%3D%20%5Cboldsymbol%7B%5Ctheta%7D%20%5Ccdot%20%5Cmathbf%7Ba_2%7D%5E%7B%28n_l%29%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BM%7D%20%3A%3D%20%5Cmathbf%7BM%7D%20-%20%5Cmathbf%7B1_n%7D%20%5Ccdot%20%5B%5Coperatorname%7Bmax%7D%28%5Cmathbf%7BM%7D%29%5D%5ET%20%5Ctext%7B%20%20%20%20%28subtracts%20each%20column%20by%20its%20maximum%20element%29%7D%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BP%7D%20%3A%3D%20%5Cexp%28%5Cmathbf%7BM%7D%29%20%5Coslash%20%5B%5Cmathbf%7B1_%7Bnn%7D%7D%20%5Ccdot%20%5Cexp%28%5Cmathbf%7BM%7D%29%5D%20%5Ctext%7B%20%20%20%20%28divide%20each%20column%20of%20%7D%5Cexp%28%5Cmathbf%7BM%7D%29%20%5Ctext%7B%20by%20its%20sum%29%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%20%5Cdelta%20%5E%7B%28n_l%29%7D%20%3A%3D%20-%20%28%7B%5Cboldsymbol%7B%5Ctheta%7D%7D%5ET%20%5Ccdot%20%28%5Cmathbf%7BG%7D%20-%20%5Cmathbf%7BP%7D%29%29%20%20%5Codot%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_2%7D%5E%7B%28n_l%29%7D%29%20%5Codot%20%28%5Cmathbf%7B1_%7Bit%7D%7D%20-%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_2%7D%5E%7B%28n_l%29%7D%29%29%20%20%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%20%5Cdelta%20%5E%7B%28l%29%7D%20%3A%3D%20%28%20%28%5Cmathbf%7BW_1%7D%5E%7B%28l%29%7D%29%5ET%20%5Ccdot%20%5Cdelta%20%5E%7B%28l%2B1%29%7D%20%29%20%20%5Codot%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_2%7D%5E%7B%28l%29%7D%29%20%5Codot%20%28%5Cmathbf%7B1_%7Bvt%7D%7D%20-%20%5Coperatorname%7BS%7D%28%5Cmathbf%7Ba_2%7D%5E%7B%28l%29%7D%29%29%20%20%5Cquad%20l%20%5Cin%20%5C%7Bn_l%20-%201%2C%20%5Cdots%2C%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cmathbf%7BW_1%7D%5E%7B%28l%29%7D%7D%20%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%5Cboldsymbol%20%5Cdelta%5E%7B%28l%2B1%29%7D%20%5Ccdot%20%28%5Cmathbf%7Ba_2%7D%5E%7B%28l%29%7D%29%5ET%20%5Cquad%20l%20%5Cin%20%5C%7B0%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cmathbf%7Bb_1%7D%5E%7B%28l%29%7D%7D%20%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%3A%3D%20%5Cfrac%7B1%7D%7Bt%7D%20%5Cboldsymbol%20%5Cdelta%5E%7B%28l%2B1%29%7D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%5Cquad%20l%20%5Cin%20%5C%7B0%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cboldsymbol%7B%5Ctheta%7D%7D%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%20%3A%3D%20%20-%5Cfrac%7B1%7D%7Bt%7D%20%28%5Cmathbf%7BG%7D%20-%20%5Cmathbf%7BP%7D%29%20%5Ccdot%20%20%28%5Cmathbf%7Ba_2%7D%5E%7B%28n_l%29%7D%29%5ET%20%20%2B%20%5Clambda%20%5Ccdot%20%5Cboldsymbol%7B%5Ctheta%7D%24%24)

* Compute the cost:

![equation](http://latex.codecogs.com/png.latex?%24%24%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%20%3A%3D%20-%5Cfrac%7B1%7D%7Bt%7D%5Cmathbf%7B1_n%7D%5ET%20%5Ccdot%20%5B%5Cmathbf%7BG%7D%20%5Codot%20%5Clog%28%5Cmathbf%7BP%7D%29%5D%20%5Ccdot%20%5Cmathbf%7B1_t%7D%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2%7D%5Cmathbf%7B1_n%7D%5ET%20%5Ccdot%20%28%5Cboldsymbol%7B%5Ctheta%7D%5Codot%20%5Cboldsymbol%7B%5Ctheta%7D%29%20%5Ccdot%20%5Cmathbf%7B1_i%7D%20%24%24)

* Update the parameters:

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7BW_1%7D%5E%7B%28l%29%7D%20%3A%3D%20%5Cmathbf%7BW_1%7D%5E%7B%28l%29%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cmathbf%7BW_1%7D%5E%7B%28l%29%7D%7D%20%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%20%5Cquad%20l%20%5Cin%20%5C%7B0%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cmathbf%7Bb_1%7D%5E%7B%28l%29%7D%20%3A%3D%20%5Cmathbf%7Bb_1%7D%5E%7B%28l%29%7D%20-%20%5Calpha%20%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cmathbf%7Bb_1%7D%5E%7B%28l%29%7D%7D%20%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%20%5Cquad%20l%20%5Cin%20%5C%7B0%2C%20%5Cdots%2C%20n_l%20-%201%5C%7D%24%24)

![equation](http://latex.codecogs.com/png.latex?%24%24%20%5Cboldsymbol%7B%5Ctheta%7D%20%3A%3D%20%5Cboldsymbol%7B%5Ctheta%7D%20-%20%5Calpha%20%5Cboldsymbol%7B%5Cnabla%7D_%7B%5Cboldsymbol%7B%5Ctheta%7D%7D%20J%28%5Cboldsymbol%7B%5Ctheta%7D%2C%5Cmathbf%7BW_1%7D%2C%20%5Cmathbf%7Bb_1%7D%29%20%24%24)

## The Code

To code the equations of the previous session using CUDA, we used the [CUBLAS] library extensively. For some more specific tasks, we implemented CUDA kernels for the job, but sure they can be optimized. All the CUDA kernels, CUBLAS wrappers and some constantes are in header file [helper.cuh](./Visual Studio/DNN/include/helper.cuh).

Besides the helper header, we have for now three other headers, each one implementing one of the above architectures. The following class diagram show the classes we have currently implemented and their relationship:

![class](./Visual Studio/DNN/ClassDiagram.png)

We also provide a file [mnist.cu](./Visual Studio/MNIST/mnist.cu), with an example application for digit recognition using the [MNIST] dataset. The data is read from text files stored in column-major order. The data are compressed in the file [Visual Studio/MNIST/data/data.7z](./Visual Studio/MNIST/data/data.7z) and need to be extracted before running the program.

## About this documentation

This markdown file [README.md] display equations as images rendered by [CodeCogs]. But the urls of the images are generated from the file [README] by a Python script which can be found in [allanino/markdown-latex]. So, when updating this document, we should always change only the README file and generate the README.md using that Python script.

[ULFDL Tutorial]: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial
[MNIST]:http://yann.lecun.com/exdb/mnist/
[CUDA]:http://docs.nvidia.com/cuda/
[CUBLAS]:http://docs.nvidia.com/cuda/cublas/
[CodeCogs]:http://www.codecogs.com/latex/eqneditor.php
[README]:./README
[README.md]:./README.md
[allanino/markdown-latex]:https://github.com/allanino/markdown-latex