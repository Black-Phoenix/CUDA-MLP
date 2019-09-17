CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Name: Vaibhav Arcot
  
  *  [LinkedIn](https://www.linkedin.com/in/vaibhav-arcot-129829167/)
  
* Tested on: Windows 10, i7-7700HQ @ 2.8GHz (3.8 Boost) 32GB, External GTX 1080Ti, 11G (My personal laptop)

### Overview
This code creates a fully connected neural network in **CUDA** which can identify the character from the image. For training this network, 52 images (1 for each letter case) were used along with random rotations (± 10°) were used. The results show that the network was able to identify the character with a 100% accuracy. 

![Sample Network](.\img\sample_network.PNG)

### Architecture

The code was created to allow the addition of any number of hidden layers (while keeping the last layer as a softmax layer). For the toy problem, the network takes in 225 inputs (15x15) as a column vector and outputs a 52x1 array with the probability of each class. The structure, the weight matrix dimensions for the hidden layers are 98x65, 65x50, 30x25, 25x40, while the input layers weight matrix has dimensions 225x98 and the output layer has dimensions 40x52. 

Between the layers, ReLu was the activation function used and softmax as the final layer and the loss function used was cross entropy loss.

### Dependencies
* OpenCV (to read images)
* CUDA 10 
* Cublas (matrix multiplication)
* Curand (random GPU initialization)
### Neural Network overview

Neural networks are multi-layer networks of neurons (the blue and magenta nodes in the chart below) that we use to classify things, make predictions, etc. Each neuron activates on features, and the cascading of said neurons allows the network to activate on more complex representations of the input data. In a neural network, the final layer does the job of a support vector machine, which draws a hyperplane to classify the data (if that is the task). 
![Neural Network](.\img\MLP.png)

#### Neuron
A neuron is the building block of a neural network. It takes in a value and returns a nonlinear transformation of the input. We define a layer as a stack of neurons. The nonlinearity is the crucial part because you can show that any linear combination of layers can be simplified down to just 1 layer.
![Forward pass](.\img\Weighting.png)

#### Activation functions

For the hidden layers, ReLu was the activation function of choice. This was because The function and its derivative both are monotonic. This is a nice property to have for the gradients. The only issue is the gradient blow up at zero (which can be solved using leaky ReLu)

For the final layer, I decided to use a softmax activation function because we are performing a multi class classification task. This way, we get a probability distribution over all possibilities. We can just take the max to get the prediction.

#### Forward propagation

To use the network for inference, the forward pass through the network is used. Input data is passed into the network and at each layer, the weight matrix (w) and the bias is added (b). The equations for the forward prop are shown below:

![](.\img\fp.png)

#### Back propagation

To actually train the network, we need to update the weights and biases. This is done using gradient decent on each of the parameters with respect to the final loss. To find these gradients, the chain rule is used. 

The gradients are shown below:

![Back Prop gradients](.\img\bp.png)

Once we have the gradients, we use the following equation to update the weights and biases
![Gradient decent equation](.\img\gradient_decent.png)

### Modifications

Besides getting the base neural network to work, listed below are some of the modifications I ended up doing in the quest for better performance

#### Stochastic Gradient Descent with momentum (SGD)

Regular gradient decent has a tendency of pulling the value rather quickly. This can result in loss curves being jagged. To combat this, SGD was implemented. The idea is that while updating the weights and biases. This changes the update equation to weight the last update along with the new update. This adds another hyper parameter β

![SGD](.\img\SGD.png)

#### Adaptive learning rate

During gradient decent, the learning rate has a huge impact on the training performance. If the value is too high, the loss oscillates around the optimal value (near the end). Too slow and it takes too long to converge. To combat this, an adaptive learning rate was used. The learning rate starts out at a value, and every X epochs (hyper parameter), the learning rate is halved. This allows the neural network to learn rapidly initially and slow down near the end.

#### Reduction on GPU

For the softmax layer, the equation for the layer is given by

![Softmax](.\img\softmax.png)

The denominator of this involves a sum over all the elements in the array. To do this, I used the upsweep phase of the work efficient scan I implemented as part of this assignment to allow for the sum to be calculated on the GPU (rather than actually copying it over to the CPU). The same function is also used to calculate the cross entropy loss (which also has a summation inside it).

#### Random rotations

Once the neural network was able to learn all the characters, to test the network, I started training the network using the same images but rotating them by a random angle (± 10°). The training data is using the rotated images while the testing was done using only the unrotated images. The results show that the network is kind of resilient to rotation (I did not push the limits).

#### Initializations

To initialize the weights, I decided to go with a modified version glorot initialization ([link](https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/)). Weights are drawn from a normal distribution, with 0 mean and  
$$
Var = \frac{2}{inputs}
$$

#### Vectorized entire code

Another optimization done to the code was that all equations for the gradient propagation were done using matrix math. This made it faster to train and infer. This also made it such that no math was done on the CPU for the forward and back propagation.

#### Hyper Parameters

Here are the list of hyper parameters that had the network working at 100% accuracy (Weights for these parameters were given)

| Parameter Name              | Value                            |
| --------------------------- | -------------------------------- |
| Learning Rate               | 0.0038                           |
| SGD β                       | 0.65                             |
| Adaptive learning rate      | 52*100 epochs half learning rate |
| epochs                      | 40000 (batch size 1)             |
| Random rotation limits      | ± 10°                            |
| Number of hidden layers     | 5                                |
| Dimensions of hidden layers | {98, 65, 50, 30, 25, 40}         |



### Results

#### Loss vs Epochs

![Loss vs Epoch](.\img\loss_vs_epoch.png)

In this plot, it is clear that the loss with momentum and decay performed the best. The kink in the middle (I believe) was due to suboptimal tuning of the decay rate. With a little more tuning the rate would decay slightly faster to allow for the drop but not the oscillation. The raw data is also uploaded. One important thing to mention was that with the pure learning rate approach, the max correct it got (for this run) was 51 out of 52 characters. This is not always the case (I have seen it getting a full but it requires more time). For the speed, the code can run 1 forward pass in **1.13577 ms** and 1 backward pass in  and **0.505299 ms**  average (with the architecture mentioned above).  

 Try diff architectures

### Rotation loss plots (given more iterations)

![Loss vs Epochs for random rotations](.\img\loss_vs_epoch_rand.PNG)

For the above plot, a random rotation of ± 10°  was given to the training data. The performance of this was 52 out of 52 cases (shown below), which is amazing considering it took the same number of iterations!

![Rot acc](.\img\rotation_matrix_acc.PNG)

### Observations

#### Neural network occasionally struggle with "L" and "h"

For some reason, with just gradient decent my network had a hard time distinguishing between L and h. This was kind of resolved by reducing the learning rate and giving it more time, but fully resolved when using SGD and adaptive learning rate fixes.

#### Cublas API

Cublas API isn't well documented. It took me a long long time to get matrix multiplication along with transpose working.

