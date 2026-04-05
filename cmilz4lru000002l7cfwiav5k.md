---
title: "The Art of Neural Networks: a mathematical deep dive"
seoTitle: "Exploring the Math Behind Neural Networks"
seoDescription: "Explore neural networks' mathematical foundations: linear algebra, calculus, perceptrons, and gradient descent insights"
datePublished: 2025-11-30T17:08:03.258Z
cuid: cmilz4lru000002l7cfwiav5k
slug: the-art-of-neural-networks-a-mathematical-deep-dive
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1764522365620/bfd63bd8-5054-4d64-8c5d-4b377f32e600.png
tags: python, machine-learning, neural-networks, mathematics, pytorch, backpropagation-neural-netowrk, gradient-descent

---

## The origins: Neural Networks and the human brain

Neural networks were inspired by the network of neurons and synapses in the biological brain. A biological neuron receives input from its dendrites, performs some operation on the inputs, and then fires an output based on certain conditions. Early artificial neurons (perceptrons) mimic this: they take inputs, weight them (multiply them by weights), sum the weighted inputs, apply a nonlinear activation, and produce an output (Note: This is just the feed-forward process of a perceptron). Over the years, this idea has evolved from single perceptrons to deep networks with many layers trained to make predictions on data.

The image below shows a biological neuron (left) and a perceptron (right).

![ (Source: towardsdatascience.com)](https://cdn.hashnode.com/res/hashnode/image/upload/v1764351626539/79eee863-e50c-4a31-8cee-8c0747a16e2f.webp align="center")

(Source: towardsdatascience.com)

## The 2 mathematical pillars of NN: Linear Algebra and Calculus

As you might have already guessed, neural networks are just a bunch of math operations (**seemingly** complex math). The math behind neural networks boils down to two main fields: linear algebra (more specifically, matrix multiplication) and calculus (more specifically, the chain rule). I would explain these mathematical subjects and their relevance in NN before going into actual neural networks. If you already understand these concepts, you can just skip to the serious stuff, although a refresher wouldn’t hurt.

### Matrix Multiplication

There is a very big chance that you already know how matrix multiplication works, but do you know how it works in neural networks? If you don’t, that will change very soon.

The main operations that happen in an artificial neuron have already been explained; here it is stated clearly:

1. Inputs enter into neuron through links
    
2. inputs are multiplied by the weights of their respective links
    
3. all weighted inputs are summed in the neuron, and a bias term is added
    
4. the sum of the weighted inputs is passed into an activation function
    
5. the result of the activation function on the sum of weighted inputs (and bias) is the output of the neuron
    

When there are a lot of inputs and neurons to keep track of, matrix multiplication comes to the rescue. Let’s take a simple example.

Suppose we have a single neuron receiving 3 inputs; each input would have its own weight. Below is a visual representation:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764487771446/c8791c4a-ea0a-4775-ac98-a922133f8957.jpeg align="center")

The pre-activation output of the neuron is:

$$z=wTx+b=w_1​x_1​+w_2​x_2​+w_3​x_3​+b.$$

We can represent the inputs and the weights as vectors instead of independent numbers. This would enable smoother computations. Given the input vector x=\[x1​,x2​,x3​\] and the weight vector w=\[w1​,w2​,w3​\]T and bias b, the formula for the pre-activation output can be written as:

For a neural network with two neurons:

Now let’s take an example; below is a visual representation of what we want to compute:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764496098616/bd3b8c06-afe6-47a4-8a8d-c44cbbfc70b9.jpeg align="center")

Here is the solution:

$$% Weight matrix, input vector, and bias vector W = \begin{bmatrix} 2 & -1 & 0 \\ 0 & 1 & 3 \end{bmatrix}, \qquad x = \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}, \qquad b = \begin{bmatrix} 0 \\ 1 \end{bmatrix}.$$

$$Pre-activation \,\,\,output, z = Wx + b, \qquad$$

$$Wx = \begin{bmatrix} 2 & -1 & 0 \\ 0 & 1 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 2\cdot 1 + (-1)\cdot 2 + 0\cdot(-1) \\ 0\cdot 1 + 1\cdot 2 + 3\cdot(-1) \end{bmatrix} = \begin{bmatrix} 2 - 2 + 0 \\ 0 + 2 - 3 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}.$$

$$Add \,\,\,bias\,\,\, z = Wx + b = \begin{bmatrix} 0 \\ -1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}.$$

### The Chain Rule

Time for a quick calculus recap. Calculus is basically the mathematics of change; it is a field of mathematics that is used to model the rate of change of one thing in respect to another thing (It can literally be anything; okay, maybe not anything). So now what is the chain rule?

Imagine we have two sets of data; one is a table of age and height, and another is a table of height and weight. Let age be a, height h, and weight w. The change in height with respect to age (i.e., how height changes when there is a change in age) can be calculated easily using the normal differentiation formula; the same goes for the change in weight with respect to height. But if we wanted to know the change in weight with respect to age, then there is a problem since there is no available relationship between them. This would require the chain rule; simply put, the change in weight with respect to age is equal to the change in height with respect to age, multiplied by the change in weight with respect to height. We can see it in action below:

$$\frac{dw}{d a} = \frac{d h}{d a} \cdot \frac{dw}{dh}$$

If you look closely, you will notice that a numerator and denominator can cancel out, leading to our desired derivative (change of something with respect to something else).

Now that we have the fundamentals dealt with, let’s move on to the actual neural networks.

## The perceptron: One neuron

A perceptron is the simplest neural network architecture; it has only one neuron.

given input vector X, weight vector W, bias b, and activation function f(). The output of a perceptron is given as:

$$f(z)= \sigma(z) = \frac{1}{1 + e^{-(Wx+b)}}$$

f is an activation function, and there are quite a few of them. For the purpose of this article, I would stick with the sigmoid activation function because it is the most commonly used one for classification. The output graph of a sigmoid activation function is shown below:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764496472750/7b493f69-ca39-43d7-9068-dba4a4f2cf0f.png align="center")

Now let’s take an example of a perceptron used for binary classification: (if you need to read on machine learning fundamentals, check [here](https://ysolomon.hashnode.dev/ai-foundations-implementing-machine-learning-models-from-scratch).)

*Note: in the case of binary classification, when the neuron outputs a value (this value is going to be between 0 and 1 because of the sigmoid function), if the output is greater than 0.5, then it would be predicted as class 1; else, it would be predicted as class 0.*

Now let’s take a perceptron with 3 inputs and use Python for it.

So far we have been able to cover what is known as **forward propagation** in a perceptron, which is simply how we get output from the neural network. I am sure that you know that neural networks are supposed to **learn** from data. How does this happen? We would cover that next.

### Loss functions

Before it can “learn,” we would need to be able to quantify (or rate) its current performance, and then we can improve on that. This is where **loss functions** come in; it is basically a function that is used to quantify the performance of a neural network (machine learning models in general, actually). Just like activation functions, there are quite a few loss functions too. A loss function basically measures how well the network predicts labels on data. For binary classification with sigmoid outputs, the standard choice is **binary cross-entropy**.

$$L = -\left( y \log(\hat{y}) + (1-y)\log(1 - \hat{y}) \right)$$

The loss is the output of the loss function; it is a scalar value (a single value) that we aim to minimize. Minimization of the loss would lead to a maximization of the neural network prediction performance.

### How the perceptron learns weights: gradient and gradient descent

What is a gradient?

A gradient is basically a vector of partial derivatives. It describes the change in loss with respect to model parameters. By computing the gradient of a perceptron, we would know how much the loss changes (i.e., how much better the perceptron gets) with change in the perceptron parameters (weights and bias).

When we know how the loss changes in respect to the perceptron parameters, we can then be able to reasonably decrease the loss by tweaking the model parameters.

Learning via gradient descent

Let the loss be L(w); we want to minimize L(w). Gradient descent updates the weights like so:

$$w_{new}=w_{old}−η\frac{dL(w)}{dW}$$

where η&gt;0 is the **learning rate** controlling how much the weights change at each step. We do this thing multiple times until we achieve our desired loss, or we reach the maximum number of steps, or the loss doesn’t change much anymore.

*Note: Gradient descent is not restricted to neural networks; it is just a method of optimization in machine learning that is mainly used for neural networks. It can be used for logistic regression and linear regression, to name a few.*

Let’s clearly define the steps for learning weights in a perceptron with gradient descent. Then we would take an example.

1. **Compute the gradient:** calculate the derivative of loss w.r.t. weights.
    
2. **Initialize** weights to random values.
    
3. **Compute slope**: evaluate the gradient at current weights.
    
4. **Compute step**: step = learning\_rate × gradient. (We subtract this from weights; note we move opposite the gradient to **reduce** the loss.)
    
    * **Why use a learning rate?** It scales updates. Too large, then we overshoot minima or diverge. Too small leads to slow convergence (i.e., it would take more steps to reduce the loss significantly).
        
    * **Global minima vs local minima**: For simple convex problems (like logistic regression), gradient descent finds the global minimum. For deep neural nets nonconvexity can produce many local minima or low points; good initialization and learning rates help convergence.
        

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla_w L$$

The above equation shows the last step, and then we continue from step 3 until:

* We exceed maximum number of steps
    
* The loss does not change much
    
* We have attained a preferred performance
    

Now let us take an example with a simple perceptron with a single input, bias, and 1 output, using the sigmoid activation function for classification.

The formulas for training a perceptron given the parameters:

Single input x, weight w, and bias b. (Note: y is the actual value and y\_hat is the predicted value from the model.)

1. Forward Propagation:
    

$$Weighted\,sum:z = wx + b$$

$$Sigmoid \,\,\,activation: \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

$$2.\,\,\,Loss\,\,Function (Cross-Entropy)$$

$$Binary\,\,\,cross-entropy\,\,\,loss: L = -\left( y \log(\hat{y}) + (1-y)\log(1 - \hat{y}) \right)$$

$$3.\,\,\,Backpropagation$$

$$We\,\,want\,\,the\,\,gradients: \frac{\partial L}{\partial w}, \quad \frac{\partial L}{\partial b}$$

$$Step\,\,1: Derivative\,\,of\,\,loss \,\,w.r.t.\,\,prediction \frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

$$Step\,\,2: Derivative \,\,of\,\, prediction \,\,w.r.t. \,\,weighted\,\,sum \frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

$$Step\,\,3: Derivative \,\,of \,\,weighted \,\,sum \,\,w.r.t.\,\, parameters \frac{\partial z}{\partial w} = x , \frac{\partial z}{\partial b} = 1$$

$$4.\,\,Gradient\,\,of \,\,the \,\,Weights$$

$$By\,\,the \,\,chain\,\,rule: \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$Substituting: \frac{\partial L}{\partial w} = \left(\frac{-y+\hat{y}}{\hat{y}(1 - \hat{y})}\right) \cdot \left(\hat{y}(1 - \hat{y})\right) \cdot x$$

$$Simplifying: \frac{\partial L}{\partial w} = (-y + \hat{y})x$$

$$Using\,\,the\,\, more\,\, general\,\, form: \boxed{ \frac{\partial L}{\partial w} = (\hat{y} - y)x }$$

$$5. Gradient\,\,\, of\,\,\, the\,\,\, Bias$$

$$Similarly: \frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial b}$$

$$Substituting: \frac{\partial L}{\partial b} = \left(\frac{-y+\hat{y}}{\hat{y}(1 - \hat{y})}\right) \cdot \left(\hat{y}(1 - \hat{y})\right) \cdot 1$$

$$Simplifying: \boxed{ \frac{\partial L}{\partial b} = \hat{y} - y }$$

$$6. Gradient\,\, Descent\,\, Updates: Let \,\,the\,\, learning\,\, rate\,\, be\,\,\eta.$$

$$Weight\,\, update: w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}$$

$$Bias\,\,update: b_{\text{new}} = b_{\text{old}} - \eta \frac{\partial L}{\partial b}$$

$$Substituting\,\,the\,\,gradients: w_{\text{new}} = w - \eta (\hat{y}-y)x \qquad b_{\text{new}} = b - \eta (\hat{y}-y)$$

$$7. Summary\,\,of \,\,Key\,\, Equations$$

$$Forward\,\,pass: z = wx + b,\quad \hat{y} = \sigma(z),\quad L = -\left( y\log(\hat{y}) + (1-y)\log(1 - \hat{y}) \right)$$

$$Gradients: \frac{\partial L}{\partial w} = (\hat{y}-y)x, \quad \frac{\partial L}{\partial b} = \hat{y}-y$$

$$Updates: w_{\text{new}} = w - \eta (\hat{y}-y)x, \quad b_{\text{new}} = b - \eta (\hat{y}-y)$$

*Note: The process of calculating the gradients and updating weights is what is known as backpropagation.*

*Note: To derive the change in loss with respect to the weights, we calculate the derivative of all the relationships that were available:*

*1\. The change in loss with respect to the (post-activation) output.*

*2\. The change in post-activation output with respect to the pre-activation output.*

*3\. Finally, the change in the pre-activation output with respect to the weights (and bias).*

Then we multiply all to find the derivative of the first (loss) with respect to the last (weights and bias) values on the **chain (wink** wink).

Here is an example where we use the derived formulas for a full training step (forward propagation and back propagation):

$$x = 2, \quad y = 1, \quad w = 0.5, \quad b = -1, \quad \eta = 0.1$$

Forward Pass:

$$Weighted \,\,sum: z = wx + b = (0.5)(2) - 1 = 0 \quad Sigmoid\,\, activation: \hat{y} = \sigma(z) = \frac{1}{1 + e^{-0}} = 0.5$$

$$Cross-entropy \,\,loss: L = -\log(\hat{y}) = -\log(0.5) = 0.6931$$

$$Gradient\,\, of\,\, loss\,\, w.r.t.\\,\, weight: \frac{\partial L}{\partial w} = (\hat{y} - y)x = (0.5 - 1)(2) = -1$$

$$Gradient \,\,of \,\,loss \,\,w.r.t.\\,\, bias: \frac{\partial L}{\partial b} = \hat{y} - y = 0.5 - 1 = -0.5$$

Backward Pass:

$$New \,\,weight: w_{\text{new}} = w - \eta\frac{\partial L}{\partial w} = 0.5 - 0.1(-1) = 0.6 \quad New \,\,bias: b_{\text{new}} = b - \eta\frac{\partial L}{\partial b} = -1 - 0.1(-0.5) = -0.95$$

Below is a Python implementation of the step for training a perceptron.

```python
import numpy as np

# data
X = np.array([-1.0,  1.0])   
Y = np.array([ 0.0,  1.0])   

# initial random parameters
w = 0.5
b = 0.0
eta = 0.1

# Sigmod activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# forward
z = w * X + b            
y_hat = sigmoid(z)       

# losses (per example)
eps = 1e-12 # To avoid log(0)
losses = -(Y * np.log(y_hat + eps) + (1 - Y) * np.log(1 - y_hat + eps))
loss = losses.mean()
print("z:", z)
print("y_hat:", y_hat)
print("loss per example:", losses)
print("mean loss:", loss)

# backprop (vectorized)
dz = y_hat - Y               # shape (2,)
dw = np.mean(dz * X)        # scalar
db = np.mean(dz)            # scalar

print("dz:", dz)
print("dw:", dw)
print("db:", db)

# update
w_new = w - eta * dw
b_new = b - eta * db
print("w_old, b_old:", w, b)
print("w_new, b_new:", w_new, b_new)
```

When we perform forward propagation with the updated parameters and compute the loss, it would have a lower value than the initial values. This is proof of learning. This simple concept is what powers more complex neural networks (although some more concepts are added on top for adaptation to other problems and for better performance)

Now for a more general python implementation of a perceptron

```python
import numpy as np

class Perceptron():
    def __init__(self, n_input, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []

        # Single neuron:
        # weights shape: (n_input, 1)
        # bias shape: (1,)
        self.w = np.random.randn(n_input, 1) * 0.01 # Initialize random weights
        self.b = np.zeros((1,))

    # Sigmoid activation
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Derivative of sigmoid
    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        for i in range(self.epochs):

            # Forward pass
            z = np.dot(X, self.w) + self.b         # (m,1)
            a = self.sigmoid(z)                     # prediction

            # Cross-entropy loss
            loss = -np.mean(y * np.log(a + 1e-15) +
                            (1 - y) * np.log(1 - a + 1e-15))
            self.losses.append(loss)

            # Backpropagation
            dz = a - y                               # (m,1)
            dw = (1/m) * np.dot(X.T, dz)             # (n_input,1)
            db = (1/m) * np.sum(dz)                  # scalar

            # Gradient descent update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
```

For a single layer neural network, we would have to update weights and bias for each neuron in the network. Let’s check it out

## Single Layer Neural Network

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764478404353/7041fca7-a321-4234-bab2-564028dd845a.jpeg align="center")

Let’s skip the forward pass and jump right into the back propagation. Here, we would need to update the weights and bias of all the neurons. (the same process in perceptron, but three times)

Let the target be y = 1.

*Note: I made a mistake, it should be w3 = 0.5 sorry.*

Calculating the Loss would give us L = -1.11. Also, let the learning rate be 1 (this is never done by the way).

for W1:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$\frac{\partial L}{\partial w} = (\hat{y} - y)x$$

$$\frac{\partial L}{\partial b} = (\hat{y} - y)$$

$$\frac{\partial L}{\partial w1} = (0.9-1)1$$

$$\frac{\partial L}{\partial b} = (0.9-1)$$

$$w1_{\text{new}} = w - \eta\frac{\partial L}{\partial w1} = 0.2 - 1(-0.1) = 0.3$$

$$\frac{\partial L}{\partial w2} = (0.9-1)2$$

$$w3_{\text{new}} = w - \eta\frac{\partial L}{\partial w} = 0.3 - 2(-0.1) = 0.5$$

$$\frac{\partial L}{\partial w3} = (0.9-1)3$$

$$w3_{\text{new}} = w - \eta\frac{\partial L}{\partial w} = 0.5 - 3(-0.1) = 0.8 \quad b_{\text{new}} = b - \eta\frac{\partial L}{\partial b} = 0 - (-0.1) = 0.1$$

When we go through the forward propagation with the new weights, we would see that our loss has reduced, the new loss L = 0.07, this is proof of learning because the loss is reducing we can conclude that the neural network is getting better.

Now let’s look at a python implementation of a more general use of a single layer neural network.

```python
## Implementing neural network from scratch
# This would be a neural network for binary classification
# There would be the input layer, hidden layer and then the output layer (one neuron)

import numpy as np
import matplotlib.pyplot as plt

class Neural_Network():
  def __init__(self, n_input, n_hidden, learning_rate, epochs):
    self.lr = learning_rate
    self.epochs = epochs
    self.losses = []

    # Initialize the weights and bias for the hidden layer and the output layer
    self.w1 = np.random.rand(n_hidden, n_input) * 0.01 # Shape : hidden_neurons, input_neurons
    self.b1 = np.zeros((1, n_hidden)) * 0.01 # Shape : (1, hidden_neurons) one for each neuron
    self.w2 = np.random.rand(n_hidden, 1) * 0.01 # Shape : 1, hidden (for binary classification, only one output neuron is required)
    self.b2 = np.zeros((1, 1)) * 0.01 # basically just a scalar

  def sigmoid(self, z):
      return 1/ (1 + np.exp(-z)) # sigmoid activation function

  def sigmoid_derivative(self, z):
    return z * (1 - z) # derivartive of activation function for the backward pass

  def fit(self, X, y): # Training Loop
    m = X.shape[0] # getting the number of features of X
    y = y.reshape(-1,1) # Reshape into a column vector

    for i in range(self.epochs):
      # Forward Pass
      output1 = np.dot(X, self.w1.T) + self.b1
      activation_output1 = self.sigmoid(output1)
      output2 = np.dot(activation_output1, self.w2) + self.b2
      activation_output2 = self.sigmoid(output2) # prediction

      # Calculate the loss (binary cross entropy loss)
      # 1e-15 to prevent log zero
      self.losses.append(-np.mean(y * np.log(activation_output2 + 1e-15) + (1 - y) * np.log(1 - activation_output2 + 1e-15)))

      # Backward propagation
      # output layer gradients
      error = activation_output2 - y
      dw2 = (1/m) * np.dot(activation_output1.T, error)
      db2 = (1/m) * np.sum(error, axis=0, keepdims=True)

      # hidden layer gradients
      delta = (np.dot(error, self.w2.T)) * self.sigmoid_derivative(activation_output1)
      dw1 = (1/m) * np.dot(X.T, delta)
      db1 = (1/m) * np.sum(delta, axis=0, keepdims=True)

      # Update the weights and biases
      self.w1 -= self.lr*dw1.T
      self.b1 -= self.lr*db1
      self.w2 -= self.lr*dw2
      self.b2 -= self.lr*db2

      if i % 100 == 0:
        print(f"Epoch {i}, Loss: {self.losses[i]:.4f}")

  def predict_proba(self, X):
    # Forward pass
    output1 = np.dot(X, self.w1.T) + self.b1
    activation_output1 = self.sigmoid(output1)
    output2 = np.dot(activation_output1, self.w2) + self.b2
    activation_output2 = self.sigmoid(output2)
    return activation_output2.T

  def predict(self, X):
    # Using the output from the forward pass for prediction
    output = self.predict_proba(X)
    return (output >= 0.5).astype(int) # threshold at 0.5
```

We would now move on to the final section of this article: Multi-layer neural networks.

## Multi-layer Neural network

Let us tweak the first neural network a little:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764482779457/314c5e90-7272-4705-9a0c-f4e871feb20c.jpeg align="center")

Now we have two layers, the layer before the output neuron and the one before that containing only one neuron (for simplicity).

To get the change in the loss with respect to w1-1, we would have to calculate all the derivatives from the output neron point up to that weight. Like so:

$$By\,\,the \,\,chain\,\,rule: \frac{\partial L}{\partial w1-1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y2}}{\partial z2} \cdot \frac{\partial z2}{\partial \hat{y} 1-1} \cdot \frac{\partial \hat{y}1-1}{\partial z1-1} \cdot \frac{\partial z1-1}{\partial w1-1}$$

Lets take it step by step

1. This is the change in Loss with respect to the post activation function at the output neuron. First Term
    
2. THis is the change in the post activtion output at the output neuron with respect to the pre activation output at the output neuron (weighted sum + bias). Second Term
    
3. this is the change in the input to the second layer with respect to the post activation output from the first layer from the first neuron (the neuron shown in the diagram above). Third Term
    
4. this is the change in the post activation output from the first layer from the first neuron with respect to the pre-activation output from that neuron. Fourth Term
    
5. Then finally this is the change in the pre activation output with respect to the weight we want to update. Fifth Term
    

For simplicity during calculation, we use a term delta to simplify the derivatives required for each layer. For the second Layer, delta is given by:

$$\delta = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

For the first layer, this is what we use to update the weights:

$$\frac{\partial L}{\partial w_{1-1}} = \underbrace{\delta^{(2)}}_{\frac{\partial L}{\partial z^{(2)}}} \cdot \underbrace{w^{(2)}1}{\frac{\partial z^{(2)}}{\partial a^{(1)}_1}} \cdot \underbrace{\sigma'(z^{(1)}1)}{\frac{\partial a^{(1)}_1}{\partial z^{(1)}1}} \cdot \underbrace{x}{\frac{\partial z^{(1)}1}{\partial w{1-1}}}.$$

General formula for delta is given by:

$$% Hidden layer delta formula \delta^{[l]}i = f'\!\left(z^{[l]}i\right) \sum{k} w^{[l+1]}{k i} \, \delta^{[l+1]}_k$$

Where:

* f’(z) is the derivative of the post activation output of the neuron for the weight you want to update
    
* W(ki) is the weight that connects the neurons
    
* delta(k) is the delta of the next layer of the neurons connected to the neuron, whose weight you want to update
    
* There might be multiple neurons connected to that neuron, so the last two values are multiplied and summed for each of those neurons
    

There are more things to consider, but that would not be covered here, I would move on to implementation in python with pytorch.

For a deeper look into deltas, check here : [A Look into NNs Training Dynamics! | by Adam Elimadi | AI Advances](https://ai.gopubby.com/if-youre-a-ml-beginner-learn-this-first-e2d64cbcbafb#7cfa)

python implementation of a multilayer neural network using pytorch. With the training step also implemented:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# MULTILAYER PERCEPTRON FOR BINARY CLASSIFICATION

# Network architecture:
# Input  -> Hidden_Layer1 -> Hidden_Layer2 -> Output(sigmoid)

class MultiLayerNN(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2):
        super().__init__()
        # Linear layers automatically create weight matrices:
        #   W^(l) of shape (neurons_in_layer, neurons_previous)
        # PyTorch initializes weights randomly.

        self.layer1 = nn.Linear(n_input, n_hidden1)   # First hidden layer
        self.layer2 = nn.Linear(n_hidden1, n_hidden2) # Second hidden layer
        self.output = nn.Linear(n_hidden2, 1)         # One output neuron

        # Activation function (sigmoid everywhere as required)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass:
        #   z1 = xW1 + b1
        #   a1 = sigmoid(z1)
        #   z2 = a1W2 + b2
        #   a2 = sigmoid(z2)
        #   z3 = a2W3 + b3
        #   yhat = sigmoid(z3)
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.output(x))  # Final prediction (probability)
        return x



# DATASET (dummy example)
#     X: shape (m, n_features)
#     y: shape (m, 1)

X = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)



# CREATE MODEL

model = MultiLayerNN(
    n_input=1,     # 1 feature input
    n_hidden1=4,   # First hidden layer neurons
    n_hidden2=3    # Second hidden layer neurons
)



# LOSS FUNCTION AND OPTIMIZER

criterion = nn.BCELoss()  # Binary Cross Entropy - this is basically cross entropy, but for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Gradient Descent Optimizer


# TRAINING LOOP

epochs = 200

for epoch in range(epochs):

    # 1. Forward pass (done by model.forward)
    y_pred = model(X)

    # 2. Compute loss
    loss = criterion(y_pred, y)

    # 3. Backward pass:
    # PyTorch automatically:
    #   - Computes gradients using chain rule
    #   - Propagates deltas backward through layers
    optimizer.zero_grad()
    loss.backward()  # This is where backprop happens

    # 4. Update weights using the computed gradients
    optimizer.step()

    # Print occasionally
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



# PREDICTION/INFERENCE

with torch.no_grad():  # No gradients needed for prediction
# We basically switch off gradient computation (pytorch would comput gradients after a forward pass
# unless explicitly told not to)
    preds = model(X) # Basically a forward pass with the input data.
    preds_binary = (preds >= 0.5).int()

print("\nPredicted probabilities:\n", preds)
print("\nBinary predictions:\n", preds_binary)
```

And that brings us to the end of the deep dive, hope you learnt someting. I will be covering CNNs next

## Bonus Terms and Concepts

1. GPU; This is a Graphical Processing unit, you might have heard a lot about it in the ML/AI space. They were originally made to render pixels for graphics. They are so popular in the ML/AI space because they thrive in Matrix based computations (which if you recall is one of the connerstones of Neural networks) hence they train Neural networks faster than traditional CPUs.
    
2. TPU: This is a Tensor Processing unit. This Processor is specifically made for matrix computations, they have the circuitry required for performing matrix operations fast. They are faster than GPUs but they are basically only good for one thing.
    
3. Tensor: In simple terms, it is a generalization of matrices, vectors and scalars. A scalar is a 0-D tensor, a vector is a 1-D tensor, and a matrix is a 2-D tensor, while more complex data, like images are often represented as 3-D tensors (height × width × color channels). In deep learning, even higher-dimensional tensors exist, such as 4-D tensors used to store batches of images (batch size × height × width × channels)
    
4. Batch Size: In deep learning this is how much data is stacked together, it could be scalars, vectors, matrices or even images. After the model performs forward propagation with a batch, step sizes are aggregated then the weights are updated. This actual helps with speed of trainng as well as reducing noise in the trainng data (the aggregation allows the model to reach more general parameters that work well on the whole data set i.e. both seen and unseen data). Batch size while training a model is the number of data points that go through the model before the weights are updated.
    
5. Epochs: This is the number of times the whole training data set goes through the model.
    
6. Training iterations: This is the number of times that weights are updated during training.
    
7. Dropout: This is a method of preventing overfitting by disabling neurons during training.
    

References

1. [A Look into NNs Training Dynamics! | by Adam Elimadi | AI Advances](https://ai.gopubby.com/if-youre-a-ml-beginner-learn-this-first-e2d64cbcbafb#7cfa)
    
2. Statquest with Josh Starmer : [here](https://youtu.be/CqOfi41LfDw?si=z7puswq0l3GBPFit).
    
3. **Brokttv. (2025).** *Training Models from Scratch Using NumPy and Linear Algebra on a Piece of Paper*. GitHub. Retrieved from [https://github.com/Brokttv/training\_models\_from\_scratch/tree/main](https://github.com/Brokttv/training_models_from_scratch/tree/main?utm_source=chatgpt.com)
    
4. deep learning with pytorch by Udacity [here](https://www.udacity.com/course/deep-learning-pytorch--ud188).