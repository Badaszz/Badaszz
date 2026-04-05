---
title: "AI foundations - Implementing machine learning models from scratch"
seoTitle: "Building ML Models from Scratch"
seoDescription: "Implement machine learning models from scratch to explore algorithms like linear regression, ridge, KNN, KMeans, enhancing ML intuition and application."
datePublished: 2025-10-03T11:31:19.020Z
cuid: cmgark5f0000c02jp236j6d3x
slug: ai-foundations-implementing-machine-learning-models-from-scratch
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1759490800923/f043c681-5cb5-412d-8977-3f568dbf75be.png
tags: artificial-intelligence, machine-learning, mathematics, unsupervised-learning, linearregression, machine-learning-algorithm, machine-learning-models, math-for-machine-learning

---

It is relatively easy to use a machine learning model, all the math has been abstracted, and you can just call a function and add some parameters to train a model. You can achieve good results from using this plug and play method, you can apply multiple combinations of hyperparameters to arrive at your desired accuracy. But Machine learning is more than just calling functions, it is complex, and to use it effectively as AI/ML engineers, we must understand how these algorithms work behind the scenes.

Here, I document my journey to understand some machine learning models and learn how they do what they do (how they learn). Understanding how these models work will provide the intuition needed to fully harness AI's potential. I'll begin with simple linear supervised learning models for regression, then proceed to nonlinear supervised learning before exploring unsupervised learning.

Important concepts to understand:

Artificial Intelligence: A field of study concerned with enabling machines to think and solve problems like human beings. It is aimed at recreating the intelligence of human beings in computers.

Machine learning: A branch of AI that achieves the goal of AI by enabling the computers and machines to learn. it achieves AI by enabling machines to make decisions based on past data. Data is a very important part of machine learning. Machines learn from data.

Supervised learning: It is a machine learning approach that uses labeled data to train the model. Training is the process through which the machine learns from data. In supervised learning, the model is restricted to being correct based on what we say is correct. An illustration would be showing a baby a cat and telling him to call it a cat, showing him a dog and telling him to call it a dog; the baby is trained based on predetermined names of these animals. These names are called labels and this particular problem is classification.

Unsupervised learning: another machine learning approach, but this one uses unlabeled data. The model is not restricted to being correct by what we think is correct; the model is free to make conclusions of its own. Imagine showing a baby two cats and two dogs (let us assume this baby understands English) and then telling the baby to group the animals into two groups based on their similarity. No names are given to the baby. Eventually the baby should be able to recognize that the cats belong to one group, and the dogs belong to another (if we had more cats and dogs, the baby could even go as far as to be able to group different cat and dog breeds). This particular problem is called clustering.

Machine learning models: These are basically the algorithms (i.e., steps taken) to learn from data. There is more than one possible machine learning problem; likewise, there are numerous ways to solve machine learning problems and learn from data.

Now let’s dive into the actual model implementations. We’ll start simple with linear regression, because it gives us the foundation for more complex models later.

## Linear supervised learning models

They are called linear because the ultimate goal of these models is to learn a function for a line. These models could be extended to learn non-linear input-output relationships through feature transformation of the input features (e.g. polynomial, trigonometric functions etc.). They generally learn weights(model parameters) that have a linear relationship with input features.

*The model essentially learns numerical weights (parameters) that are multiplied with the inputs (which may be raw or transformed). These weight-input products are then summed, to produce the output. This is why the model is considered linear in its parameters, even if the inputs themselves have been transformed non-linearly.*

I will be covering linear regression and ridge regression. Regression is basically a machine learning problem where we try to train a model on data to make continuous numerical outputs as conclusions. like predicting stock prices, age, weight, height, etc. Where X is input and Y is output, a linear regressor tries to solve this problem by defining a line (y = mX + c, where y is an approximation of the original Y) that maps X to Y in a way that minimizes the difference between the correct Y and the predicted y.

This is a sample line learnt by a linear regressor:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758967681536/1fd5db4e-9f78-491f-893d-0de6c54e22f8.png align="center")

### Linear Regression

Now, let me run through the whole problem and the solution to the problem.

The problem:

* We have data X and Y, where X is the part of the data that would always be known, and Y is the part of the data that we don’t know. X could be the height of a person, and Y could be their age.
    
* So essentially, given the height of an individual, we want to estimate their age. This is a standard regression problem. The height is called a feature, and the age is called the target.
    
* X and Y are arrays of numbers, i.e., X = \[100, 130, 150, 140\] and Y = \[15, 21, 25, 23\]. This is the data we are going to use for our problem. (the underlying relationship is Y = 0.2X - 5)
    

The objective:

* For linear regression, the objective is to make a line (y = mX + c), where m is the slope and c is the intercept; therefore, the objective is to find m and c.
    
* This line should map X to Y in such a way that the error is minimized.
    
* There are various ways to calculate the error between the predicted value and the actual value; one of the ways to do this is to use the Mean Squared Error (MSE). Just as the name implies, we basically calculate all the differences between the predicted and actual values, square them, and then find their mean.
    
* The objective has now evolved to finding ‘m’ and ‘c’ such that the MSE for y = mX + c is minimized.
    

The solution:

* Luckily for us, in mathematics we can find the value of a variable (m and c, in this case) that would achieve the minimum of a function (MSE in this case) by finding the gradient of that function with respect to the variable (by finding how MSE changes with change in m and c) and then equating it to zero.
    
* What I am saying here is that we can achieve our objective (finding m and c such that MSE is minimized) by solving for change in MSE with change in m and C equal to zero. This is the math side of it, and while it is important, it is not a must to be able to solve it or prove it. The most important part of the algorithm is the intuition behind it.
    

Step 1 : The desired model

$$\hat{y} = mx + c$$

Step 2: Mean Squared Error to be minimized

$$MSE(m, c) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - (mx_i + c)\right)^2$$

Step 3: Find the change in (partial derivatives)MSE with respect to m and c, then equate it to zero

$$for\ m:\ \frac{\partial MCE}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i \left(y_i - (mx_i + c)\right) = 0$$

$$for\ c:\ \frac{\partial J}{\partial c} = -\frac{2}{n} \sum_{i=1}^{n} \left(y_i - (mx_i + c)\right) = 0$$

The Solution becomes:

$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \quad c = \bar{y} - m\bar{x}$$

As seen in the equation above, the solution to our problem is merely a formula.

We can put this formula to use with this Python code:

```python
# Data
X = [100, 130, 150, 140]  # This is our height, which each value representing the height of an individual
Y = [15, 21, 25, 23] # These are our ages, Y = 0.2X - 5

# Learning
# Now lets calculate our slope and intercept with the formula from earlier
x_mean = np.mean(X)
y_mean = np.mean(Y)

m = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2) # this gives us 0.2
c = y_mean - m * x_mean # This gives us -5

# Predict
x_new = 200
y_new = (x_new*m) + c
```

You might be thinking, “What about X and Y relationships that are not linear?” In order for the regression model to learn a non-linear input-output relationship, we would need to perform *feature expansion* on our input data, i.e., we would have to extract new data from our input data, that would make the input-output relationship non-linear. This new data would just be a transformation of the input data; we could raise it to a power, apply a trigonometric function, or take its logarithm. All this is considered feature expansion.

Let’s use another set of data points, and the underlying relationship in this data would be y = sin(x) + 3. Using the formula we derived earlier, we would not get a very good line for the data.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758986450180/4d01045d-423e-42f4-a1a5-18210da97b88.png align="center")

If we were to perform feature expansion on X, so instead of using X we use sin(x) in our equation to find m and c (we replace x with sin(x)), this is the line we would get:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1758986491019/183949cd-9e25-4c50-9817-1f629899a02b.png align="center")

Congratulations! Using the derived formula, we have been able to successfully implement linear regression from scratch.

Note: This is a simple implementation using only one feature. In real-world cases we would have multiple features, and instead of X being a 1-D array, X would be a matrix, with each column corresponding to a feature. For a regression problem with more than one feature, the previously derived formula would not hold. The general formula to find the slope and intercept for regression is given below

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

So here we are working with matrices. Our solution is no longer a single number; it is now a 1D array and it accounts for both the slope, m and the intercept, C.

here is a python implementation; the underlying relationship is y = 3X1 + 5X2 + 6X3 + 9 = 0 (Note that the coefficient of the intercept, 9, is X^0)

```python
import numpy as np

# Step 1: Define features
X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])        # Feature 1
X2 = np.array([5, 3, 6, 2, 7, 1, 8, 4, 9, 10])        # Feature 2
X3 = np.array([2, 7, 1, 8, 3, 9, 4, 10, 5, 6])        # Feature 3

# Step 2: Generate y using the true relationship
y = 3*X1 + 5*X2 + 6*X3 + 9

# Step 3: Build the design matrix (with bias term)
# Here we are adding a column of 1s to account for the intercept
# Think of it as adding a column of X^0, which is the coefficent of the intercept
X = np.column_stack((np.ones(len(X1)), X1, X2, X3))

# Step 4: Normal Equation
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Estimated coefficients (β):") # The first is the intercept
print(beta)
```

This is the output of the above code:

Estimated coefficients (β): \[9. 3. 5. 6.\]

The solution for linear regression can give us really large weights. This happens when we are dealing with highly correlated data or data with a lot of features. Ridge regression solves this problem though, so its all good.

### Ridge Regression

The Problem:

* Ridge regression seeks to solve a problem with linear regression.
    
* Yes, linear regression has a great flaw: when dealing with much larger data, our solution (using the general formula) tends to get too large (a more accurate thing to say is that the weights become too specific).
    

The objective:

* The problem addressed with ridge regression is basically the same as linear regression, but we want to go further than just finding the slope and intercept of the best line; we also want to prevent the slope and intercept from getting too large, which may mess with our best-fit line, making it overly constrained to our training data.
    
* We want to be able to restrict our solution from getting too large. We want to be able to ***penalize*** large solutions to reduce their value.
    
* This process of penalizing our solutions is called ***regularization.***
    
* The addition of regularization to our objective changes our solution (but not too much).
    
* The objective is no longer to just minimize MSE (since we are going to be penalizing the solutions). The new objective is now to minimize:
    

$$newMSE \ (\beta) = \|y - X\beta\|^2 + \lambda \|\beta\|^2$$

Where:

* ∥y−Xβ∥2 is the MSE (same as linear regression).
    

$$λ∥β∥^2$$

* is the penalty term that keeps coefficients small.
    
* λ is a hyperparameter that controls the strength of the penalty.
    

The solution:

Solving the minimization problem, we get:

$$\hat{\beta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$$

As we can see, the only difference is the λ term, multiplied with a unit matrix. This term prevents the solutions from getting too small(reaching zero) or too large.

Here is the python implementation of Ridge regression from scratch, we are going to create our own model class:

```python
class RidgeRegression:
    def __init__(self, lam=1.0):
        self.lam = lam # This is the regularization parameter
        self.coef = None # This is our weights (accounting for both the slope and intercept)

    def fit(self, X, y):
        n_features = X.shape[1]
        I = np.eye(n_features) # The identity matrix
        self.coef_ = np.linalg.inv(X.T @ X + self.lam * I) @ X.T @ y # The Ridge regression formula

    def predict(self, X):
        return X @ self.coef # Multiply the input with the weights to predict new values
```

Ridge regression is preferred to linear regression, because it generalizes better *this basically means that the model would make better predictions on data that it has not seen before.*

Before we wrap up with linear supervised learning models, I want to mention another type of regression model. Lasso Regression also solves the problem of large coefficients, it uses L1 regularization to do this. *It is also worth mentioning that ridge regression uses L2 regularization.*

## Non linear supervised learning models

These models do not work like the linear models, they do not learn weights. These models learn rules. I would be implementing two classification models here: K Nearest Neighbors and Decision Tress. Both of which can be modified for regression.

What is classification? Classification is basically being able to put data into its respective groups(or classes, get it?). Being able to say a cat is a cat and a dog is a dog based on their features. KNN and decision trees learn rules to make these “classifications".

### KNN

The intuition:

This is one of the simplest (and laziest) machine learning models. Lets think of a problem: we have the heights and weights of cats and dogs (this is not nearly enough to be able to differentiate them, but it works for this illustration) and we want to be able to class new heights and weight. Lets now think of how we would normally differentiate cats and dogs, we are able to tell that a cat is a cat because all cats look alike right?

Using this logic we can use our data (heights and weights) to differentiate between cats and dogs by classifying them based on their similarity to known data i.e. how close the heights and weights are to each other.

Given a new height-weight pair, we can classify it by simply looking at all the heights and weights in our dataset, finding the closest one to our new data, then classifying our new data as the class of the closest data point. This might have been a little confusing, so lets break it down.

The problem:

* We have features X1,X2 and a label Y. Unlike the regression problem, Y is not a number it is a class: cats or dogs, tall or short, old or young etc.
    
* We want to be able to determine the class of new X1,X2 pairs.
    

The solution:

* Store the data set (this is why KNN is a lazy model, it just stores the whole dataset).
    
* Given new data, calculate the distance from all the data points in the stored dataset. The most common way to calculate distances is to use the Euclidean distance.
    
* The Euclidean distance between two points P1 and P2 is given by:
    

$$d(P_1, P_2) = \sqrt{(x_1 - x_2)^2 + (z_1 - z_2)^2}$$

* where x is height and z is weight, according to the previously mentioned example.
    
* Pick out the K shortest distances(The K nearest neighbors).
    
* Classify the new data as the class of the most prevalent of the K nearest neighbors. K is an arbitrary number that you choose. It is preferable that K is an odd number, to avoid voting ties.
    

This is an example of classification using KNN:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759047321059/f41e919a-4378-4687-8370-e39a0a8c4487.png align="center")

In the above plot, we can boldly classify the new query point as a cat, the 3 nearest neighbours consists of 2 cats and 1 dog. WIth cats being the majority of the KNN, cat is going to be the predicted class.

This is a python implementation:

```python
import numpy as np

# Dataset (X1 = height, X2 = weight)
X = np.array([
    # cats (class 0)
    [25, 2.5],
    [27, 2.8],
    [30, 3.0],
    [35, 3.4],
    [28, 2.9],
    [22, 2.2],
    [32, 3.1],
    [26, 2.7],
    [29, 3.0],
    [31, 3.2],
    # dogs (class 1)
    [50, 12.0],
    [55, 14.5],
    [48, 11.0],
    [60, 18.0],
    [52, 13.2],
    [47, 10.5],
    [58, 16.0],
    [54, 13.8],
    [49, 11.4],
    [56, 15.0],
])

# Labels: 0 = cat, 1 = dog
y = np.array([0]*10 + [1]*10)

# New data point to classify
new_point = np.array([40, 8.0])

# Step 1: Compute Euclidean distance from new_point to all points
distances = []
for i, x in enumerate(X):
    dist = np.sqrt((x[0] - new_point[0])**2 + (x[1] - new_point[1])**2) # Euclidean distance formula
    distances.append((dist, y[i]))

# Step 2: Sort distances to pick the shortest distances
distances.sort(key=lambda x: x[0])

# Step 3: Pick the 3 nearest neighbors, for k = 3, k can be any odd number
k = 3
neighbors = distances[:k]

# Step 4: Count classes
cats = sum(1 for d in neighbors if d[1] == 0)
dogs = sum(1 for d in neighbors if d[1] == 1)

# Step 5: Predict class
prediction = 0 if cats > dogs else 1
```

Note: Again, this is a simple implementation using only two classes, and only two features. In the real world we would usually have more, but this does not change our solution in any way, the same rules apply. But, a more general formula for Euclidean distance would be:

$$d(P_1, P_2) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

Where n is the number of features and i = (feature\_1, feature\_2 …… feature\_n)

A more general python implementation would be:

```python
class KNN:
  def __init__(self, k = 3): # Setting 3 as the default value for k
    self.k = k # K is the number of nearest meighbours to consider

  def fit(self, X, y): # Fitting the model to the triain data set
    # Fitting is basically just storing the values in the training data
    self.X = X
    self.y = y

  def euclidean_distance(self, x1, x2): # Function to measure closeness
    return np.sqrt(np.sum((x1 - x2)) ** 2)

  def predict(self, x):
    """
      Predict the value for a single data point, x:
      1. compute the euclidean distance from all the training data
      2. find the k nearest neighbours
      3. predict the class of the mode of the k nearest neighbours
    """
    distances = [self.euclidean_distance(x, X) for X in self.X]
    k_neighbours = np.argsort(distances)[:self.k]
    labels = [self.y[i] for i in k_neighbours]
    dct = {}
    for label in labels: # using a dictionary to count number of occurence
      if label in dct.keys():
        dct[label]+=1
      else:
        dct[label] = 1
    return  max(dct, key = dct.get)## predicting the most ocuring neighbour

# Train KNN model
model = KNN(k=7)
model.fit(X, y)

# Make Predictions
predictions = [model.predict(X_data) for X_data in X_test ]
```

To modify KNN for regression; the algorithm is the same for classification, until we reach the point to make a prediction. Classification predicts the majority class of the K nearest neighbors, while regression predicts the mean of the K nearest neighbors’ outputs. i.e. if we were predicting the age of the animals, given their heights and weights; then we would get the k closest heights and weight for the new data and average their ages.

Congratulations! We have implemented three machine learning models from scratch. On to the next.

### Decision Trees

The intuition:

Decision trees are a perfect example of machine learning models that learn rules. Lets go back to the cats and dogs problem. Another way we can differentiate them based on their heights and weights is by defining boundaries that split the heights and weights into cats and dogs. For example: *If the height is less than 50cm and the weight is greater than 1 lb, then it is a cat.*

This is essentially rule-based learning. We can create as many rules as we want. But we have to be careful:

* Too many rules, then the model learns overly specific rules, essentially memorizing the data (overfitting).
    
* Too few rules, then the model is too simple to classify correctly (underfitting).
    

Making rules like this would lead to a sort of tree structure. Each split is a question, and the answer leads to another split or a final decision. Here is an example of what we are trying to achieve with decision trees:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759047385085/6c632276-8dbf-446e-adfd-fc91bd85b38a.png align="center")

Each rectangle is called a node, and the arrows are called branches. The nodes that do not have any branches are known as leaf nodes.

So how exactly do we make these rules?

The problem:

* Given features X1, X2 and class Y; We want to split the data, one feature at a time, in a way that splits would lead to more pure groups (groups that mostly contain just cats, or just dogs).
    
* Example: in the image above. The split on weight &lt;= 4.8, leads to the right side containing only cats (2) and the left side containing only dogs (1).
    
* We can achieve this by making multiple splits. Since we do not want too many rules we would only make splits that give us the best outcome.
    

The objective:

We want to :

* Make **multiple splits** on the data.
    
* Choose splits that create the **purest groups** (lowest impurity).
    
* Avoid making unnecessary splits.
    

This minimizes the number of splits but maximizes the classification accuracy.

How do we measure the purity of a split?

We use a measure called Gini impurity. It determines the purity of a split by measuring how mixed the classes are within a group after the split.

$$Gini = 1 - \sum_{i=1}^{C} \left( \frac{n_i}{N} \right)^2$$

Where:

$$-\ n_i = number\ of\ samples\ in\ class\ i$$

$$-\ N = total\ number\ of\ samples\ in\ group\ after\ split$$

$$- \ C\ = number\ of\ classes$$

The solution:

1. For each feature in the data, try all possible split points.
    
2. Calculating the gini impurity of the groups, for each split point.
    
3. Choose the split that gives the highest reduction on impurity (this is the split that makes the resulting group the purest).
    
4. Create two new nodes (left and right, as shown in the decision tree diagram) from the split.
    
5. Repeat the steps from 1 to 4 for each node, treating each split group as its own dataset.
    

Stopping conditions:

* If all samples in a node belong to the same class.
    
* If the maximum tree depth is reached.
    
* If no split reduces impurity more than a set amount or zero.
    

*Stopping conditions are very important in machine learning. Most model algorithms are iterative processes that can run forever if given the chance to, or they can run for too long and give a nearly perfect model (for the data we have) and perform badly on data we have not seen before. Machine learning models should be defined with proper conditions to stop training.*

At the end, the model would have learnt a tree of decisions, where each internal node is a rule (height &lt;= 50.0), and each leaf node is a class prediction.

A decision tress is more complex than what we have covered before, so i would break its python implementation into digestible chunks of wood (*wink wink*)

**Step 1**: We would start by defining the model class initialization. We would make provisions for all the model hyperparameters and then the class for the nodes that do the actual splitting.

```python
class Node:
  # Defining the class for tree nodes
  def __init__(self, feature_index = None, threshold = None, left = None, right = None, *, value = None ):
    self.feature_index = feature_index  # index of the feature used for split
    self.threshold = threshold          # threshold for the split
    self.left = left                    # left child
    self.right = right                  # right child
    self.value = value                  # value (final prediction) if it's a leaf node

class DecisionTreeClassifier:
  # Defining the class for the classifier
  def __init__(self, max_depth = 5, min_samples_split = 2):
    self.root = None
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
```

The hyperparameters (these are values that control training) include the max depth (the depth in the example image is 3) and the minimum samples split is the minimum amount of data points that must be left after a split.

**Step 2**: We would define methods (functions) for calculating the Gini impurity and picking the best split.

```python
def gini(self, y):
    classes = np.unique(y)
    impurity = 1.0
    for clas in classes:
      p = np.sum(y==clas) / len(y)
      impurity -= p**2
    return impurity

 # Method to find the best split (greedy algorithm)
 def best_split(self, X, y):
   best_gain = -1
   split_idx, split_thresh = None, None
   current_impurity = self.gini(y) # initial impurity is impurity of the whole dataset
   n_features = X.shape[1]

   for f_indx in range(n_features):
     # Loop through all the features
     thresholds = np.unique(X[:, f_indx])
     for thresh in thresholds:
       # Loop through possible splits in the features
       left_idx = np.where(X[:, f_indx] <= thresh)[0]
       right_idx = np.where(X[:, f_indx] > thresh)[0]
       if len(left_idx) == 0 or len(right_idx) == 0:
           continue

       left_impurity = self.gini(y[left_idx])
       right_impurity = self.gini(y[right_idx])
       n = len(y)
       n_left, n_right = len(left_idx), len(right_idx)
       # Calculate impurity based on this split
       weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
       # Calculate Info gain, (we want to minimize whole impurity of nodes by maximizing Info Gain)
       info_gain = current_impurity - weighted_impurity

       if info_gain > best_gain:
         # If this split has the best info gain, then this is the best split
         best_gain = info_gain
         split_idx = f_indx
         split_thresh = thresh

   return split_idx, split_thresh, best_gain
```

The function for picking the best split returns the feature that the best split is made on, the number used for the split and the Gini impurity gain which we would need later.

**Step 3**: We would now define the method to build the decision tree.

```python
def build_tree(self, X, y, depth = 0):
    # Define recursive function to build trees
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    # condition for stopping
    if depth >= self.max_depth or n_labels == 1 or n_samples <self.min_samples_split:
      leaf_value = self.majority_class(y)
      return Node(value=leaf_value)

    # else we find the best split
    feature_index, threshold, gain = self.best_split(X,y)
    if gain == 0:
      # If the gain is 0, then make a leaf node
      leaf_value = self.majority_class(y)
      return Node(value = leaf_value)

    # else we update indexes of remaining data for more splitting
    left_idx = np.where(X[:, feature_index] <= threshold)[0]
    right_idx = np.where(X[:, feature_index] > threshold)[0]

    # we call build_function again to continue until the stopping conditions are met
    left_subtree = self.build_tree(X[left_idx], y[left_idx], depth + 1)
    right_subtree = self.build_tree(X[right_idx], y[right_idx], depth + 1)
    return Node(feature_index, threshold, left_subtree, right_subtree)

## function for finding majority class for leaf nodes0
  def majority_class(self, y):
    counts = np.bincount(y)
    return np.argmax(counts)

  def fit(self, X, y):
    # Fit tree, with first build as the root
    self.root = self.build_tree(X,y)
```

Building the actual decision tree is a recursive process, the `build_tree` function calls itself (just like in the steps for the solution where step 5 is to do steps 1 to 4 again until a particular condition is met) until one of the 3 stopping conditions are met.

**Step 4**: Lastly we would define the function to make predictions.

```python
  def _predict(self, node, x):
    if node.value is not None:
      return node.value
    if x[node.feature_index] <= node.threshold:
      return self._predict(node.left, x)
    else:
      return self._predict(node.right, x)

  def predict(self, data):
    return np.array([self._predict(self.root, x) for x in data])
```

The first function is to make a prediction on a single data point and the second one is to make a prediction on an array of data points.

The actual code for the implementation of the decision tree is admittedly quite complex. But, if you only read the intuition, problem and solution behind decision trees. Then you would understand what all the hyperparameters do and you should be able to train more efficient decision trees.

Knowledge of Decision Trees can also be applied to Random Forests. Random forests is just a bunch of decision trees, then a prediction is made by a vote among these trees. The trees are trained by randomly picking a part of the features (3 out of the 4 features in the dataset) of the data and training the tree using only those features. This is done a number of times to create a Random Forest. Many trees → Forest, whoever named these things is a genius.

Moving on from supervised learning, we would now look at unsupervised machine learning models that are trained on unlabeled data.

## Unsupervised learning models

I have already explained unsupervised learning. Clustering is an unsupervised learning problem much similar to classification; the only difference is that we do not tell the model what is what (who is a cat and who is a dog). We just give the model the data and it makes groups by itself, it basically clusters datapoints (get it?) putting them into different groups.

A machine learning model that solves this clustering problem is called Kmeans clustering.

### KMeans

The intuition: We have a bunch of data points, and we want to divide them into K (this is the number of clusters you want the model to learn) distinct groups, called clusters. Points within the same cluster should be similar to each other, while points in different clusters should be as different as possible (points in a group should be alike and points in different groups should be different). In order to achieve this, we pick K data points that represent the middle of the clusters, called centroids. We want to pick these centroids in a such a way that they are well separated. Once the centroids are chosen, each data point is assigned to the nearest centroid, forming clusters around them. In practice, instead of directly maximizing distances between clusters, we minimize the total distance between data points and their assigned centroid.

*If a centroid is placed too close to another centroid, it would increase the distance from points to their assigned centroids, making clusters less compact. On the other hand, when centroids are far from each other, each cluster covers its own “region”, so data points are naturally closer to their centroid i.e., the total distance between data points and their assigned clusters is minimized.*

Lets visualize this:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759091808000/3c020094-32ef-4b34-8721-f6203b1e3528.png align="center")

on the left, we have the centroids really close to each other. As seen in the image, this makes data points far from their assigned centroids. On the left, we have the centroids well separated. The data points are no longer too far from their assigned data point. If we focus on minimizing the distances from data points to their assigned centroids, we would eventually arrive at proper clusters.

Lets clearly define the problem, objective and the solution.

The problem:

* We have a bunch of data points, pairs, triplets or quadruplets of numbers.
    
* We also want to split these data points into K groups (clusters).
    

The objective:

* We would achieve these K clusters by finding K centroids.
    
* We want to reduce the total distances between data points and their assigned centroids (data points are assigned to the closest centroid of course).
    
* Therefore, our objective is to pick K centroids such that the function below is minimized
    

$$J = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} \, \| x_i - \mu_k \|^2$$

J is known as the sum of Within Cluster Sum of Squares, WCSS.

where:

$$\mu_k\ is\ the\ centroid\ of\ cluster\ K$$

$$r_{ik} = \begin{cases} 1 & \text{if data point } x_i \text{ belongs to cluster } k, \\ 0 & \text{otherwise}. \end{cases}$$

The solution:

* How do we minimize the WCSS? Just like in the linear regression, we solve it mathematically.
    
* Step 1: Expand WCSS.
    

$$\begin{aligned} J(\mu) &= \sum_{i=1}^{n} (x_i - \mu)^\top (x_i - \mu) \\ &= \sum_{i=1}^{n} \big( x_i^\top x_i - 2 x_i^\top \mu + \mu^\top \mu \big) \\ &= \sum_{i=1}^{n} x_i^\top x_i \;-\; 2 \left(\sum_{i=1}^{n} x_i\right)^\top \mu \;+\; n\,\mu^\top \mu. \end{aligned}$$

* Step 2: Compute the gradient with respect to mu.
    

$$Differentiate\ J(\mu)\ with\ respect\ to\ \mu\ (using\ matrix\ calculus\ rules):$$

$$\nabla_{\mu} J(\mu) \;=\; -2 \sum_{i=1}^{n} x_i \;+\; 2 n \mu.$$

* Step 3: Equate to zero and solve.
    

$$-2 \sum_{i=1}^{n} x_i + 2 n \mu \;=\; 0 \quad\Longrightarrow\quad \mu \;=\; \frac{1}{n}\sum_{i=1}^{n} x_i.$$

From the solution, we can see that the value for the centroid that minimizes the WCSS is merely the mean of the cluster. We can now define an algorithm to learn the clusters that minimize the WCSS.

The Algorithm:

1. Randomly select K points that would act as the initial centroids.
    
2. Assign points to clusters based on the nearest centroid.
    
3. Calculate the mean of each cluster.
    
4. The mean of each cluster is selected as its new centroid.
    
5. Continue from step 2.
    

Stopping conditions:

* The maximum number of iterations is exceeded.
    
* The new centroid is not much different from the old one.
    

Lets see what happens from following these steps.

This is the first step where the centroids are chosen at random:

*The data was generated in such a way that the three clusters should be fairly obvious, so we can see where we should have our centroids.*

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759403524746/7a65e65c-ad55-46ca-8e90-3f48e4d29559.png align="center")

Then after a few iterations, the centroids can be seen to approach better positions:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759421850109/11696832-a5df-4644-a9bc-a0f198186cba.png align="center")

Then finally, when the centroids do not change any further (this is called convergence), we get this:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759421934130/990f4bf8-1913-44fc-8942-a97063fdea71.png align="center")

This is what the algorithm does, it recalculates the centroid over and over again, until the stopping conditions are met.

Now let us implement the algorithm with python code:

First, we import all necessary modules, then we write the initialization method for the Kmeans class

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
  # Initialization of the KMeans class
  def __init__ (self, n_clusters = 3, max_iter = 100, tolerance = 1e-14, random_state = 42):
    """
      n_clusters is the number of clusters
      max_iter is the maximum number iterations (going through the algorithm steps __ times)
      tolerance is the tolerance for change in centroid, 
      i.e. if the difference of the last centroid and the centroid chosen 
      in the current step does not exceed this value, STOP training
    """
    self.n_clusters = n_clusters # K
    self.max_iter = max_iter
    self.tol = tolerance
    self.random_state = random_state
    self.centroids = None
```

In the code above, we have the Kmeans class initialized with attributes that basically control how the model learns.

Second, we define the method for the actual learning, this is where we use the Kmeans algorithm

```python
 def fit(self, X):
    # Defining the fit method, to train the model
    np.random.seed(self.random_state) # to make the model recreateable, due to the random element of the algorithm
    n_samples, n_features = X.shape
    ## Training algorithm
    # Step 1 : Random;y select clusters from the data points
    random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
    self.centroids = X[random_indices]

    for i in range(self.max_iter):
      # Step 2 : Assign each data point to the neares centroid
      self.labels = self.predict(X)

      # Step 3 : Compute the new centroids as the mean of the clusters 
      # This is the centroid value that minimizes the WCSS the most, 
      # essentially acheiving our objective (minimizing WCSS) iteratively
      new_centroids = np.array([X[self.labels == k].mean(axis = 0) for k in range(self.n_clusters)])

      # Step 4 : Check for tolerance, (if the centroids are indeed changing i.e. Converging)
      if np.all(np.abs(new_centroids - self.centroids) < self.tol):
          break
      # Assign the means of the clusters as the new centroids if there is indeed convergence, else the algorithm stops
      self.centroids = new_centroids
```

We firstly select clusters based on the randomly selected centroids. Then we start the algorithm loop that runs only a specific number of times or stops when there is no longer a minimum change in the centroid.

Finally, we would define the methods to make predictions and another to plot the learnt clusters.

```python
def predict(self, X):
    # Method to assign the datapoints to the clusters of the nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
    return np.argmin(distances, axis=1)

  def plot_clusters(self, X):
    # Method to visualize centroids and clusters
    # Only plots it using 2 features
    plt.figure(figsize = (10,10))
    for k in range(self.n_clusters):
      cluster_points = X[self.labels == k]
      plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {k}")
      plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()
```

The new data points are assigned to the cluster of the closest centroid.

We are done with the Kmeans clustering implementation. This sort of algorithm that continuously minimizes a particular function as seen in KMeans and Decision trees, is the backbone of Machine learning. Most complex models (especially Neural Networks) are trained using an iterative process. At the end of each iteration, a value is calculated (Gini Impurity in Decision Trees) and then that value is minimized in some way (Picking the mean as the new centroid in KMeans). Understanding this concept is paramount to being able to effectively train machine learning models.

Another unsupervised learning problem is Dimensionality reduction, and a machine learning model that solves this problem is Principal Component Analysis.

### Principal Component Analysis

This is another unsupervised learning algorithm; it is a machine learning model for dimensionality reduction. In the real world, we might come across data that has a lot of features. We normally can’t plot data with more than 3 features (that would require more than 3 axes). If we were to investigate the correlation or spread of such data, it would be tedious to consider every feature. Thats where dimensionality reduction comes in; it reduces the dimension of data while keeping its correlation (how data points are related to each other) and spread (how data is distributed) intact. PCA seeks to describe the correlation and spread of data using fewer features.

The truth is that we do not actually use any sort of complicated algorithm to solve this. The solution is highly mathematically. I will just walk through the steps taken, with an example data set:

|  | feature\_1 | feature\_2 | feature\_3 | feature\_4 |
| --- | --- | --- | --- | --- |
| point\_1 | 1.0 | 2.0 | 1.5 | 2.2 |
| point\_2 | 0.9 | 1.8 | 1.6 | 2.1 |
| point\_3 | 1.1 | 2.1 | 1.4 | 2.0 |
| point\_4 | 8..0 | 7.5 | 8.2 | 7.8 |
| point\_5 | 7.9 | 7.8 | 7.6 | 8.1 |
| point\_6 | 8.2 | 7.9 | 7.9 | 8.0 |

*To relate to real world data. The data could be data of the employees in a company. The features can represent things like, salary range, age, years of experience, productivity etc. The points represent the employees. Given data like this, if we were asked to group the employees and make a plot on the groups, we would need PCA to do this.*

Before we get into the steps for PCA, i want you to carefully examine the data. We have 6 data points (rows) and 4 features (columns). Merely investigating the values of each row, it becomes obvious that the first three are similar to each other (they have values ranging from 0.9-2.2 across all features) and the last three are also similar but different from the first three (they have values ranging from 7.5-8.2 across features). Our goal with PCA is to keep this correlation while reducing the matrix to have only 2 columns. We want to retain the '“clusters“ and reduce the matrix columns.

*The general goal of PCA is to reduce features and keep correlation, so the number of features can be reduced to 3 or even 1, or even higher than that. Based on the needs of the user and the initial number of features. Also note that we cannot increase the number of features with PCA, only reduce.*

Lets get into the actual steps:

1. Convert data to matrix format: This is not really an “official” step, but it is important to note that PCA operates on matrices. The data of rows representing data points and columns representing features; should be converted to a matrix with the same format of points and features.
    

```python
X = [
 [1.0, 2.0, 1.5, 2.2],   # cluster A
 [0.9, 1.8, 1.6, 2.1],   # cluster A
 [1.1, 2.1, 1.4, 2.0],   # cluster A
 [8.0, 7.5, 8.2, 7.8],   # cluster B
 [7.9, 7.8, 7.6, 8.1],   # cluster B
 [8.2, 7.9, 7.9, 8.0],   # cluster B
]
```

2. Mean-center the data: To do this, we will firstly compute the mean of each column of the matrix. Then subtract each value by the mean of its respective column. Here is the mean of each column, then the mean centered data:
    

$$\mu = \begin{bmatrix} 4.5167 & 4.8500 & 4.7000 & 5.0333 \end{bmatrix}$$

$$Subtracting\ \mu\ from\ each \ row\ gives the\ mean-centered\ matrix\ X_{\text{centered}}$$

$$X_{\text{centered}} = \begin{bmatrix} -3.5167 & -2.8500 & -3.2000 & -2.8333 \\ -3.6167 & -3.0500 & -3.1000 & -2.9333 \\ -3.4167 & -2.7500 & -3.3000 & -3.0333 \\ \;\;3.4833 & \;\;2.6500 & \;\;3.5000 & \;\;2.7667 \\ \;\;3.3833 & \;\;2.9500 & \;\;2.9000 & \;\;3.0667 \\ \;\;3.6833 & \;\;3.0500 & \;\;3.2000 & \;\;2.9667 \end{bmatrix}$$

3. Compute the covariance matrix, C of the mean-centered data:
    

* In statistics, **covariance** measures how two variables change together.
    
    * If two variables increase together, their covariance is positive.
        
    * If one increases while the other decreases, their covariance is negative.
        
    * If they are unrelated, covariance is close to zero.
        
* In PCA, we don’t just want to know the spread of each individual feature (that’s variance), but also how features relate to each other (covariance).
    

Where n is the number of rows, the covariance matrix is given by:

$$C = \frac{1}{n-1} X_{\text{centered}}^\top X_{\text{centered}}$$

$$C = \begin{bmatrix} 14.854 & 12.179 & 13.506 & 12.375 \\ 12.179 & 10.003 & 11.048 & 10.158 \\ 13.506 & 11.048 & 12.328 & 11.248 \\ 12.375 & 10.158 & 11.248 & 10.339 \end{bmatrix}$$

Note: the matrix multiplication happening is (4 by 6) X (6 by 4) which will produce a (4 by 4) matrix as seen above.

4. Perform Eigen Decomposition on C: Now this is actually the most important part of PCA, also the most technical. I wouldn’t go into what exactly is done here mathematically. I will just explain what eigen decomposition is intuitively. Eigen decomposition breaks down the covariance matrix into two, the eigen vectors and the eigen values.
    

* An **eigenvector** (principal component direction) is basically a new feature that captures variance without “changing” the direction. These are the features that would describe our data.
    
* An **eigenvalue** measures how much variance the data has along its eigenvector. Larger eigenvalue means more variance along that direction i.e. more “information” is retained by using its respective eigenvector for describing the data. The eigenvalues in descending order are:
    

$$\lambda \approx \{47.443,\; 0.0646,\; 0.0155,\; 0.00005\}$$

The eigen vectors:

$$V = \begin{bmatrix} -0.559 & 0.028 & 0.496 & -0.663 \\ -0.459 & 0.534 & 0.308 & 0.640 \\ -0.509 & -0.789 & -0.338 & -0.004 \\ -0.466 & 0.302 & -0.744 & 0.412 \end{bmatrix}$$

Here, the first eigenvalue is the largest, this means that its corresponding eigenvector (These would become the new features, the Principal Components) would retain the most information.

When I say ‘new feature ‘, I do not mean it literally. To get the new features we would still need to project the data onto the Principal Components.

5. Project the centered data unto the top 2 components: We firstly pick the top 2 eigenvectors based on the eigenvalues.
    

$$W = \begin{bmatrix} -0.559 & 0.028 \\ -0.459 & 0.534 \\ -0.509 & -0.789 \\ -0.466 & 0.302 \end{bmatrix}$$

$$X_{\text{reduced}} = X_{\text{centered}} \, W$$

$$X_{\text{reduced}} \approx \begin{bmatrix} 6.225 & 0.048 \\ 6.368 & -0.170 \\ 6.267 & 0.123 \\ -6.236 & -0.413 \\ -6.152 & 0.308 \\ -6.472 & 0.103 \end{bmatrix}$$

Now we have a 6 by 2 matrix, this is the final output of PCA, a reduced matrix that still captures the variance and the correlation of the original data. Remember how our data had a very apparent distinction between two groups, the first three and the last three. Once again, we can look at our data and see the clusters once again, let us plot this to make it clearer:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759483591137/a76ba785-79d7-4167-8f38-89e417bdd1bf.png align="center")

The above image shows the plot of the reduced data. We can see that the clusters still remain, with the first three points being similar and the last three being similar. (the PCA also sees the 4th point as the most different, but it’s still not by a lot).

The python implementation of PCA:

```python
# Mean cente the data
X_meaned = X_train - np.mean(X_train, axis = 0)

# Compute the covariance matrix
cov_matrix = np.cov(X_meaned, rowvar = False)
print("Covariance:\n", cov_matrix)

# Eigen decomposition of covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort the eigen vectors and values
sorted = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted]
eigenvectors = eigenvectors[:, sorted]

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors (principal components):\n", eigenvectors)

# 6. transform data using the eigen vecotrs 6D -> 3D
X_pca = np.dot(X_meaned, eigenvectors)

# PCA-transformed data, should find 3 clusters
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')  # 3D axis

ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2],
           c='red', alpha=0.6)

ax.set_title("After PCA (6D → 3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()
```

The example above, reduces the data from 6 features to 3 features and then makes a plot.

With this you can understand the necessity of PCA and what it does exactly.

We have come to the end of the article. I hope you have been able to learn a thing or two.

I would also write another article on deep learning, explaining hoe neural networks work and also exploring neural network architectures like CNN, RNN and LSTMs.

References:

1. [Machine learning I by Columbia](https://plus.columbia.edu/content/machine-learning-i).
    
2. PCA by stat quest on YouTube [here](https://youtu.be/FgakZw6K1QQ?si=dK6Pz-jRZV4XgS6E).
    
3. PCA by Steve Brunton on YouTube [here](https://youtu.be/fkf4IBRSeEc?si=izUvE9tgOXAZH8Uq).
    
4. All images were plotted with python, here is a jupyter notebook with all the plots [here](https://colab.research.google.com/drive/1OHbEb-Zh_kEDly-RdhMMU5K6nX2C_9pz?usp=sharing).
    
5. GitHub repo containing all the implementations [here](https://github.com/Badaszz/ML_models_from_scratch).
    
6. KMeans Clustering by stat quest on YouTube [here](https://youtu.be/4b5d3muPQmA?si=dyQR7QpzS-Zp-rHh).