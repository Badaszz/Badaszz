---
title: "Getting Started with Image Classification Using CNNs: Traffic Signs and Plant Diseases."
datePublished: 2025-04-19T13:48:59.238Z
cuid: cm9o9xxph000d0alh5smk8i72
slug: getting-started-with-image-classification-using-cnns-traffic-signs-and-plant-diseases
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1745065844068/19b3c427-a057-41ef-a189-882a4bcb13d3.png
tags: python, machine-learning, cnn-for-begginers, cnns-convolutional-neural-networks, streamlit-generative-ai-data-science-machine-learning-python-web-development-and-session-management

---

## Introduction

In the world of artificial intelligence, computer vision and image classification are one of the most widely used applications, allowing machines to identify and categorize images based on visual features. From autonomous vehicles recognizing traffic signs to helping farmers detect plant diseases, image classification models are transforming industries and revolutionizing robotics.

In this blog post, I will walk you through two of my image classification projects: **Traffic Sign Classification** and **Leaf Disease Detection**. Both projects leverage **Convolutional Neural Networks (CNNs)**, a deep learning architecture highly effective for image recognition tasks. You’ll get an insight into the full development process, from data preprocessing to model deployment, and the challenges faced along the way.

By the end of this post, you should have a solid understanding of how to approach an image classification task and be able to apply similar techniques to your projects.

## Tools and Technologies Used:

* **Programming Language**: Python
    
* **Deep Learning Framework**: TensorFlow/Keras
    
* **Deployment Tool**: Streamlit
    
* **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark), Plant Disease Dataset
    
* **Other**: GitHub for version control, Matplotlib for visualization and Google Colab for model training
    

# Project 1: Traffic sign classification

## The problem

The goal here was to build an image classification model capable of accurately recognizing and categorizing the traffic signs in the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. This is a key step in building self-driving car systems, where road sign recognition plays a crucial role in safe navigation.

## Dataset Overview

The **German Traffic Sign Recognition Benchmark (GTSRB)** dataset is a collection of images representing various traffic signs commonly seen on German roads. The dataset contains over **50,000 images** of 43 different classes of traffic signs, including stop signs, speed limits, yield signs, and more.

## The Approach

For this project, I used a **Convolutional Neural Network (CNN)** to classify the images. CNNs are well-suited for image recognition tasks because they can automatically detect important features, such as edges and textures, in an image without needing manual feature extraction.

firstly, the required libraries were imported

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from PIL import Image
import cv2
import pickle
import csv
import random
import pandas as pd
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
```

### Data preprocessing

Before feeding the images into the model, I performed **normalization**, where the pixel values were normalized to a range between 0 and 1 by dividing the pixel values by 255 (since they originally ranged from 0 to 255).

**Note**: I am reading values from a folder containing pickle data.

Then I performed one-hot encoding on the labels.

One-hot encoding is a technique used to convert **categorical data** (in this case traffic sign names) into a numerical format that machine learning models can understand. Instead of assigning arbitrary numbers to categories (which might imply an order), one-hot encoding creates a **binary vector** for each category.

```python
#function for reading the pickle data
def load_p_data(fle):
  #reading the file
  with open(fle, 'rb') as f:
    #reading the contents of the pickle file as a dictionary
    d= pickle.load(f, encoding='latin1')
    feats = d['features'] #4D numpy array containing raw pixel data of the traffic sign images
    labels = d['labels'] #1D numpy array containing the label id of the traffic sign images (labels are encoded)
    sizes = d['sizes'] #2D numpy array containing the sizes of the images i.e. (width,height)
    coords = d['coords'] #2D array contains the coordinates for a bounding frame arround the image
  return feats, labels, sizes, coords


train_features, train_labels, train_sizes, train_coords = load_p_data("/content/train.p")
test_features, test_labels, test_sizes, test_coords = load_p_data("/content/test.p")
valid_features, valid_labels, valid_sizes, valid_coords = load_p_data("/content/valid.p")

#create a function that normalizes the images
image_datagenerator = ImageDataGenerator(rescale = 1/255)

#OneHotEncode labels

train_labels = to_categorical(train_labels, num_classes = 43)
test_labels = to_categorical(test_labels, num_classes = 43)
valid_labels = to_categorical(valid_labels, num_classes = 43)

#generating new images and normalizing them
train_data = image_datagenerator.flow(train_features, train_labels, batch_size = 43)
test_data = image_datagenerator.flow(test_features, test_labels, batch_size = 43)
valid_data = image_datagenerator.flow(valid_features, valid_labels, batch_size = 43)
```

### CNN Architecture

I designed a simple CNN architecture using **Keras**, with the following layers:

1. **Conv2D**: The first convolutional layer detects basic features (edges, corners) in the image.
    
2. **MaxPooling2D**: After each convolutional layer, a pooling layer is applied to down sample the image, reducing its size and computational complexity.
    
3. **ReLU Activation**: ReLU activation helps introduce non-linearity, allowing the network to learn more complex patterns.
    
4. **Flattening**: The 2D matrix output from the convolutional layers is flattened into a 1D vector.
    
5. **Dense Layer**: Fully connected layers that perform the final classification based on the features learned in the previous layers.
    

```python
#Building the CNN model architeture
model = Sequential()

model.add(Conv2D(filters = 43, kernel_size = 5, activation = 'relu', padding = 'same', input_shape = [32 , 32, 3]))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

#output layer
model.add(Dense(43, activation = 'softmax')) #number of neurons should be the number of classes and this is the last class

#compile model and visualize
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary() # to visualize the model and output the number of parameters
```

## Training the Model

I trained the CNN model for **50 epochs** with a batch size of **32**, using the **Adam optimizer** and **categorical cross-entropy** loss function. The training process involved adjusting hyperparameters like the learning rate and batch size to minimize the loss and improve accuracy.

During training, I used **early stopping** to monitor the validation loss and stop training once the model stops learning (i.e., the loss does not get better or the accuracy does not get better for multiple epochs), preventing overfitting.

## Model Evaluation

After training, the model achieved an impressive **accuracy of 91%** on the test set. I also evaluated the model using a **confusion matrix**, which helped identify misclassifications. For instance, some classes with visually similar signs were often misclassified.

```python
#defining the early stopping of the model
early_stopping = EarlyStopping( monitor = 'val_loss', patience = 5, verbose = 1, restore_best_weights = True)

#training the model
model_history = model.fit(x = train_data, epochs = 20, validation_data = valid_data, callbacks = [early_stopping])

#visualizing losses
plt.plot(model_history.history["loss"], label = "Train Loss")
plt.plot(model_history.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

#model evaluation
model.evaluate(test_data)

#plotting a confusion matrix
#defining function to get the confusion matrix
def prediction_with_confusion_matrix(test_file, model):
    x_test, y_test, _, _ = load_p_data(test_file)
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow(x_test, y_test, batch_size=43, shuffle=True)
    y_pred = []
    y_true = []
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        batch_pred = model.predict(batch_x)
        batch_pred_classes = np.argmax(batch_pred, axis=1)

        y_pred.extend(batch_pred_classes)
        y_true.extend(batch_y)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    unique_labels = np.unique(y_true)
    return confusion_matrix(y_true, y_pred, labels=unique_labels)

cmt = prediction_with_confusion_matrix("/content/test.p", model)

plt.figure(figsize=(17, 17))
sns.heatmap(cmt, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#saving the model
model.save('traffic_sign_model.h5')

#saving the labels in a csv file 
filename = 'labels.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write each row from the list
    writer.writerows(labels)
```

## Deployment

The next step was to make the model **accessible and interactive**. That’s where **Streamlit** came in —a lightweight Python framework that lets you integrate machine learning models with shareable web apps with just a few lines of code.

## Why Streamlit?

I chose Streamlit because:

* It’s incredibly fast to set up (no need for HTML, CSS, or JavaScript).
    
* It supports interactive widgets like file uploaders, buttons, and sliders.
    
* It’s great for visualizing model predictions in real-time.
    

Key steps in the code:

1. **Model Loading**: The trained model (`.h5` file) is loaded using`keras.models.load_model()`
    
2. **Image Upload**: Users can upload an image.
    
3. **Preprocessing**: The image is resized, normalized, and reshaped to match the model’s input requirements.
    
4. **Prediction**: The model makes a prediction, and the result is mapped to the corresponding class name.
    
5. **Display**: The uploaded image and prediction result are displayed using `st.image()`.
    

#### From Local to Live

To make the app accessible online:

* I hosted the project on **GitHub**.
    
* Then deployed it via **Streamlit Community Cloud**, which is free and very beginner-friendly.
    

```python
#loading trained model
model = load_model('traffic_sign_model.h5')

#classes
#read the label.csv file 
# Load the CSV file
with open('labels.csv', 'r') as file:
    reader = csv.reader(file)
    data = [list(row) for row in reader]  # Convert strings to integers

CLASS_NAMES = []
for i in range(len(data)):
    CLASS_NAMES.append(''.join(data[i]))

#making the prediction
prediction = model.predict(np.expand_dims(img_rescaled, axis=0))
Y_pred = np.argmax(prediction, axis=1)
  
#displaying the prediction      
st.title(str("The traffic sign is " + CLASS_NAMES[Y_pred[0]])) 
st.title(str("You're welcome broski" ) )
st.balloons()
```

Note : the above code is not the full code, the full code can be gotten here:

[traffic sign classification](https://github.com/Badaszz/traffic-sign-classification)

The second project basically follows the same process

[Plant disease classification](https://github.com/Badaszz/plant-disease-detection-streamlit)

#### Final Thoughts on streamlit

Using Streamlit turned a plain machine learning model into a **usable tool**. It not only made the model interactive but also showcased how deep learning can be applied to real-world problems.

## What’s Next?

These projects were a great learning experience, but there’s always room for improvement. I’m planning the following next steps:

1. **Transfer Learning**: I want to experiment with pre-trained models like **ResNet** or **VGG16**, which might perform better with less training data.
    
2. **Mobile Deployment**: I’m exploring options to deploy these models on **mobile devices** or **edge devices** for real-time applications in agriculture or transportation.
    
3. **Explainability**: I’m interested in adding **Grad-CAM** visualizations to show which parts of the images the model is focusing on when making predictions.
    

## Conclusion

These image classification projects have been a fun and educational journey. They have deepened my understanding of CNNs, data preprocessing, and model deployment, and I’m excited to continue improving these models. I hope this post has helped you understand the steps involved in image classification and inspired you to dive into your own projects.

Feel free to explore the code on GitHub:

* [Traffic Sign Classification GitHub](https://github.com/Badaszz/traffic-sign-classification)
    
* [Leaf Disease Detection GitHub](https://github.com/Badaszz/plant-disease-detection-streamlit)