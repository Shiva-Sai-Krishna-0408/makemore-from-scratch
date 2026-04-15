# Makemore-from-scratch

## Introduction
Building Bigram model and MLP model from scratch following Andrej Karpathy's makemore series from youtube.

## Bigram Model
Bigram model is a simple neural network with no hidden layers. It takes a single character as input and predicts the next character using learned probabilities. The model is trained by optimizing the negative log-likelihood loss over the full dataset using gradient descent.

## Multi-layer Perceptron Model
Multi-layer Perceptron model (MLP) - In this model, we take 3 inputs characters from the dataset and try to predict the 4th character. We use pytorch to optimize the training process and to calculate backpropagation with efficiency. Calculate loss by batches splitting the    dataset into 3 splits. 80% training split, 10% dev/validation split, 10% testing split.  Plotting the step vs loss curve and then visualizing the character's embeddings in a 2D grid plot. Determining the training and dev/validation loss at the end. The result of the dev loss obtained (2.1659) is slightly less than Karpathy's in the video (2.1701) and he challenges it himself to improve the loss if it can by twisting the knobs.
