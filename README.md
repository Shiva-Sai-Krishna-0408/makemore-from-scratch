#Makemore-from-scratch

##Introduction

Rebuilding Andrej Karpathy's makemore series from scratch — typed every line by hand, no copy-pasting. The goal was to build intuition for neural network training from first principles before moving into production AI systems.

##Part 1 — Bigram Model
Built a simple bigram character-level language model with no hidden layers. Learned how neural networks learn probability distributions, how negative log-likelihood loss works, and how gradient descent updates weights. First real experience reading and writing PyTorch training loops from scratch.

##Part 2 — Multi-Layer Perceptron
Scaled up to a 3-character context window MLP. Implemented the full training pipeline: embedding lookup, hidden layer, softmax output, cross-entropy loss, mini-batch gradient descent, and train/dev/test splits. Beat Karpathy's dev loss benchmark of 2.1701 with 2.1659.

##Part 3 — Training Diagnostics & Batch Normalization
The hardest part. Learned to diagnose broken training by plotting activation distributions, gradient distributions, and update/data ratios. Discovered and fixed tanh saturation at initialization. Implemented Kaiming initialization, batch normalization with running mean/std buffers, and refactored everything into PyTorch-style classes (Linear, BatchNorm1d, Tanh) written from memory. Final dev loss: 2.08.

Key lessons: broken training is invisible without visualization. Small init bugs waste thousands of gradient steps. Deeper isn't always better without careful tuning.
