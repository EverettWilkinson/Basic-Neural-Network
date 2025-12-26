# Feed-Forward Neural Network from Scratch

A fully functional multi-layer perceptron (MLP) implementation built entirely with NumPy, demonstrating core deep learning concepts without relying on high-level ML frameworks.

## Overview

This project implements a customizable feed-forward neural network capable of multi-class classification. Trained and tested on the MNIST handwritten digit dataset, it achieves strong performance using only fundamental NumPy operations.

## Key Features

### Core Implementation
- **Custom backpropagation algorithm** with gradient descent optimization
- **Flexible architecture** supporting multiple hidden layers of arbitrary sizes
- **Multiple activation functions**: ReLU, Sigmoid, Tanh, and Softmax
- **He initialization** for weights to prevent vanishing/exploding gradients
- **Stochastic Gradient Descent (SGD)** with mini-batch training

### Mathematical Components
- Cross-entropy loss function for multi-class classification
- Analytical derivatives for each activation function
- Efficient vectorized operations for forward and backward passes

## Architecture

The current implementation uses a 3-layer network:
- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (digit classes 0-9)

## Technical Highlights

1. **From-Scratch Implementation**: No TensorFlow, PyTorch, or scikit-learn—just NumPy and math
2. **Proper Gradient Computation**: Implements the full backpropagation algorithm with chain rule
3. **Batch Processing**: Efficient mini-batch SGD for faster convergence
4. **Modular Design**: Easy to modify layer sizes, activation functions, and hyperparameters

## Requirements

```bash
numpy
```

## Dataset

The project uses the MNIST dataset:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- Format: 28×28 grayscale images of handwritten digits (0-9)

Download the MNIST dataset files and place them in your project directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Usage

```python
# Initialize network layers
l1 = NeuralNetwork.Layer(784, 128, activeFunction="ReLU")
l2 = NeuralNetwork.Layer(128, 128, activeFunction="ReLU")
l3 = NeuralNetwork.Layer(128, 10, activeFunction="Softmax")

# Create MLP with cross-entropy loss
mlp = NeuralNetwork.MLP([l1, l2, l3], "Cross-Entropy", learningRate=0.001)

# Train with mini-batch SGD
mlp.SDG_batch(labels, images, batch_size=4, batch_number=120000)

# Evaluate accuracy
accuracy = mlp.accuracy(test_labels, test_images)
```

## Results

The network achieves competitive accuracy on MNIST digit classification, demonstrating that fundamental neural network concepts can be implemented effectively without specialized libraries.

## Learning Outcomes

This project reinforced understanding of:
- The mathematics behind backpropagation and gradient descent
- How different activation functions affect learning
- The importance of weight initialization strategies
- Mini-batch training and its impact on convergence
- Vectorization for computational efficiency

## Future Enhancements

- Add L2 regularization and dropout for better generalization
- Implement additional optimizers (Adam, RMSprop)
- Add learning rate scheduling
- Create visualization tools for training progress
- Support for convolutional layers

## License

MIT License

---

**Built with**: Python, NumPy, and Mathematics
