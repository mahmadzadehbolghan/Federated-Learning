# README: CIFAR-10 Classification Using Convolutional Neural Networks (CNN)

## Overview
This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes, making it a standard benchmark for image classification tasks. The code preprocesses the dataset, builds a CNN model, trains it, and evaluates its performance.

## Dataset
- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Classes**:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

The dataset is automatically loaded and split into training and test sets using TensorFlow's `keras.datasets` module.

## Preprocessing
1. Labels are reshaped into a 1D array.
2. Image pixel values are normalized by dividing by 255.0 to scale them between 0 and 1.
3. Basic label inspection is performed by printing the first few test labels.

## Model Architecture
The project uses a **Convolutional Neural Network (CNN)** designed with the following layers:
1. **Convolutional Layers**:
   - Extract spatial features using filters of size \(3 \times 3\).
   - Feature maps are generated with 32 and 64 filters in two consecutive layers.
2. **Pooling Layers**:
   - Downsample feature maps using max-pooling with a \(2 \times 2\) filter.
3. **Flatten Layer**:
   - Converts 2D feature maps into a 1D vector for dense layers.
4. **Dense Layers**:
   - A fully connected layer with 64 neurons and ReLU activation.
   - An output layer with 10 neurons and softmax activation for classification across 10 classes.

## Training
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy**
- Metrics: **Accuracy**
- Training Epochs: 10

The model is trained on the normalized CIFAR-10 training dataset.

## Evaluation
After training, the model evaluates accuracy on the test set and can generate a confusion matrix and classification report for further performance analysis.

## Additional Features
The code includes a commented-out section for implementing a basic Artificial Neural Network (ANN) for comparison. The ANN uses fully connected dense layers without convolutional operations.

---

### Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

### Usage
1. Clone the repository and ensure the required libraries are installed.
2. Run the script to train the CNN model on CIFAR-10.
3. Evaluate the model's accuracy and inspect metrics.

### Future Improvements
- Add data augmentation to enhance model generalization.
- Experiment with deeper architectures like ResNet.
- Perform hyperparameter tuning for optimal performance.

---

### Acknowledgments
- CIFAR-10 dataset provided by [Alex Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html).
- TensorFlow/Keras library for model implementation. 
