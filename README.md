# Zero-to-Nine: Handwritten Digit Recognition

**Zero-to-Nine** is a machine learning project designed to recognize handwritten digits (0-9) using the MNIST dataset. It employs a fully connected neural network (FCNN) implemented in TensorFlow/Keras, with image preprocessing handled by OpenCV (`cv2`). This project serves as a beginner-friendly introduction to image classification.

---

## **Project Overview**

- **Purpose**: To classify handwritten digits into categories (0 through 9) using a machine learning model.
- **Dataset**: The [MNIST dataset](http://yann.lecun.com/exdb/mnist/), a collection of 28x28 grayscale images of handwritten digits.
- **Model Architecture**:
  - Fully Connected Neural Network (FCNN) with three dense layers.
  - Activation Functions: ReLU for hidden layers, Softmax for output.
  - Optimizer: Stochastic Gradient Descent (SGD).
  - Loss Function: Categorical Crossentropy.

---

## **Project Workflow**

1. **Image Preprocessing**:
   - Images are processed using OpenCV (`cv2`) to ensure uniform size (28x28 pixels) and normalized to values between 0 and 1.
   - The input data is flattened into vectors of size 784 (28x28) before feeding into the model.

2. **Model Training**:
   - The FCNN model is trained on the MNIST dataset for 15 epochs with a batch size of 32.
   - Training optimizes model weights to minimize categorical crossentropy loss.

3. **Model Evaluation**:
   - After training, the model is tested on unseen MNIST data to evaluate its accuracy in classifying digits.

4. **Digit Prediction**:
   - Given a new 28x28 grayscale image, the model predicts the digit it represents.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/zero-to-nine.git
   cd zero-to-nine
