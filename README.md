# Zero-to-Nine: Handwritten Digit Recognition

**Zero-to-Nine** is a machine learning project designed to recognize handwritten digits (0-9) using the MNIST dataset. The project leverages OpenCV (`cv2`) for image preprocessing and implements a fully connected neural network (FCNN) in TensorFlow/Keras. This serves as an introductory project for understanding image classification.

---

## **Project Overview**

- **Purpose**: To classify handwritten digits into categories (0 through 9) using a machine learning model.
- **Dataset**: The [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which consists of 28x28 grayscale images of handwritten digits.
- **Model Architecture**:
  - Fully Connected Neural Network (FCNN) with three dense layers:
    - **Input Layer**: 256 neurons with ReLU activation.
    - **Hidden Layer**: 64 neurons with ReLU activation.
    - **Output Layer**: 10 neurons (one for each digit class) with Softmax activation.
  - **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.001.
  - **Loss Function**: Categorical Crossentropy.
- **Libraries Used**:
  - **TensorFlow/Keras**: For building and training the neural network.
  - **OpenCV (`cv2`)**: For preprocessing images into a format suitable for model input.
  - **NumPy**: For numerical computations.
  - **Matplotlib**: For visualizing training progress and predictions.

---

## **Project Workflow**

1. **Image Preprocessing**:
   - Images are resized to 28x28 pixels and normalized to values between 0 and 1 using OpenCV (`cv2`).
   - The processed images are flattened into vectors of size 784 (28x28) before being fed into the model.

2. **Model Training**:
   - The model is trained on the MNIST dataset for 15 epochs with a batch size of 32.
   - During training, weights are adjusted to minimize the categorical crossentropy loss using the SGD optimizer.

3. **Model Evaluation**:
   - After training, the model is tested on unseen data from the MNIST test set to assess its classification accuracy.

4. **Digit Prediction**:
   - The trained model can predict the digit represented by a new 28x28 grayscale image.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/zero-to-nine.git
   cd zero-to-nine
2. Install the required dependencies:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib
3. Ensure the MNIST dataset is available (downloaded automatically in the training script).

---

## **Acknowledgments**

This project uses the MNIST dataset, made available by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
