# Unsupervised Anomaly Detection

## Overview
This project implements **unsupervised anomaly detection** using a **Convolutional Autoencoder (CAE)** on the **CIFAR-10** dataset.  
The model is trained only on a single *normal* class (Airplane) and detects anomalies based on **reconstruction error**. Images that cannot be reconstructed well are classified as anomalies.

This approach demonstrates how deep learning models can learn normal behavior without labeled anomaly data.

---

## Key Idea
- Train the autoencoder **only on normal data** (Airplanes).
- The model learns to reconstruct normal images with low error.
- Images from other classes produce **higher reconstruction errors**.
- A statistical threshold is used to flag anomalies.

---

## Dataset
- **Dataset:** CIFAR-10  
- **Total Images:** 60,000 (32×32 RGB)
- **Classes:** 10  
- **Normal Class:** Airplane (label = 0)

### Data Split
- **Training:** Only airplane images  
- **Testing:** All CIFAR-10 classes (mixed)

---

## Model Architecture – Convolutional Autoencoder

### Encoder
- Convolutional layers with ReLU activation
- Feature extraction and compression
- Latent space dimension: **128**

### Decoder
- Linear layer + Transposed Convolutions
- Reconstructs images from latent space
- Sigmoid activation for output

### Training Configuration
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Epochs:** 10
- **Batch Size:** 64
- **Device:** CUDA (if available) / CPU

---

## Anomaly Detection Method
1. Compute reconstruction errors on training data.
2. Set threshold as **99th percentile** of training errors.
3. During testing:
   - Error ≤ Threshold → **Normal**
   - Error > Threshold → **Anomaly**

---

## Evaluation & Visualization
- Reconstruction error calculation for test samples
- Visual outputs include:
  - Original image
  - Reconstructed image
  - Error heatmap
  - Anomaly verdict

---

## Results
- Airplane images are reconstructed accurately.
- Images from other classes show higher reconstruction errors.
- The model successfully distinguishes **normal vs anomalous** samples without supervised labels.

---

## Applications
This technique can be extended to:
- Industrial fault detection
- Medical image anomaly detection
- Fraud detection
- Network intrusion detection
- Quality inspection systems

---

## Code
- Implemented using **PyTorch**
- Dataset handled via **torchvision**
- Visualization using **Matplotlib**

---

## Conclusion
This project demonstrates the effectiveness of **autoencoders for unsupervised anomaly detection**.  
By learning only normal patterns, the model identifies anomalies through reconstruction errors, making it suitable for real-world scenarios where labeled anomaly data is scarce.

---

