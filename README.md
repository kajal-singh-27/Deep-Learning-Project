# Fighter Jet Image Classification Using Deep Learning

**Student:** Kajal
**Course:** Professional Certificate in Data Science
**Platform:** Newton School

---

## Project Overview

This project involves building a deep learning model for the automatic classification of fighter jet images. The aim is to assist military and defense operations through automated identification from drone footage, satellite images, or surveillance systems.

---

## Tools and Technologies

* **Development Environment:** Google Colab
* **Frameworks:** TensorFlow, Keras
* **Libraries:** OpenCV, Matplotlib, Seaborn

---

## Objective

Develop a convolutional neural network (CNN) that can classify images of fighter jets into one of five categories, helping reduce manual efforts and improve identification accuracy in real-time defense scenarios.

---

## Problem Statement

Build a machine learning model that:

* Classifies fighter jet images into categories: V22, Tu160, T50, RQ4, J10
* Generalizes well on unseen data
* Handles variations in image quality and jet appearance

---

## Dataset Description

* **Source:** Fighter Planes Dataset
* **Classes:** V22, Tu160, T50, RQ4, J10
* **Total Images:** \~800
* **Image Size:** 224x224 pixels
* **Train/Test Split:** 80% training, 20% testing

### Preprocessing Steps

* Image normalization (scaling pixel values to \[0, 1])
* Label encoding using one-hot encoding
* Splitting dataset using `train_test_split`

---

## Model Architecture

A custom Convolutional Neural Network (CNN) was used with the following layers:

| Layer (Type)             | Output Shape         | Parameters |
| ------------------------ | -------------------- | ---------- |
| Conv2D                   | (None, 222, 222, 32) | 896        |
| MaxPooling2D             | (None, 111, 111, 32) | 0          |
| Conv2D                   | (None, 109, 109, 64) | 18,496     |
| MaxPooling2D             | (None, 54, 54, 64)   | 0          |
| Conv2D                   | (None, 52, 52, 128)  | 73,856     |
| MaxPooling2D             | (None, 26, 26, 128)  | 0          |
| Flatten                  | (None, 86528)        | 0          |
| Dense (512 units)        | (None, 512)          | 44,302,848 |
| Dense (5 output classes) | (None, 5)            | 2,565      |

* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **Evaluation Metric:** Accuracy

---

## Training Results

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| ----- | ----------------- | ------------- | ------------------- | --------------- |
| 1     | 14.79%            | 8.8937        | 12.50%              | 1.7161          |
| 2     | 21.58%            | 1.6172        | 12.50%              | 1.6062          |
| 3     | 35.16%            | 1.5846        | 15.00%              | 1.6023          |
| 4     | 39.75%            | 1.5006        | 37.50%              | 1.5671          |
| 5     | 50.93%            | 1.4350        | 17.50%              | 1.6410          |
| 6     | 61.41%            | 1.1532        | 45.00%              | 1.6217          |
| 7     | 63.96%            | 0.9588        | 27.50%              | 1.7740          |
| 8     | 79.36%            | 0.6530        | 37.50%              | 1.9433          |
| 9     | 87.10%            | 0.4733        | 40.00%              | 2.1448          |
| 10    | 88.73%            | 0.3434        | 45.00%              | 2.4348          |

### Observations

* The training accuracy improved consistently to \~89%
* Validation accuracy peaked at 45%, indicating overfitting

---

## Visualizations

* Accuracy vs Epoch
* Loss vs Epoch
* Confusion Matrix
* Per-Class Classification Report

---

## Insights and Applications

* Strong performance on training data confirms learning capability
* The model can be a baseline for AI-based jet classification in defense
* Can be integrated into real-time drone or satellite surveillance systems

---

## Limitations

* Dataset size is small, limiting generalization
* Visual similarity among jets leads to misclassifications
* Variation in image quality and backgrounds affects accuracy

---

## Recommendations for Improvement

* Implement data augmentation to enrich dataset
* Apply transfer learning with models like ResNet50 or VGG16
* Use regularization (Dropout, L2 penalties)
* Perform hyperparameter tuning
* Add early stopping during training

---

## Project Files

* `Fighter_Jet_Image_Classification.ipynb` – Model training code
* `model.h5` – Trained model file
* `dataset/` – Image dataset folder
* `Final Project Report/` – Evaluation results and predictions
* `README.md` – Project documentation

---

## Conclusion

This project demonstrates the use of deep learning for fighter jet classification. While the current model faces generalization challenges, it provides a strong foundation. With further improvements, the system has potential for real-world defense and surveillance deployments.



⚠️ Limitations:
Small dataset → limits generalization

Jets have similar appearance → leads to misclassification

Image quality/background variations affect accuracy



✅ Recommendations:
Add data augmentation to increase dataset variety

Use transfer learning (ResNet, VGG) for better generalization

Introduce regularization techniques (Dropout, L2)

Perform hyperparameter tuning

Use early stopping to prevent overfitting




📁 Project Files:
 Fighter_Jet_Image_Classification.ipynb – Model training code

model.h5 – Trained model

dataset– Jet images

Final Priject Report – Predictions & reports

README.md – Project overview



✅ Final Thoughts:
This project shows how deep learning can be used to classify fighter jets automatically. While the current model struggles with generalization, with more data and enhancements, it can evolve into a robust solution for defense and surveillance applications.
