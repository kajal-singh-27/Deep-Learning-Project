🚀 Fighter Jet Image Classification Using Deep Learning
👩‍🎓 Student: Kajal
Course: Professional Certificate in Data Science
Platform: Newton School

🛠 Tools & Libraries Used:
Google Colab

TensorFlow / Keras

OpenCV

Matplotlib & Seaborn

📌 Project Goal:
To build a deep learning model that can automatically classify different types of fighter jets using images. This kind of automation is important for military surveillance and real-time threat detection.

❓ Problem Statement:
Create an AI model that can recognize and classify images of fighter jets into one of several categories, with good performance even on unseen data.

📂 Dataset Overview:
Source: Fighter Planes Dataset

Categories:

V22

Tu160

T50

RQ4

J10

Images: ~800 resized to 224x224 pixels

Split: 80% for training, 20% for testing

Preprocessing:

Image normalization to [0, 1]

One-hot encoding for labels

Used train_test_split

🧠 Model Architecture:
Type: Custom Convolutional Neural Network (CNN)

Layers:

Conv2D → ReLU → MaxPooling → Dropout

Flatten → Dense → Softmax

Loss Function: Categorical Crossentropy

Optimizer: Adam

Metric: Accuracy

📊 Training & Results:
Epoch	Training Accuracy	Training Loss	Validation Accuracy	Validation Loss
1	14.79%	8.8937	12.50%	1.7161
2	21.58%	1.6172	12.50%	1.6062
3	35.16%	1.5846	15.00%	1.6023
4	39.75%	1.5006	37.50%	1.5671
5	50.93%	1.4350	17.50%	1.6410
6	61.41%	1.1532	45.00%	1.6217
7	63.96%	0.9588	27.50%	1.7740
8	79.36%	0.6530	37.50%	1.9433
9	87.10%	0.4733	40.00%	2.1448
10	88.73%	0.3434	45.00%	2.4348

⚠️ Observation:
The training accuracy improved steadily, reaching nearly 89%, but the validation accuracy peaked at 45%, showing clear signs of overfitting.

📈 Visuals:
Accuracy & Loss vs Epoch graphs

Confusion matrix

Classification report per class

💡 Key Takeaways:
The model performs well on training data but not on new data.

Still, it lays a good foundation for jet image classification using AI.

It can be extended for real-time drone or satellite-based surveillance systems.

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

dataset/ – Jet images

Final Priject Report – Predictions & reports

README.md – Project overview

✅ Final Thoughts:
This project shows how deep learning can be used to classify fighter jets automatically. While the current model struggles with generalization, with more data and enhancements, it can evolve into a robust solution for defense and surveillance applications.
