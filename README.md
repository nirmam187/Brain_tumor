# 🧠 Brain Tumor Classification using CNN

This project uses deep learning to classify brain MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

The model is trained using Convolutional Neural Networks (CNN) and provides a web interface using **Streamlit** to allow image uploads or URL-based prediction.

---

## 📂 Dataset

**Source**: [Kaggle - Brain Tumor MRI Dataset by MasoudNickParvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset consists of:
- 4 categories:
  - `glioma_tumor`
  - `meningioma_tumor`
  - `pituitary_tumor`
  - `no_tumor`
- Structured in `Training/` and `Testing/` folders.

---

## 📦 Project Structure
brain_tumor_classifier/
├── brain_tumor_classifier.h5 # Trained Keras CNN model
├── app.py # Streamlit web app
├── utils/ # (Optional) Preprocessing scripts
├── notebook/ # Jupyter notebooks for training and EDA
├── README.md # Project README
└── requirements.txt # Python dependencies


---

## 🚀 Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU and MaxPooling
- Dropout for regularization
- 3 Dense layers
- Final softmax layer for classification

**Input size**: 224x224x3  
**Preprocessing**: Normalized pixel values (scaled to [0, 1])

---

## 🎯 Performance (on test set)

- **Accuracy**: ~96.2%
- **Precision**: ~96.3%
- **Recall**: ~96.2%
- **Loss**: ~0.14

---

## 🖼️ Streamlit Web App

Run the app locally:

```bash
streamlit run app.py

