# ♻️ Smart Waste Classifier
### *AI & Deep Learning Project by Shubham Vrushabhanath Patil*

---

## 📘 Project Overview

This project presents a **Smart Waste Classifier**, an AI-powered system that classifies waste images into **Organic** or **Recyclable**.  
A Convolutional Neural Network (CNN) model was trained and optimized using **TensorFlow Lite (TFLite)** for fast inference.  
A simple and user-friendly **Streamlit** web application is developed for real-time waste classification.

---

## 🧠 Business Problem

Improper waste segregation leads to:
- Contaminated recyclables  
- Faster landfill growth  
- Increased processing cost  
- Environmental pollution  

Manual waste sorting is unsafe, slow, and inaccurate.  
A smart AI-based classifier helps automate segregation and improve recycling efficiency.

---

## 🎯 Objective

The main objectives of this project are:
- To build a **Deep Learning model** that classifies waste as Organic or Recyclable  
- To deploy the model using a **Streamlit web app**  
- To provide **instant predictions** with confidence scores  
- To offer **eco-friendly disposal tips**  
- To support **smart city waste management systems**

---

## 📂 Dataset Information

**Dataset Source:** Kaggle – Waste Classification Data  
🔗 https://www.kaggle.com/datasets/techsash/waste-classification-data

Categories included:
- Organic Waste  
- Recyclable Waste  

---

## 📚 Data Dictionary

| Field | Description |
|-------|-------------|
| Image | Waste image file |
| Label | Organic / Recyclable |
| Pixel Matrix | Resized 224×224 image converted to array |
| Softmax Score | Model confidence |
| Prediction | Final waste classification |

---

## 📈 Key Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **AI-based classification** | Organic vs Recyclable |
| 2 | **Real-time inference** | Lightweight TensorFlow Lite model |
| 3 | **Confidence scores** | Shows probability % |
| 4 | **Eco tips** | Compost or recycle suggestions |
| 5 | **Did You Know facts** | Waste-management awareness |
| 6 | **Streamlit UI** | Clean, responsive, simple |
| 7 | **Fast processing** | Optimized `.tflite` model |

---

## ⚙️ Tools & Technologies Used

- Python  
- TensorFlow / TensorFlow Lite  
- Keras  
- NumPy  
- OpenCV  
- PIL  
- Streamlit  
- Jupyter Notebook  

---

## 🧮 Workflow & Steps

### 1️⃣ Data Preparation  
- Loaded Kaggle dataset  
- Resized + normalized images  
- Applied data augmentation  

### 2️⃣ Model Training  
- Built a CNN architecture  
- Trained with softmax classifier  
- Evaluated using accuracy and confusion matrix  

### 3️⃣ Model Conversion  
- Converted model to `.tflite`  
- Reduced size and improved speed  

### 4️⃣ App Deployment  
- Integrated model into Streamlit  
- Added image upload feature  
- Displayed predictions, scores, tips, and facts  

---
## 🗂️ Folder Structure
Smart-Waste-Classifier/
│
├── app.py                     # Streamlit application
├── waste_classifier_v2.tflite # TFLite model file
├── requirements.txt           # Required Python packages
├── AIML.ipynb                 # Training notebook
├── screenshots/               # App screenshots
│   ├── home.png
│   ├── organic.png
│   └── recyclable.png
└── README.md                  # Documentation


---

## 🖼️ Dashboard Preview

![Home Screen]("![Home Screen](https://raw.githubusercontent.com/Shubhamsp30/Smart-Waste-Classifier/main/Home%20Page.png)
")
![Organic Waste Prediction]("C:\Users\SHUBHAM PATIL\OneDrive\Pictures\Screenshots\Screenshot 2025-10-29 003636.png")
![Recyclable Waste Prediction]("C:\Users\SHUBHAM PATIL\OneDrive\Pictures\Screenshots\Screenshot 2025-10-29 004118.png")

---

## 💡 Key Insights

- Organic waste classification is easier due to consistent color & texture  
- Recyclable objects vary widely, requiring robust training  
- Data augmentation increases accuracy and stability  
- TensorFlow Lite significantly improves execution speed  
- Streamlit offers a smooth and interactive user experience  

---

## 🚀 Future Enhancements

- Add more classes: Metal, Glass, Paper, E-Waste  
- Build a mobile app using TensorFlow Lite  
- Integrate with IoT-based Smart Bins  
- Enable real-time detection using a camera  
- Upgrade to EfficientNet or MobileNet architectures  

---

## 👨‍💻 Author

**Shubham Vrushabhanath Patil**  
B.Tech – Electronics & Telecommunication Engineering  
AI/ML & Data Science Enthusiast  

🔗 GitHub: https://github.com/Shubhamsp30  
🔗 LinkedIn: https://www.linkedin.com/in/shubhamsp30  

---

## ⭐ Contribute

If you liked this project:
- ⭐ Star this repository  
- 🍴 Fork it  
- 🛠️ Improve it  
- 💬 Share feedback  

---

## 🏷️ Tags

#AI #DeepLearning #TensorFlow #WasteManagement #Streamlit  
#ML #ImageClassification #CNN #Sustainability #TFLite


