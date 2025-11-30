# 📦 Pneumonia Detection Using CNN & Streamlit

*A Deep Learning Web App for Chest X-Ray Diagnosis*

# LIVE LINK 
https://pneumonia-detection-eeadnxmid8p9npnmksmpd4.streamlit.app/

# Demo


https://github.com/user-attachments/assets/a356fb76-b2a0-4bfa-aed6-e4a9f7a2b4b9


---

## 📝 **Overview**

This project is an end-to-end Chest X-Ray Pneumonia Detection system built using:

* **TensorFlow / Keras** – to train the CNN model
* **Streamlit** – to deploy a simple, interactive web interface
* **ImageDataGenerator** – for augmentation and preprocessing
* **.keras Saved Model** – for inference through the web app

The app allows users to upload a chest X-ray image (JPG/PNG) and receive a predicted diagnosis:

* **NORMAL**
* **PNEUMONIA**

The model uses a custom CNN trained on the **Kaggle Chest X-Ray Pneumonia dataset** provided by Guangzhou Women and Children's Medical Center.

---

## 🚀 **Features**

### 🩻 Medical Image Diagnosis

Upload a chest X-ray and get an instant diagnosis using a trained CNN.

### 🧠 Deep Learning Pipeline

Includes preprocessing, augmentation, model architecture, training loop, and model saving.

### 🌐 Streamlit Web App

Simple and beautiful UI for interaction.

### ⚙️ Easily Deployable

Can be hosted on:

* Streamlit Cloud
* HuggingFace Spaces
* Render
* Local Machine

### 🎒 Student-Friendly

Simple .keras model + clean inference script.

---

## 📁 **Project Structure**

```
📦 pneumonia-detection
├── app.py                 # Streamlit web app
├── my_model.keras         # Saved trained CNN model
├── requirements.txt       # Dependencies
├── README.md              # Project documentation

```

---

## 🧠 **Model Architecture**

A simple yet effective CNN:

* **3 Convolution Blocks**

  * Conv2D → ReLU → MaxPooling
* **Flatten**
* **Dense(128) + ReLU**
* **Dropout(0.5)**
* **Dense(1) + Sigmoid**

Optimized using:

* **Adam optimizer**
* **Binary Cross-Entropy loss**

---

## 📊 **Dataset**

**Source:**
*Kaggle – Chest X-Ray Images (Pneumonia)*
Originally by *Guangzhou Women and Children’s Medical Center*.

**Classes:**

* NORMAL
* PNEUMONIA

**Data split:**

```
train/
    NORMAL/
    PNEUMONIA/
val/
    NORMAL/
    PNEUMONIA/
test/ (optional)
```

---

## 🛠️ **Installation Guide**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/valive_421/pneumonia-detection.git
cd pneumonia-detection
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv env
```

Activate it:

Windows:

```bash
env\Scripts\activate
```

Linux/Mac:

```bash
source env/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ **How to Run the Streamlit App**

Inside the project folder:

```bash
streamlit run app.py
```

Streamlit will open the browser automatically, or you can visit:

📍 **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 **How Diagnosis Works**

1. Upload a chest X-ray image
2. Image gets resized → (128×128)
3. Normalized → `/255`
4. Converted to an array
5. Fed to the CNN model
6. Output:

   * `0.0–0.49` → **NORMAL**
   * `0.50–1.0` → **PNEUMONIA**

---

## 🧵 **Code Snippets**

### **Prediction Helper**

```python
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)
```

### **Run Prediction**

```python
prob = model.predict(input_data)[0][0]
diagnosis = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
```

---

## 🔮 **Future Enhancements**

* Add **Grad-CAM heatmaps** for visual explainability
* Add **batch prediction**
* Use a **Transfer Learning model** (MobileNetV2, EfficientNet)
* Add metrics dashboard + training charts

---

## 🏥 **Medical Disclaimer**

> This project is for **educational and research purposes only**.
> It is **NOT** approved for clinical or diagnostic use.
> Always consult certified medical professionals for health decisions.

---

## 👨‍💻 **Author**

**vaibhav avhad**
Machine Learning & AI Enthusiast
Feel free to connect or contribute!

---

## ⭐ **Support the Project**

If you found this useful:

❤️ Star the repo
🔱 Fork it
👨‍💻 Open issues and PRs

---

