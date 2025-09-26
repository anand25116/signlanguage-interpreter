# 🖐️ Sign Language Interpreter

This project focuses on building a **Sign Language Interpreter** using deep learning techniques to recognize the **American Sign Language (ASL) alphabet**.  
It explores both **Convolutional Neural Networks (CNNs)** and **Spiking Neural Networks (SNNs)**, with the goal of comparing their performance on static hand gesture classification.

---

## 📌 Project Overview
- Recognizes **ASL alphabet signs** from image datasets.
- Trained a **CNN model** for baseline classification.
- Currently working on an **SNN model** inspired by brain-like computing.
- Final step will be **performance comparison (accuracy, efficiency, energy usage)** between CNNs and SNNs.

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Frameworks & Libraries:** TensorFlow / PyTorch, Keras, Nengo (for SNN), OpenCV, NumPy, Matplotlib  
- **Dataset:** ASL Alphabet Dataset  

---


## 🚀 How to Run
1. Clone this repository  
```bash
git clone https://github.com/anand25116/sign-language-interpreter.git
cd sign-language-interpreter
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Train CNN model
```bash
python src/train_cnn.py
```
4. Train SNN model (in progress)
```bash
python src/train_snn.py
```
## 📊 Current Progress

✅ CNN Model trained on ASL alphabet dataset.

🔄 SNN Model under development (using spiking neuron frameworks).

📈 Plan to compare:
- Accuracy
- Training time
- Inference speed
- Energy efficiency

## 🔮 Future Work

- Real-time hand gesture detection via webcam (OpenCV).

- Extend to word-level sign recognition.


- Deploy as a web/mobile app.
