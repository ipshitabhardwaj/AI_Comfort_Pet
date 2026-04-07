# 🌙 Emotion-Aware Digital Comfort Pet

**An Emotion-Aware Digital Companion**
Luna is a lightweight, empathetic AI companion designed to provide digital comfort. Unlike traditional chatbots, Luna uses an Ensemble Machine Learning pipeline to sense your emotional state and react with tailored support, guided breathing, and journaling prompts.

---

## 📋 Project Overview

### What It Does
-Emotion Detection: Analyzes text to detect 4 core emotions (Joy, Sadness, Anger, Fear) using a high-accuracy Ensemble model.

-Empathetic Responses: Generates context-aware, comforting messages based on your specific mood.

-Interactive Pet: Features a cat themed interface that reacts visually to your input.

-Wellness Tools: Automatically triggers breathing exercises and coping strategies when you need them most.

-Mood Tracking: Maintains a local session history with CSV export functionality.

### Key Features
✨ Ensemble ML Model - Combined SVC + Logistic Regression for robust accuracy.
⚡ Fast Inference - Real-time emotion detection in milliseconds.
🎨 Aesthetic Design - Soft pastel pink theme with glassmorphism and smooth animations.
📊 Privacy First - All data stays in your current session; no external LLM calls or data logging.
🚀 Zero-Heavyweight - No 100MB+ transformers. Pure, efficient Scikit-Learn logic.

---

## 🏗️ Project Structure

```
AI_Comfort_Pet/
│
├── app.py                  # Main Streamlit UI & Logic
├── utils.py                # NLP Pipeline & Ensemble Response Engine
├── train_model.py          # Model Training & Validation script
├── requirements.txt        # Python dependency manifest
├── README.md               # Professional documentation
│
├── models/                 # Serialized "Brain" folder
│   └── emotion_model.pkl   # Trained Ensemble Model (~2MB)
│
├── data/                   # Dataset storage
│   └── emotion.csv         # Labeled training data
│
└── cat.png                 # Core UI Visual Asset
```

---

## 📊 Dataset Details

### Source
- **Dataset**: Emotion Dataset (dair-ai/emotion)
- **Platform**: Hugging Face
- **Size**: ~2MB (lightweight!)
- **Samples**: 6000+ labeled emotion sentences

### Emotion Categories
- **Joy** (0): happiness, excitement, gratitude
- **Sadness** (1): depression, grief, disappointment
- **Anger** (2): frustration, rage, irritation
- **Fear** (3): anxiety, worry, terror
- **Surprise** (4): amazement, shock, wonder
- **Disgust** (5): revulsion, contempt, aversion
- **Neutral** (6): no strong emotion

### Download Method
The training script automatically:
1. Downloads the dataset from Hugging Face (if online)
2. Falls back to a backup synthetic dataset (if offline)
3. Saves it locally as `data/emotion.csv`
4. Cleans and prepares it for training

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- ~500MB disk space (for dependencies and model)

### Step 1: Clone or Download the Project
```bash
# Option A: If you have the files
cd emotion-comfort-pet

# Option B: Create a new directory
mkdir emotion-comfort-pet
cd emotion-comfort-pet
# (Copy all files here)
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **streamlit**: For the web UI
- **pandas**: For data handling
- **numpy**: For numerical operations
- **scikit-learn**: For the ML model
- **python-dateutil**: For timestamp handling

Total download size: ~200MB (very lightweight!)

### Step 4: Train the Model
```bash
python train_model.py
```

**What happens:**
1. Downloads the emotion dataset (or uses backup)
2. Creates TF-IDF vectorizer for text processing
3. Trains Naive Bayes classifier
4. Evaluates model performance
5. Saves model to `models/emotion_model.pkl` (~2MB)

## 🚀 Running the Application

### Start the Streamlit App
```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

```

### Open in Browser
- **Desktop**: Click the link or go to `http://localhost:8501`
- **Mobile/Network**: Use the Network URL from above
- **iPad/Tablet**: Works perfectly with responsive design

---

## 🎮 How to Use the App

### 1. **Customize Your Pet**
   - Enter a name for your comfort pet (default: "Luna")
   - The name is saved in your session

### 2. **Share Your Feelings**
   - Click in the text area "What's on your mind?"
   - Type how you're feeling (e.g., "I'm really happy today!")
   - Click the "Share with [Pet Name]" button

### 3. **Get Emotional Support**
   - Your emotion is detected and displayed with a color badge
   - The pet reacts to your emotion
   - You receive a personalized, comforting response
   - Response appears with a typing animation

### 4. **Track Your Mood**
   - Every interaction is saved in "Your Mood Journey"
   - See timestamps and emotion distribution
   - View mood trends and insights
   - Statistics show total interactions and most common mood

---

## 🧠 Model Architecture

### ML Pipeline
```
User Input (Text)
       ↓
TF-IDF Vectorization (Bigram Analysis)
       ↓
Ensemble Voting (SVC + Logistic Regression)
       ↓
Emotion Prediction + Confidence Score
       ↓
Dynamic Response Generation
```

### Why This Approach?
✅ **Fast**: Training and inference are very quick
✅ **Lightweight**: Model is only 2MB
✅ **Interpretable**: Easy to understand how decisions are made
✅ **Reliable**: Works well without large training data
✅ **No Heavy Dependencies**: Only scikit-learn needed
✅ **Scalable**: Can handle new emotions with retraining

---
## IPSHITA BHARDWAJ
### B.Tech Robotics & Artificial Intelligence (Department of Emerging Technologies)
---
https://aicomfortpet.streamlit.app/
