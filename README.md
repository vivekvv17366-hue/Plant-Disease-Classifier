
# Plant Disease Classification Using Deep Learning

### AI-Powered Tomato Leaf Disease Detection | MobileNetV2 | TensorFlow | Streamlit

## 1. Project Overview

Plant Disease Classification is a Computer Vision and Deep Learning project designed to identify tomato leaf diseases from digital images.

The application leverages Transfer Learning with the MobileNetV2 architecture to classify uploaded leaf images into four categories. It integrates a trained Deep Learning model with an interactive Streamlit interface, enabling users to obtain predictions and visualize classification probabilities.

The primary objective is to demonstrate the practical implementation of Deep Learning, image preprocessing, model evaluation, and deployment in an AI-powered application.

---

## 2. Key Features

- **Image Classification:** Identifies tomato leaf conditions from uploaded images.
- **Transfer Learning:** Utilizes MobileNetV2 with pretrained ImageNet weights.
- **Image Preprocessing:** Resizes and prepares input images for model inference.
- **Prediction Confidence:** Displays the model's predicted probability.
- **Probability Visualization:** Presents classification probabilities across supported categories.
- **Interactive Web Application:** Provides a user-friendly interface developed using Streamlit.
- **Model Evaluation:** Includes accuracy, precision, recall, F1-score, and confusion matrix analysis.

---

## 3. Supported Classification Categories

The model is designed to classify tomato leaf images into the following four categories:

| Category | Description |
|---|---|
| Tomato Bacterial Spot | Bacterial leaf disease |
| Tomato Early Blight | Fungal leaf disease |
| Tomato Late Blight | Fungal leaf disease |
| Tomato Healthy | Healthy tomato leaf |

---

## 4. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep Learning framework |
| MobileNetV2 | Transfer Learning architecture |
| NumPy | Numerical computations |
| Pandas | Data handling |
| Matplotlib | Model visualization |
| Scikit-learn | Model evaluation |
| Pillow | Image processing |
| Streamlit | Web application development |
| Git & GitHub | Version control and project management |

---

## 5. Model Architecture

The project implements Transfer Learning using MobileNetV2, a lightweight Convolutional Neural Network architecture pretrained on the ImageNet dataset.

### Architecture Workflow

```text
Input Leaf Image
       |
       v
Image Resizing
  (224 x 224)
       |
       v
Data Preprocessing
       |
       v
Data Augmentation
       |
       v
Pretrained MobileNetV2
       |
       v
Feature Extraction
       |
       v
Global Average Pooling
       |
       v
Dropout Layer
       |
       v
Dense Classification Layer
       |
       v
Softmax Activation
       |
       v
Disease Prediction
```

### Model Configuration

- Architecture: MobileNetV2
- Learning Approach: Transfer Learning
- Input Dimensions: 224 × 224 × 3
- Classification Type: Multi-class Classification
- Output Classes: 4
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

---

## 6. Project Structure

```text
Plant_Disease_Classification/
│
├── datasets/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_healthy/
│   └── Tomato_Late_blight/
│
├── train.py
├── evaluate.py
├── predict.py
├── app.py
│
├── class_names.json
├── plant_disease_analyser.keras
│
├── accuracy.png
├── loss.png
├── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The dataset and virtual environment are excluded from the GitHub repository.*

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or compatible version
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### Step 2: Navigate to the Project Directory

```bash
cd Plant_Disease_Classification
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Prepare the Dataset

Download the PlantVillage dataset and organize the required tomato categories inside the `datasets` directory.

Ensure that the folder names match the class names used during model training.

---

## 8. Model Training

To train the classification model, execute:

```bash
python train.py
```

The training pipeline includes:

1. Dataset loading and splitting.
2. Image resizing and preprocessing.
3. Data augmentation.
4. Transfer Learning using MobileNetV2.
5. Model compilation and training.
6. Accuracy and loss visualization.
7. Model saving.

---

## 9. Model Evaluation

Run the evaluation script:

```bash
python evaluate.py
```

The evaluation process includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These evaluation metrics help assess the model's classification performance across different disease categories.

---

## 10. Running the Application

Launch the Streamlit application using:

```bash
streamlit run app.py
```

Open the local URL provided in the terminal.

### Application Workflow

1. Launch the application.
2. Upload a tomato leaf image.
3. Preview the uploaded image.
4. Click the Analyze Plant Image button.
5. View the predicted disease category.
6. Explore the model confidence and class probability distribution.

---

## 11. Dataset Information

The project uses the PlantVillage plant disease image dataset.

Dataset Source:

https://www.kaggle.com/datasets/emmarex/plantdisease

The project utilizes selected tomato leaf categories for multi-class classification.

---

## 12. Future Enhancements

Potential improvements include:

- Expanding classification to additional plant species.
- Implementing model fine-tuning for improved performance.
- Integrating Grad-CAM for visual model interpretability.
- Adding prediction history and downloadable reports.
- Comparing multiple Deep Learning architectures.
- Deploying the application to a cloud platform.
- Evaluating performance on real-world field images.

---

## 13. Limitations and Disclaimer

This project is developed for educational and research purposes.

The model is trained on a selected image dataset and may not generalize reliably to all real-world agricultural conditions.

Predictions should not be considered professional agricultural diagnoses or used as the sole basis for treatment decisions.

---

## 14. Author

**Vivek V V**

Aspiring AI/ML Engineer

### Areas of Interest
- Machine Learning
- Deep Learning
- Computer Vision
- Generative AI

---

**Developed using Python, TensorFlow, MobileNetV2 and Streamlit.**