# Expert System for Respiratory Disease Detection from Breathing Sounds

## Project Overview

This project implements an expert system for detecting respiratory diseases from breathing sounds using machine learning and deep learning techniques. The system can classify audio recordings into five categories: **Bronchial**, **COPD**, **Healthy**, **Pneumonia**, and **Asthma**.

### Key Features

- **Multi-Model Approach**: Implements four different models (SVM, KNN, CNN, LSTM) for comprehensive comparison
- **Feature Extraction**: Extracts multiple audio features including MFCCs, spectral features, and time-domain features
- **Data Augmentation**: Applies gain scaling, time shifting, and noise addition to increase dataset diversity
- **Preprocessing**: Bandpass filtering (20-2000 Hz) to focus on lung sound frequencies
- **Comprehensive Evaluation**: Generates detailed performance metrics and visualizations for each research question
- **GUI Application**: Interactive graphical interface for real-time audio classification
- **Research Question Analysis**: Addresses 5 research questions with figures and tables

### Research Questions Addressed

1. **RQ1**: How accurately can the proposed expert system detect different types of respiratory diseases?
2. **RQ2**: Which features are most effective for respiratory disease classification?
3. **RQ3**: How does the hybrid (rule-based + machine learning) model compare to purely data-driven approaches?
4. **RQ4**: What preprocessing methods most effectively mitigate noise and variability?
5. **RQ5**: Can the system detect early-stage respiratory diseases, and what are the potential implications?

---

## Dataset Source

### Dataset Information

- **Dataset Name**: Asthma Detection Dataset Version 2
- **Source**: The dataset contains respiratory sound recordings organized by disease type
- **Structure**: 
  ```
  Dataset/
  ├── Bronchial/
  ├── copd/
  ├── healthy/
  ├── pneumonia/
  └── asthma/
  ```

### Dataset Characteristics

- **Total Samples**: ~1,211 audio files
- **Classes**: 5 (Bronchial, COPD, Healthy, Pneumonia, Asthma)
- **Format**: WAV files
- **Sampling Rate**: Variable (resampled to 22,050 Hz)
- **Audio Duration**: Variable lengths

### Dataset Path Configuration

Update the `FOLDER_PATH` variable in the notebook to point to your dataset location:

```python
FOLDER_PATH = r"path/to/your/Asthma Detection Dataset Version 2"
```

**Note**: The dataset should be organized with subdirectories named exactly as the class names: `Bronchial`, `copd`, `healthy`, `pneumonia`, `asthma`.

---

## Model Architecture

### 1. Support Vector Machine (SVM)

- **Type**: Traditional Machine Learning
- **Kernel**: RBF (Radial Basis Function)
- **Features**: Extracted audio features (MFCCs, spectral, time-domain)
- **Preprocessing**: StandardScaler normalization
- **Use Case**: Baseline model with good interpretability

### 2. K-Nearest Neighbors (KNN)

- **Type**: Traditional Machine Learning
- **Neighbors**: 5
- **Features**: Extracted audio features
- **Preprocessing**: StandardScaler normalization
- **Use Case**: Simple, interpretable model

### 3. Convolutional Neural Network (CNN)

- **Type**: Deep Learning
- **Input**: Mel Spectrograms (2D image-like representation)
- **Architecture**:
  - Conv2D layers with MaxPooling2D
  - BatchNormalization
  - Dropout for regularization
  - Dense layers for classification
- **Optimizer**: Adam
- **Use Case**: Captures spatial patterns in spectrograms

### 4. Long Short-Term Memory (LSTM)

- **Type**: Deep Learning (Recurrent Neural Network)
- **Input**: MFCC sequences (temporal sequences)
- **Architecture**:
  - Bidirectional LSTM layers
  - Dense layers for classification
  - Dropout for regularization
- **Optimizer**: Adam
- **Use Case**: Captures temporal dependencies in audio sequences

### Model Comparison

| Model | Type | Interpretability | Accuracy | Best For |
|-------|------|------------------|----------|----------|
| SVM | Traditional ML | High | Moderate | Baseline, interpretable |
| KNN | Traditional ML | High | Moderate | Simple, interpretable |
| CNN | Deep Learning | Low | High | Spatial patterns |
| LSTM | Deep Learning | Low | High | Temporal patterns |

---

## Instructions to Reproduce Results

### Prerequisites

#### 1. Install Required Python Packages

```bash
pip install numpy pandas matplotlib librosa scipy scikit-learn tensorflow openpyxl seaborn
```

Or use the requirements file (if available):

```bash
pip install -r requirements.txt
```

#### 2. Dataset Setup

1. Download the "Asthma Detection Dataset Version 2" dataset
2. Organize the dataset with the following structure:
   ```
   Asthma Detection Dataset Version 2/
   ├── Bronchial/
   │   ├── file1.wav
   │   ├── file2.wav
   │   └── ...
   ├── copd/
   ├── healthy/
   ├── pneumonia/
   └── asthma/
   ```
3. Update the `FOLDER_PATH` variable in Cell 2 of the notebook:
   ```python
   FOLDER_PATH = r"path/to/your/Asthma Detection Dataset Version 2"
   ```

### Step-by-Step Execution

#### Option 1: Run Full Workflow (Recommended)

1. **Open the Jupyter Notebook**: `P-R.ipynb`

2. **Run All Cells in Order**:
   - Cell 0: Project information
   - Cell 1: Import libraries
   - Cell 2: Configuration and utility functions
   - Cell 3-16: All function definitions
   - Cell 17: Visualization helper functions
   - Cell 18: Enhanced output function with visualizations

3. **Execute the Main Workflow**:
   ```python
   # At the end of the notebook, run:
   all_results, models, feature_importance = run_full_workflow()
   ```

   This will:
   - Load and preprocess all audio files
   - Extract features
   - Train all 4 models
   - Generate visualizations and save results

#### Option 2: Run Step by Step

1. **Step 1: Load Data**
   ```python
   audio_data, labels = step1_read_data(FOLDER_PATH, CLASSES, TARGET_SR)
   ```

2. **Step 2: Preprocessing**
   ```python
   data augmentation = increasing the size of the dataset
   filtered_data = step2_preprocessing(audio_data, augmented_data)
   ```

3. **Step 3: Feature Extraction**
   ```python
   features, feature_labels, feature_names = step3_feature_extraction(filtered_data)
   ```

4. **Step 4: Model Training**
   ```python
   all_results, models, feature_importance = step4_model_training(
       features, feature_labels, feature_names, filtered_data, 
       epochs=50, batch_size=32
   )
   ```

5. **Step 5: Generate Results with Visualizations**
   ```python
   step5_output_results_with_visualizations(
       all_results, feature_importance, models, audio_data, labels
   )
   ```

### Output Files

After running the workflow, the following files will be generated:

#### Model Files (in `models/` directory)
- `svm_model.pkl` - Trained SVM model
- `knn_model.pkl` - Trained KNN model
- `cnn_model.h5` - Trained CNN model
- `lstm_model.h5` - Trained LSTM model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `comprehensive_results.pkl` - All results summary

#### Research Question Results (in `research_question_results/` directory)

**Figures (PDF format):**
- `RQ1_Fig1.pdf` - Model Performance Comparison
- `RQ1_Fig2.pdf` - Confusion Matrices
- `RQ2_Fig1.pdf` - Feature Importance Bar Chart
- `RQ2_Fig2.pdf` - Feature Category Comparison
- `RQ3_Fig1.pdf` - Interpretability Comparison
- `RQ4_Fig1.pdf` - Preprocessing Methods Comparison
- `RQ5_Fig1.pdf` - Early-stage Detection Analysis

**Tables (Excel format):**
- `RQ1_Table1.xlsx` - Model Performance Metrics
- `RQ1_Table2.xlsx` - Confusion Matrices (all models)
- `RQ2_Table1.xlsx` - Top 20 Feature Importance
- `RQ2_Table2.xlsx` - Feature Category Summary
- `RQ3_Table1.xlsx` - Model Interpretability Comparison
- `RQ4_Table1.xlsx` - Preprocessing Methods Comparison
- `RQ5_Table1.xlsx` - Sensitivity (Recall) by Disease Type

### Running the GUI Application

To use the interactive GUI for real-time classification:

```python
# Run in terminal or notebook
python P-R.ipynb --gui

# Or in notebook:
run_gui()
```

The GUI allows you to:
- Load audio files for classification
- View predictions from all models
- See confidence scores
- Visualize audio waveforms and spectrograms

### Expected Runtime

- **Data Loading**: ~1-2 minutes (depends on dataset size)
- **Preprocessing**: ~2-5 minutes
- **Feature Extraction**: ~5-10 minutes
- **Model Training**:
  - SVM: ~1-2 minutes
  - KNN: ~1 minute
  - CNN: ~10-30 minutes (depends on epochs)
  - LSTM: ~10-30 minutes (depends on epochs)
- **Total**: ~30-60 minutes for full workflow

### Troubleshooting

#### Common Issues

1. **FileNotFoundError**: Update `FOLDER_PATH` to correct dataset location
2. **Memory Error**: Reduce batch size or number of epochs
3. **Import Error**: Install missing packages with `pip install <package_name>`
4. **CUDA/GPU Issues**: TensorFlow will automatically use CPU if GPU is unavailable

#### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: ~2GB for dataset + models
- **Python**: 3.7 or higher
- **OS**: Windows, macOS, or Linux

---

## Project Structure

```
PR-Final/
├── P-R.ipynb                          # Main Jupyter notebook
├── README.md                           # This file
├── models/                            # Trained models (generated)
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── cnn_model.h5
│   ├── lstm_model.h5
│   └── ...
└── research_question_results/         # Results (generated)
    ├── RQ1_Fig1.pdf
    ├── RQ1_Table1.xlsx
    └── ...
```

---

## Group Information

- **Group Number**: WS25-PR3
- **Project Title**: Expert System for Respiratory Disease Detection from Breathing Sounds
- **Group Members**:
  - Cherry Blossom Garcia - Student 1
  - Regina Aileen Alegrid - Student 2
  - Aaron Isaac - Student 3

---