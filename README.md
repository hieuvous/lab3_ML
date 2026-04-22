# Fashion-MNIST Image Classification Project

This project explores various machine learning models (k-NN, SVM, Random Forest, and MLP) to classify the Fashion-MNIST dataset. We implement a full pipeline from data preprocessing to hyperparameter tuning and ablation studies.

## Team Members

- Phạm Khánh Linh - 23127083
- Võ Trung Hiếu - 23127190
- Nguyễn Thành Lợi - 23127408
- Lê Quốc Thiện - 23127481
- Phạm Quang Thịnh - 23127485

---

## 🚀 Getting Started

### Data & Model Setup

To run the evaluation or ablation notebooks without retraining, please download the pre-processed data and trained models from the link below:

**[Link to Google Drive - Data & Models]**  
[https://drive.google.com/drive/folders/1KbSx2WL4_74ZZMJ-MSRK9vY-CMX1eqev?usp=sharing]

1.  Download the `data` and `models` folders.
2.  Place both folders directly into the **root folder** (`Group 2/`) of this project.
3.  If you wish to re-run the pipeline from scratch:
    - Place the raw `fashion-mnist_train.csv` and `fashion-mnist_test.csv` into the `data/` folder.
    - Run `preprocessing/preprocessing.ipynb` to generate the `.npz` file.

---

## 📂 Project Structure

```text
Group 2/
├── ablation/
│   └── ablation_mlp.ipynb          # MLP component analysis
├── EDA/
│   └── EDA.ipynb                   # Exploratory Data Analysis
├── evaluation/
│   └── Evaluation.ipynb            # Final model comparison and metrics
├── models_ablation/                # Trained ablation models (joblib)
│   ├── mlp_ablation_baseline.joblib
│   ├── mlp_ablation_no_early_stopping.joblib
│   ├── mlp_ablation_no_hidden_layer.joblib
│   └── mlp_ablation_no_standardization.joblib
├── models_notebooks/               # Model training & tuning
│   ├── KNN.ipynb
│   ├── RF&MLP.ipynb
│   └── SVM.ipynb
├── preprocessing/
│   └── preprocessing.ipynb         # Data normalizing, splitting and scaling
├── results/
│   └── mlp_ablation_results.csv    # Exported ablation metrics
├── data/                           # (User provided) .npz and .csv files
└── models/                         # (User provided) .joblib final models
```

---

## 📝 Project Sections

### 1. Data Description

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

### 2. EDA (Exploratory Data Analysis)

_[Section to be completed by team member]_

### 3. Preprocessing

- **Normalization:** Pixel values were scaled to the [0, 1] range.
- **Standardization:** Applied Z-score standardization (StandardScaler) to center data (mean=0, std=1), which is crucial for distance-based models (KNN) and gradient-based models (MLP).
- **Data Partitioning:** Employed a stratified split to ensure class balance across subsets:
  - **Training:** 50,000 samples.
  - **Validation:** 10,000 samples.
  - **Test:** 10,000 samples.
- **Storage:** Exported data to `fashion_data_complete.npz` for consistent cross-model training.

### 4. Model Implementation & Tuning

#### Support Vector Machine (SVM)

- **Tuning:** Compared Linear vs. RBF kernels. RBF significantly outperformed Linear in high-dimensional space.
- **Optimization:** Identified $C=15$ as the optimal regularization parameter.
- **Result:** Highest overall accuracy (~90.06%).

#### k-Nearest Neighbors (KNN)

- **Tuning:** Tested $k$ values {1, 3, 5, 7, 11, 15} and distance metrics (Manhattan vs. Euclidean).
- **Optimization:** Selected $k=5$, $p=2$ (Euclidean) with distance-based weighting.
- **Trade-off:** While Manhattan ($p=1$) was slightly more accurate, Euclidean ($p=2$) was 10x faster, making it more practical for the full 60,000 sample dataset.

#### Random Forest (RF)

_[Section to be completed by team member]_

#### Multi-Layer Perceptron (MLP)

_[Section to be completed by team member]_

### 5. Evaluation

The models were evaluated based on three main pillars:

1.  **Predictive Power:** SVM achieved the best Accuracy (0.90) and Macro F1-score (0.90).
2.  **Inference Speed (Latency):** MLP was the fastest (0.0035 ms/sample), while SVM was the slowest due to the complexity of support vectors.
3.  **Resource Usage:** Random Forest was the most memory-stable, while SVM and k-NN showed significant memory spikes.
4.  **Final Decision:** MLP model as it was fast and accurate enough with controllable memory usage.

### 6. Ablation Study

Focuses on the MLP model to determine the impact of:

- Standardization.
- Early Stopping.
- Hidden Layer architecture.

_[Section to be completed by team member]_

### 7. Results & Conclusion

- **Best for Accuracy:** SVM is the top performer but comes with high computational costs.
- **Best for Production:** MLP offers the best balance, providing high accuracy with near-instantaneous inference speed.
- **Challenging Classes:** All models struggle with the "Shirt" class due to its visual similarity to "T-shirt" and "Coat" at low resolutions.
