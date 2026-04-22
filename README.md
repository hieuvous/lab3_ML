# Fashion-MNIST Image Classification Project

This project explores various machine learning models (k-NN, SVM, Random Forest, and MLP) to classify the Fashion-MNIST dataset. We implement a full pipeline from data preprocessing to hyperparameter tuning and ablation studies.

## Team Members

- Phạm Khánh Linh - 23127083
- Võ Trung Hiếu - 23127190
- Nguyễn Thành Lợi - 23127408
- Lê Quốc Thiện - 23127481
- Phạm Quang Thịnh - 23127485

---

## Getting Started

### Data & Model Setup

To run the evaluation or ablation notebooks without retraining, please download the pre-processed data and trained models from the link below:

**[Link to Google Drive - Data & Models]**  
[https://drive.google.com/drive/folders/1KbSx2WL4_74ZZMJ-MSRK9vY-CMX1eqev?usp=sharing]

1.  Download the `data` and `models` folders.
2.  Place both folders directly into the **root folder** (`src/`) of this project.
3.  If you wish to re-run the pipeline from scratch:
    - Place the raw `fashion-mnist_train.csv` and `fashion-mnist_test.csv` into the `data/` folder.
    - Run `preprocessing/preprocessing.ipynb` to generate the `.npz` file.

---

## 📂 Project Structure

```text
src/
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

## 1. Data Description
Fashion-MNIST is a dataset consisting of Zalando’s article images. It serves as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.
*   **Total Samples:** 70,000.
*   **Image Dimensions:** 28x28 pixels (Grayscale).
*   **Feature Count:** 784 (Flattened).
*   **Classes:** 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

## 2. Exploratory Data Analysis (EDA)
During the EDA phase, we observed the following:
*   **Class Balance:** The dataset is perfectly balanced, with each class containing exactly 7,000 images.
*   **Pixel Intensity:** The raw data ranges from 0 to 255. Most pixels are 0 (background), creating a bimodal distribution where the second peak represents the foreground object.
*   **Visual Similarity:** High overlap exists between "Shirt," "T-shirt," and "Coat," which suggests linear models might struggle to find clear boundaries.

## 3. Data Preprocessing & Experimental Setup

### 3.1 Experimental Setup
To ensure reproducibility, all benchmarks were executed under the following environment:
*   **Processor:** AMD Ryzen 5 7535HS (6 Cores, 12 Threads).
*   **Memory:** 16GB DDR5 RAM.
*   **Graphics:** NVIDIA GeForce RTX 4050 Laptop GPU.
*   **Software Stack:** Python 3.11, Scikit-learn, NumPy, Pandas, Matplotlib.
*   **Random Seed:** 42.

### 3.2 Preprocessing Pipeline
1.  **Normalization:** Pixel values were divided by 255 to scale them to the $[0, 1]$ range.
2.  **Standardization:** Applied Z-score standardization ($z = \frac{x - \mu}{\sigma}$) using `StandardScaler` to center the data at mean 0 and unit variance.
3.  **Data Partitioning:** We performed a **stratified split** to ensure subsets were representative:
    *   **Training Set:** 50,000 samples.
    *   **Validation Set:** 10,000 samples.
    *   **Test Set:** 10,000 samples.
4.  **Storage:** Finalized arrays were saved as a compressed `.npz` file for shared access.

## 4. Classification Models: Idea and Implementation

### 4.1 k-Nearest Neighbors (k-NN)
We performed a grid search on $k$ values and distance metrics. While Manhattan distance ($p=1$) was slightly more accurate, Euclidean distance ($p=2$) was significantly faster. We selected $k=5$ and $p=2$ with distance-based weighting for our final implementation.

### 4.2 Support Vector Machine (SVM)
We compared Linear and RBF kernels. The **RBF kernel** significantly outperformed the Linear kernel, achieving the highest overall accuracy. We used $C=15$ and enabled probability estimates for ROC-AUC analysis.

### 4.3 Random Forest
An ensemble model using 200 trees. It demonstrated high stability and was the most memory-efficient model during inference, showing a negligible memory spike compared to the others.

### 4.4 Multi-layer Perceptron (MLP)
We implemented a feed-forward neural network with a (256, 128) hidden layer architecture. It utilized ReLU activation and the Adam optimizer. This model provided the best trade-off between high accuracy and real-time inference speed.

## 5. Evaluation
The models were evaluated based on three main pillars:

1.  **Predictive Power:** SVM achieved the best Accuracy (0.90) and Macro F1-score (0.90).
2.  **Inference Speed (Latency):** MLP was the fastest (0.0035 ms/sample), while SVM was the slowest due to the complexity of support vectors.
3.  **Resource Usage:** Random Forest was the most memory-stable, while SVM and k-NN showed significant memory spikes.
4.  **Final Decision:** MLP model as it was fast and accurate enough with controllable memory usage.

## 6. Ablation Study
We conducted an ablation study on the MLP model to identify the impact of specific components on performance:
*   **No Standardization:** Accuracy dropped slightly, proving that centering the data helps the optimizer converge faster.
*   **No Early Stopping:** Allowed the model to reach a slightly higher accuracy but at the cost of significantly longer training time and potential overfitting.
*   **No Hidden Layer:** Reducing the MLP to a single-layer perceptron (Linear) caused the most significant drop in accuracy (~4%), proving that non-linear hidden representations are essential for this dataset.

## 7. Conclusions and Insights
*   **Performance:** Non-linear models (SVM-RBF and MLP) are necessary for Fashion-MNIST; linear boundaries are insufficient for distinguishing complex clothing textures.
*   **Practicality:** While SVM is slightly more accurate, the MLP is preferred for production deployment because it is thousands of times faster during inference.
*   **Bottleneck:** All models share the same bottleneck: the "Shirt" class. Improvement in future work should focus on feature extraction methods (like CNNs) to capture collar and button details.