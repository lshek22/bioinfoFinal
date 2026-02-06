# bioinfoFinal  
**LUAD vs Non-LUAD Classification Using Gene Expression Data**

## Abstract

Accurate classification of cancer subtypes using gene expression data is a central task in computational biology and precision medicine. In this project, I developed a machine learning pipeline to distinguish lung adenocarcinoma (LUAD) samples from non-LUAD cancer samples using RNA-seq gene expression profiles. The analysis is inspired by the approach described by Khan et al. (2001), which demonstrated the effectiveness of artificial neural networks for cancer classification using gene expression data. Using a publicly available RNA-seq dataset from Kaggle, I applied preprocessing, dimensionality reduction with principal component analysis (PCA), and supervised learning models including a neural network. Model performance was evaluated and compared across multiple classifiers, demonstrating that gene expression profiles contain strong predictive signals for LUAD classification.

---

## Introduction

Gene expression profiling has become a powerful tool for cancer classification and diagnosis. Lung adenocarcinoma (LUAD) is one of the most common subtypes of lung cancer, and accurate identification is important for treatment decisions and biological understanding. Prior studies have shown that machine learning models, particularly neural networks, can successfully classify cancers based on expression patterns.

This project builds on the methodology described by Khan et al. (2001), adapting their framework to modern RNA-seq data. The goal is to evaluate whether LUAD samples can be reliably distinguished from other cancer types using gene expression features and to compare the performance of different machine learning approaches.

---

## Dataset

The dataset used in this project is publicly available on Kaggle:

**Gene Expression Cancer RNA-seq Dataset**  
https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq

### Dataset Overview

The dataset contains RNA-seq gene expression measurements from cancer samples representing multiple cancer types. Each sample is characterized by expression levels across thousands of genes, providing a high-dimensional molecular profile. The dataset includes lung adenocarcinoma samples alongside other cancer types, enabling a binary classification task.

### Input Files

The project expects the following files to be placed in the `data/raw/` directory:

```
data/raw/
├── expression_matrix.csv
└── sample_labels.csv
```

#### expression_matrix.csv
- Each row represents a biological sample (tumor)
- Each column represents a gene
- Values correspond to normalized RNA-seq gene expression levels
- High dimensionality: thousands of genes per sample

#### sample_labels.csv
- Contains cancer type labels for each sample
- Samples labeled `LUAD` represent lung adenocarcinoma
- All other cancer types are grouped as `Non-LUAD` for binary classification

Raw data files are not included in this repository.

---

## Methods

### Data Exploration

Before modeling, I performed an initial exploration of the dataset to understand its structure and quality. This included examining the dimensions of the expression matrix and label files, verifying that sample counts matched, analyzing the distribution of cancer types, checking for missing values, and identifying which samples corresponded to LUAD. This step helped ensure that the data was suitable for further analysis and prevented potential issues later in the pipeline.

---

### Preprocessing

#### Binary Label Creation  
The original multi-class cancer labels were converted into a binary classification problem by grouping all non-LUAD cancer types into a single category. This simplified the task and aligned it with the research focus on distinguishing LUAD from other cancers.

#### Train-Test Split  
The dataset was divided into training and test sets using a stratified split to preserve class proportions. The training set was used for model development, while the test set was held out for final evaluation. This separation was performed before any transformations to avoid data leakage.

#### Feature Standardization  
Gene expression values were standardized using `StandardScaler` to have zero mean and unit variance. The scaler was fitted only on the training data and then applied to the test data using the same parameters, ensuring that information from the test set did not influence preprocessing.

---

### Dimensionality Reduction

Gene expression data is highly dimensional, which can lead to overfitting, increased computational cost, and reduced interpretability. To address this, I applied Principal Component Analysis (PCA).

PCA transforms the original gene expression features into a smaller set of uncorrelated principal components that capture the most variance in the data. The PCA model was fitted only on the training data and then used to transform both training and test sets. This reduced the number of features while retaining most of the meaningful biological signal.

This approach helped remove noise, handle collinearity among genes, and improve generalization by reducing model complexity. Although interpretability at the individual gene level was reduced, the benefits in performance and efficiency justified this trade-off.

---

### Model Training and Evaluation Strategy

This project did **not** use a separate validation set. Instead, I used a train-test split combined with 3-fold stratified cross-validation on the training set.

#### Cross-Validation Procedure

The training set was divided into three folds. In each iteration, two folds were used for training and one for validation. This process was repeated three times so that each fold served as the validation set once. The final cross-validation score was obtained by averaging the three validation scores.

This approach made efficient use of the data, provided a more stable performance estimate, and reduced the risk of relying on a single arbitrary validation split.

#### Performance Metrics

Two main metrics were used:
- **Cross-Validation Accuracy:** assessed generalization performance during training.
- **Test Accuracy:** evaluated performance on completely unseen data.

Using both metrics provided a more reliable assessment of model robustness.

---

### Neural Network Architecture

I implemented a multilayer perceptron with two hidden layers followed by a softmax output layer. The network received PCA-transformed features as input and produced class probabilities for LUAD and Non-LUAD.

The hidden layers used ReLU activation functions to capture non-linear relationships, while the output layer used softmax to generate probabilities. The model was trained using the Adam optimizer, which adapts learning rates during training for faster convergence.

Early stopping was used to prevent overfitting by monitoring validation performance and stopping training when improvements plateaued.

Neural networks were chosen due to their ability to model complex patterns in high-dimensional biological data, following the approach of Khan et al. (2001).

---

### Model Comparison

To assess whether a neural network was necessary for this task, I compared it against three other classifiers:

1. **Logistic Regression:** a simple linear model serving as a baseline.
2. **Support Vector Machine (RBF kernel):** a strong non-linear classifier for high-dimensional data.
3. **Random Forest:** an ensemble method that combines multiple decision trees.
4. **Neural Network:** the primary model.

All models were trained on the same PCA-transformed data using the same train-test split and cross-validation folds to ensure a fair comparison.

---

## Results

### Performance Comparison

| Model                | Test Accuracy | CV Accuracy | ROC-AUC | Training Time |
|---------------------|---------------|-------------|---------|---------------|
| Logistic Regression | 100.00%       | 99.82%      | 1.000   | 0.049s        |
| SVM (RBF)          | 99.59%        | 99.46%      | 1.000   | 0.013s        |
| Random Forest      | 97.93%        | 96.96%      | 0.999   | 0.161s        |
| Neural Network     | 93.78%        | 89.64%      | 0.985   | 0.043s        |

### Key Findings

All models achieved high classification accuracy, indicating that gene expression data contains strong discriminative information for LUAD.

PCA successfully reduced dimensionality while preserving important biological signals, enabling efficient training and strong performance across all classifiers.

Interestingly, simpler models such as Logistic Regression outperformed the neural network, suggesting that PCA transformed the data into a space where a linear decision boundary was sufficient.

---

### Model Complexity

The fact that Logistic Regression performed best highlights the importance of matching model complexity to the problem. After PCA, a simpler model generalized better than a neural network with many parameters.

### Comparison with Khan et al. (2001)

While Khan et al. demonstrated the effectiveness of neural networks on microarray data, this work extends that approach to modern RNA-seq data and shows that dimensionality reduction can make simpler models highly effective.

### Limitations

- The problem was simplified to binary classification rather than multi-class.
- All data came from a single dataset without external validation.
- PCA reduced interpretability at the gene level.
- There was class imbalance, though stratified splitting helped mitigate this.

---

## Conclusion

This project demonstrates that LUAD can be accurately classified using gene expression data and machine learning. While neural networks are capable models, effective dimensionality reduction allowed simpler methods to outperform them.

The study highlights the importance of preprocessing and feature engineering in biological machine learning tasks and reinforces the value of gene expression profiling for cancer classification.

---

## Project Structure

```
bioinfoFinal/
├── compare_methods.py      # Trains and compares 4 ML algorithms
├── explore_data.py         # Initial data quality assessment
├── preprocessing.py        # Data cleaning, splitting, normalization
├── pca_analysis.py         # Dimensionality reduction
├── neural_network.py       # Neural network training and evaluation
├── requirements.txt        # Python package dependencies
├── data/
│   ├── raw/                # Original dataset files (not included)
│   └── processed/          # Preprocessed, transformed data
├── results/
│   ├── metrics/            # Performance metrics (CSV files)
│   └── models/             # Trained models (pickle files)
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**Python version:** 3.12+

---

## How to Run

After downloading the dataset from Kaggle and placing files in `data/raw/`, run the scripts in order:

```bash
python explore_data.py
python preprocessing.py
python pca_analysis.py
python neural_network.py
python compare_methods.py
```

Outputs will be saved in `results/metrics/` and `results/models/`.

---

## References

Khan, J., Wei, J. S., Ringnér, M., Saal, L. H., Ladanyi, M., Westermann, F., ... & Meltzer, P. S. (2001). Classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks. *Nature Medicine*, 7(6), 673-679.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC1282521/

Dataset: Debatreyadas. (2020). Gene Expression Cancer RNA-Seq Dataset. Kaggle.  
https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq
