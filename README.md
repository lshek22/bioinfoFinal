# bioinfoFinal  
**LUAD vs Non-LUAD Classification Using Gene Expression Data**

## Abstract

Accurate classification of cancer subtypes using gene expression data is a central task in computational biology and precision medicine. In this project, we developed a machine learning pipeline to distinguish lung adenocarcinoma (LUAD) samples from non-LUAD cancer samples using RNA-seq gene expression profiles. The analysis is inspired by the approach described by Khan et al. (2001), which demonstrated the effectiveness of artificial neural networks for cancer classification using gene expression data. Using a publicly available RNA-seq dataset from Kaggle, we applied preprocessing, dimensionality reduction with principal component analysis (PCA), and supervised learning models including a neural network. Model performance was evaluated and compared across multiple classifiers, demonstrating that gene expression profiles contain strong predictive signals for LUAD classification.

---

## Introduction

Gene expression profiling has become a powerful tool for cancer classification and diagnosis. Lung adenocarcinoma (LUAD) is one of the most common subtypes of lung cancer, and accurate identification is important for treatment decisions and biological understanding. Prior studies have shown that machine learning models, particularly neural networks, can successfully classify cancers based on expression patterns.

This project builds on the methodology described by Khan et al. (2001), adapting their framework to modern RNA-seq data. The goal is to evaluate whether LUAD samples can be reliably distinguished from other cancer types using gene expression features and to compare the performance of different machine learning approaches.

---

## Dataset

The dataset used in this project is publicly available on Kaggle:

**Gene Expression Cancer RNA-seq Dataset**  
https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq

### Input files

The project expects the following files to be placed in the `data/raw/` directory:

      data/raw/
      ├── expression_matrix.csv
      └── sample_labels.csv




#### expression_matrix.csv
- rows represent samples  
- columns represent genes  
- values correspond to normalized RNA-seq gene expression levels  

#### sample_labels.csv
- one row per sample  
- contains cancer type labels  
- samples labeled `LUAD` are treated as lung adenocarcinoma  
- all other cancer types are grouped as `Non-LUAD`  

Raw data files are not included in this repository.

---

## Methods

### Data exploration

Initial inspection of the dataset was performed to assess data structure, class distribution, and missing values. This step ensured that sample counts matched between expression data and labels and provided an overview of cancer type representation.

### Preprocessing

Preprocessing steps included:
- converting multiclass cancer labels into a binary classification problem (LUAD vs Non-LUAD)
- splitting the dataset into training and test sets
- standardizing gene expression values
- saving processed datasets for downstream analysis

### Dimensionality reduction

Given the high dimensionality of gene expression data, principal component analysis (PCA) was applied to reduce feature space while retaining the majority of variance. PCA-transformed features were used as input for model training.

### Model training

A neural network classifier (multilayer perceptron) was trained on the PCA-reduced data. Model performance was evaluated using classification accuracy and ROC-AUC metrics. In addition to the neural network, other classifiers were trained for comparison.

### Model comparison

The following models were evaluated and compared:
- neural network
- support vector machine
- random forest
- logistic regression

Performance metrics and training time were used to assess relative strengths of each method.

---

## Results

The neural network achieved strong classification performance in distinguishing LUAD from non-LUAD samples. Dimensionality reduction via PCA improved training efficiency while maintaining predictive accuracy. Comparison across models showed that multiple classifiers performed well, with the neural network providing competitive or superior results depending on the evaluation metric.

These results indicate that LUAD-specific gene expression signatures are robust and can be effectively captured using supervised machine learning models.

## Results

### Performance Comparison

| Model                | Test Accuracy | CV Accuracy | ROC-AUC | Training Time |
|---------------------|---------------|-------------|---------|---------------|
| Logistic Regression | 100.00%       | 99.82%      | 1.000   | 0.049s        |
| SVM (RBF)          | 99.59%        | 99.46%      | 1.000   | 0.013s        |
| Random Forest      | 97.93%        | 96.96%      | 0.999   | 0.161s        |
| Neural Network     | 93.78%        | 89.64%      | 0.985   | 0.043s        |

**Findings:**
- All models achieved >93% test accuracy
- PCA effectively reduced dimensionality from 20,531 genes to 50 components
- Simpler linear methods (Logistic Regression) outperformed neural networks after PCA
- This suggests the classification task became linearly separable after dimensionality reduction

---

## Discussion

The findings of this project support prior work demonstrating that gene expression profiles contain sufficient information to classify cancer subtypes. The use of PCA helped mitigate the curse of dimensionality inherent to RNA-seq data, while neural networks proved effective at modeling complex expression patterns. Differences in performance across models highlight trade-offs between interpretability, computational cost, and predictive power.

---

## Conclusion

This project demonstrates that LUAD can be accurately classified from other cancer types using RNA-seq gene expression data and machine learning techniques. Inspired by earlier neural network–based approaches, the pipeline combines modern preprocessing and dimensionality reduction methods to achieve reliable classification. This work highlights the continued relevance of gene expression–based cancer classification in the era of high-throughput sequencing.

---

## Project structure
        bioinfoFinal/
        ├── compare_methods.py
        ├── explore_data.py
        ├── preprocessing.py
        ├── pca_analysis.py
        ├── neural_network.py
        ├── requirements.txt
        ├── data/
        │ ├── raw/
        │ └── processed/
        └── README.md


---

## how to run

after placing the dataset in `data/raw/`, run the scripts in the following order:

```bash
python explore_data.py
python preprocessing.py
python pca_analysis.py
python neural_network.py
python compare_methods.py


---

will be saved in:
   - `results/metrics/` - Performance metrics (CSV)
   - `results/models/` - Trained models (pickle files)
```

## References

Khan, J. et al. (2001). Classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks. *Nature Medicine*.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC1282521/

Kaggle dataset:  
https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq


