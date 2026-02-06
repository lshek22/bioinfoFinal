# bioinfoFinal

**LUAD vs Non-LUAD classification using gene expression data**

## overview

this project implements a machine learning pipeline for binary cancer classification using gene expression data. the task is to distinguish lung adenocarcinoma (LUAD) samples from non-LUAD cancer samples based on rna-seq expression profiles.

the approach is inspired by the methodology described in:

khan et al. (2001). classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC1282521/

the pipeline includes preprocessing, dimensionality reduction using pca, neural network training, and comparison with other machine learning models.

---

## dataset

the dataset used in this project is publicly available on kaggle:

**gene expression cancer rna-seq dataset**  
https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq

---

## required input files

place the following files in the `data/raw/` directory:

      data/raw/
      ├── expression_matrix.csv
      └── sample_labels.csv



### expression_matrix.csv

- rows represent samples  
- columns represent genes  
- values are normalized gene expression levels  
- optional first column may contain sample ids  

### sample_labels.csv

- one row per sample  
- must contain a column named `Class`  
- samples labeled `LUAD` are treated as lung adenocarcinoma  
- all other cancer types are grouped as `Non-LUAD`  

raw data is not included in this repository.

---

## project structure
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


