# Detection_MentalModel

In this directory, it is possible to find all dataset and codes used for the automatic detection of Mental Model.

## Overview

**preprocessing_dataset** contains Python scripts with SGD dataset preprocessing and heuristic-based features extraction. It is organized as follows:
- **train**: SGD original dataset (json)
- **CSV_train**: preprocessed SGD (csv)
- **DSCreation**: preprocessing of dataset, keeping only dialogue IDs and userss turns from SGD original dataset and converting files from json to csv format
- **FeaturesIdentification**: heuristics-based features extraction
- **babynames-clean.csv**: dataset of male and female names, used to identify presence of sensitive data during features extraction
- **imperativeVB_lexicon_complete.csv**: dataset of imperative verbs, used to identify presence of imperative questions during features extraction

**mentalmodel_detection** contains codes for Mental Model's automatic detection. The baseline consists in 3 Machine Learning models: Naive Bayes, Logistic Regression, Support Vector Machine. It is organized as follows:
- **Kfold_Split_complete_first_baseline.py**: K-fold cross-validation results for combinations of heuristic features and freaquency/semantic features
- **Kfold_Split_first_baseline.py**: K-fold cross-validation results for single frequency features and semantic features
- **Kfold_Split_heu_first_baseline.py**: K-fold cross-validation results for heuristic features
- **full.csv**: dataset with users' texts, heuristics-based features and Mental Model annotation [1 Push, 0 Pull]