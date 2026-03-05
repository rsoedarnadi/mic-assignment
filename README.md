# MIC Peptide Machine Learning Models and Feature Selection
This assignment is a part of the coursework for ICT665 AI & ML in Healthcare course at Hamad Bin Khalifa University. With it, we aim to address the following objectives
### Objectives ✍🏼
1. to analyze collinearity among input variables and address it
2. to preprocess the dataset and split it into 9:1 ratio for training and testing
3. to train Linear Regression, Polynomial Regression of Degree 2, Lasso Regression, Ridge Regression, and Random Forest Regression models using 5-fold CV
4. to apply GA to select a subset of features for all models, training them on the selected features, and report the configuration with the best results
5. to answer questions regarding coefficient values of the Linear, Polynomial, and Random Forest regressors
### Dataset Description 🦠
We have a dataset for Peptides (Data.csv). This dataset is composed of a range of features/information (39 features, F1, F2, …, F39) from peptide sequence. And the last column represents the minimum inhibitory concentration (Target) of peptide against pathogens (i.e., bacteria).
