# Example: Enantioselective Strecker synthesis

Here, we use this example to fully desmontrate all notebooks available in PythiaCHEM:

## Notebook 1: Data Analysis
This notebook enables comprehensive exploration, visualization, and analysis of the data. It facilitates the identification of patterns, trends, potential biases, and outliers in the dataset before their use in ML tasks. The notebook imports the dataset in a CSV format, automatically removes duplicates, and offers various analyses, including occurrences of the target values, molecular similarity analysis through the Tanimoto similarity index, principal component analysis (PCA), and scaling.

## Notebook 2: Classification-Fingerprints
This notebook focuses on binary classification scenarios using fingerprints. Various fingerprint options are available, including Morgan fingerprints, RDKit fingerprints, MACCS keys, atom pair fingerprints, and torsion fingerprints. It handles the removal of features represented as 0 across all fingerprint bits. When dealing with an imbalanced dataset, it is possible to generate synthetic data by upsampling the minority class using SMOTE.
Model training is performed with several classifiers and model interpretation is done using Gini feature importance, permutation feature importance, and SHapley Additive exPlanations (SHAP) analysis, which assesses the relative importance of features on the predictions. 

##Notebook 3: Classification-Mordred
This notebook uses Mordred descriptors, which offer a set of features encompassing topological, geometric, and electronic properties to represent chemical structures. Mordred descriptors are easily calculated and highly interpretable, making them widely used in the field of cheminformatics. Feature selection is conducted using correlation analysis, which tests each feature individually for its correlation with the target property and with other features. Both the Pearson correlation and Spearman correlation can be used. Correlation matrices and pairwise plots can also be generated within the notebook. Features that fail specific statistical criteria are discarded, and SMOTE is used for balancing imbalanced datasets. The model training procedure and interpretation are the same as Notebook 2.


## Notebook 4: Classification-DFT
This notebook uses physical-chemical descriptors provided by the user, which can be computed using electronic structure, for example, Hartree-Fock and Density Functional Theory(DFT) or semi-empirical methods (e.g., AM1 and PM3). It also offers a way to enhance the model interpretability using SHAP analysis, which assesses the relative importance of features on the predictions.

## Notebook 5: Regression-Fingerprints
This notebook is tailored for regression tasks with continuous target values. It can generate fingerprints for the species of interest using the dataset created by Notebook 1 or provided by the user. It also incorporates ensemble learning to improve predictive performance. This notebook offers a rapid approach to building a baseline model and assessing its predictive capabilities.

## Notebook 6: Regression-Mordred
This notebook serves the same purpose as Notebook 5 but employs Mordred descriptors. With over 1800 calculable Mordred descriptors, a feature elimination method is available to retain only relevant descriptors, reducing the noise in the models.

## Notebook 7: Regression-DFT
This notebook uses precalculated physical-chemical descriptors provided by the user. These descriptors, along with corresponding target values, are used for regression tasks. While these descriptors are particularly useful for inferring the underlying chemistry, they may exhibit high multicollinearity. LASSOCV introduces a regularization term (L1) that shrinks the coefficients of multicollinear parameters to zero, offering a way to identify the descriptors that influence the model.
