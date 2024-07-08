import unittest
from unittest.mock import patch, mock_open, MagicMock

from io import StringIO
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_curve, auc,classification_report,
                             confusion_matrix,precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,StandardScaler

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workflow_functions import (df_corr, find_correlating_features, permutation_test, general_two_class, constant_value_columns,
                                grid_search_regressor_parameters,grid_search_classifier_parameters,
                                split_test_regressors_with_optimization, kfold_test_classifiers_with_optimization, directory_names,
                                build_data_from_directory, build_data_from_directory_regr, metrics_for_regression, metrics_for_all_classes,
                                which_are_misclassified, get_feature_names_from_column_transformers, feature_categorization, scale_test_set,
                                ensemble)

class TestDfCorr(unittest.TestCase):
    def test_pearson_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        corr_method = 'pearson'
        result = df_corr(x, y, corr_method)
        expected = 1.0
        self.assertAlmostEqual(result.iloc[0, 1], expected, places=5)

    def test_invalid_method(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        corr_method = 'invalid'
        with self.assertRaises(ValueError):
            df_corr(x, y, corr_method)

class TestFindCorrelatingFeatures(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [5, 4, 3, 2, 1]
        })
        self.targets = pd.Series([1, 2, 3, 4, 5])

    def test_default_parameters(self):
        result = find_correlating_features(self.features, self.targets, thresh=0.9)
        self.assertIn('B', result)
        self.assertIn('A', result)
        self.assertIn('C', result)

    def test_plot_false(self):
        result = find_correlating_features(self.features, self.targets, plot=False, thresh=0.9)
        self.assertIn('B', result)
        self.assertIn('A', result)
        self.assertIn('C', result)

class TestPermutationTest(unittest.TestCase):
    def test_permutation(self):
        features = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [5, 4, 3, 2, 1]
        })
        targets = pd.Series([1, 2, 3, 4, 5])
        result = permutation_test(features, targets, n_sample=1000, corr_method='pearson', sig_level=0.05, significance = True)
        self.assertIsNotNone(result)

class TestGeneralTwoClass(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(general_two_class(3), 0)
        self.assertEqual(general_two_class(4), 1)
        self.assertEqual(general_two_class(5), 1)

class TestConstantValueColumns(unittest.TestCase):
    def test_constant_columns(self):
        df = pd.DataFrame({
            'A': [1, 1, 1, 1, 1],
            'B': [2, 2, 2, 2, 2],
            'C': [1, 2, 3, 4, 5]
        })
        result = constant_value_columns(df)
        self.assertIn('A', result)
        self.assertIn('B', result)
        self.assertNotIn('C', result)

logging.basicConfig(level=logging.INFO)

class TestGridSearchParameters(unittest.TestCase):
    def setUp(self):
        # Create a simple classification dataset
        self.X_classification, self.y_classification = make_classification(n_samples=100, n_features=4, random_state=42)
        self.X_classification = pd.DataFrame(self.X_classification)
        self.y_classification = pd.Series(self.y_classification)

        # Create a simple regression dataset
        self.X_regression, self.y_regression = make_regression(n_samples=100, n_features=4, random_state=42)
        self.X_regression = pd.DataFrame(self.X_regression)
        self.y_regression = pd.Series(self.y_regression)

        # Set up classifier options
        self.classifier = LogisticRegression()
        self.classifier_options = {'LogisticRegression': {'C': [0.1, 1, 10]}}
        self.classifier_names = ['LogisticRegression']

        # Set up regressor options
        self.regressor = LinearRegression()
        self.regressor_options = {'LinearRegression': {}}
        self.regressor_names = ['LinearRegression']

    def test_grid_search_classifier_parameters(self):
        best_params = grid_search_classifier_parameters(
            self.classifier, self.X_classification, self.y_classification,
            self.classifier_options, self.classifier_names, 0, no_train_output=True,
            cv=3, scoring=("accuracy",)
        )
        self.assertIn('C', best_params)
        self.assertIn(best_params['C'], [0.1, 1, 10])

def test_grid_search_regressor_parameters(self):
        # Create dummy data
        Xtrain = np.array([[1, 2], [3, 4], [5, 6]])
        ytrain = np.array([5, 7, 1])

        # Define regressor options
        rgs_options = {
            "DummyRegressor": {"constant": [0.5, 1, 1.5]}
        }
        rgs_names = ["DummyRegressor"]

        # Mock logging
        log = logging.getLogger(__name__)
        log.info = jest.mock()
        log.debug = jest.mock()

        # Call the function
        best_params = grid_search_regressor_parameters(DummyRegressor(), Xtrain, ytrain, rgs_options, rgs_names, 0,
                                                       False)

        # Assertions
        # Check if GridSearchCV was called with correct parameters
        assert log.info.call_count == 1
        assert "name: DummyRegressor parameters: {'constant': [0.5, 1, 1.5]}" in log.info.call_args[0]

        assert isinstance(best_params, dict)
        # Check if at least one parameter was searched over
        assert len(best_params) > 0

        # Check if training data has the expected shape
        log.debug.assert_called_once_with("\tCV xtrain: {}".format(Xtrain.shape))


def test_kfold_test_regressor_with_optimization():
    """
    Tests the kfold_test_regressor_with_optimization function with mock data.
    """

    # Create mock dataframes
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [4, 5, 6, 7, 8], "target": [10, 12, 15, 18, 21]})
    targets = df["target"]
    regressors = [LinearRegression]  # Replace with actual regressors
    rgs_options = {"n_jobs": [-1, 1]}  # Example options
    smiles = None
    names = None

    # Mock functions (replace with actual implementations)
    def grid_search_regressor_parameters(regs, Xtrain, ytrain, rgs_options, rgs_names, iteration, no_train_output, cv=5,
                                         name=""):
        # Mock grid search, return dummy parameters
        return {"n_jobs": -1}

    # Suppress logging
    logging.disable(logging.CRITICAL)

    # Run the test function with overwrite
    test_kfold_test_regressor_with_optimization(df.copy(), targets.copy(), regressors, rgs_options, scale=True, cv=2,
                                                n_repeats=2, random_seed=107901, overwrite=True)

    # Assertions (check for fold data)
    assert os.path.exists("Linear_Regression/0.csv")
    assert os.path.exists("Linear_Regression/1.csv")

    # Remove temporary files
    for filename in os.listdir("Linear_Regression"):
        os.remove(os.path.join("Linear_Regression", filename))
    os.rmdir("Linear_Regression")

    # Run the test function without overwrite (should raise a warning)
    with capture_warnings(record=True) as warnings:
        test_kfold_test_regressor_with_optimization(df.copy(), targets.copy(), regressors, rgs_options, scale=True,
                                                    cv=2, n_repeats=2, random_seed=107901, overwrite=False)
        assert len(warnings) == 1  # Check for a warning about existing directory

    # Re-enable logging
    logging.enable(logging.CRITICAL)

def test_split_test_regressors_with_optimization():
  """
  Tests the split_test_regressors_with_optimization function with mock data.
  """

  # Create mock dataframes
  df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [4, 5, 6, 7, 8], "target": [10, 12, 15, 18, 21]})
  test_df = pd.DataFrame({"feature1": [6, 7, 8], "feature2": [9, 10, 11], "target": [24, 27, 30]})
  targets = df["target"]
  test_targets = test_df["target"]
  regressors = [LinearRegression]  # Replace with actual regressors
  rgs_options = {"n_jobs": [-1, 1]}  # Example options

  # Mock functions (replace with actual implementations)
  def grid_search_regressor_parameters(reg, Xtrain, ytrain, rgs_options, rgs_names, iteration, no_train_output, cv=5, name=""):
    # Mock grid search, return dummy parameters
    return {"n_jobs": -1}

  # Suppress plotting (optional)
  with patch.object(plt, 'show') as mock_show:
      mock_show.side_effect = lambda: None

  # Run the test function with overwrite
  test_split_test_regressors_with_optimization(df.copy(), targets.copy(), test_df.copy(), test_targets.copy(), regressors, rgs_options, scale=True, cv=2, n_repeats=2, random_seed=107901, overwrite=True)

  # Assertions (check for output files)
  assert os.path.exists("Linear_Regression/model_Linear_Regression.sav")
  assert os.path.exists("Linear_Regression/predictions.csv")
  assert os.path.exists("Linear_Regression/train_metrics.csv")
  assert os.path.exists("Linear_Regression/test_metrics.csv")
  assert os.path.exists("Linear_Regression/0.png")

  # Remove temporary files
  for filename in os.listdir("Linear_Regression"):
    os.remove(os.path.join("Linear_Regression", filename))
  os.rmdir("Linear_Regression")


def test_kfold_test_classifiers_with_optimization():
  """
  Tests the kfold_test_classifiers_with_optimization function with mock data.

  This function performs classification using kfold cross-validation.
  """

  # Create mock dataframes
  df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [4, 5, 6, 7, 8]})
  classes = pd.Series([0, 1, 0, 1, 0])
  classifiers = [DummyClassifier]  # Replace with actual classifiers
  clf_options = {"n_jobs": [-1, 1]}  # Example options

  # Mock functions (replace with actual implementations)
  def grid_search_classifier_parameters(clf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=5, name=""):
    # Mock grid search, return dummy parameters
    return {"n_jobs": -1}

  def precision_recall_fscore_support(*args, **kwargs):
    # Mock precision_recall_fscore_support, return dummy scores
    return np.array([[0.7, 0.8, 0.9, 100], [0.6, 0.7, 0.8, 50]])

  def classification_report_imbalanced(*args, **kwargs):
    # Mock classification_report_imbalanced, return dummy report
    return "classification report"

  def sensitivity_specificity_support(*args, **kwargs):
    # Mock sensitivity_specificity_support, return dummy values
    return (0.9, 0.8, [10, 20])

  # Suppress logging
  logging.disable(logging.CRITICAL)

  # Run the test function with overwrite
  kfold_test_classifiers_with_optimization(df.copy(), classes.copy(), classifiers, clf_options, scale=True, cv=2, n_repeats=2, random_seed=107901, overwrite=True)

  # Assertions (check for fold data and reports)
  assert os.path.exists("DummyClassifier/0.csv")
  assert os.path.exists("DummyClassifier/1.csv")
  assert os.path.exists("DummyClassifier/DummyClassifier_fold_0_score.csv")

  # Remove temporary files
  for filename in os.listdir("DummyClassifier"):
    os.remove(os.path.join("DummyClassifier", filename))
  os.rmdir("DummyClassifier")

  # Re-enable logging
  logging.enable(logging.CRITICAL)

def test_kfold_test_classifiers_with_optimization_weights():
  """
  Tests the kfold_test_classifiers_with_optimization_weights function with mock data.

  This function performs classification using kfold cross-validation with class weights.
  """

  # Create mock dataframes
  df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [4, 5, 6, 7, 8]})
  classes = pd.Series([0, 1, 0, 1, 0])
  classifiers = [DummyClassifier]  # Replace with actual classifiers
  clf_options = {"n_jobs": [-1, 1]}  # Example options
  class_weight = {0: 0.1, 1: 0.9}  # Example class weights

  # Mock functions (replace with actual implementations)
  def grid_search_classifier_parameters(clf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=5, name=""):
    # Mock grid search, return dummy parameters
    return {"n_jobs": -1}

  def precision_recall_fscore_support(*args, **kwargs):
    # Mock precision_recall_fscore_support, return dummy scores
    return np.array([[0.7, 0.8, 0.9, 100], [0.6, 0.7, 0.8, 50]])

  def classification_report_imbalanced(*args, **kwargs):
    # Mock classification_report_imbalanced, return dummy report
    return "classification report"

  def sensitivity_specificity_support(*args, **kwargs):
    # Mock sensitivity_specificity_support, return dummy values
    return (0.9, 0.8, [10, 20])

  # Suppress logging
  logging.disable(logging.CRITICAL)

  # Run the test function with overwrite
  kfold_test_classifiers_with_optimization_weights(df.copy(), classes.copy(), classifiers, clf_options, scale=True, cv=2, n_repeats=2, random_seed=107901, class_weight=class_weight, overwrite=True)

  # Assertions (check for fold data and reports)
  assert os.path.exists("DummyClassifier/0.csv")
  assert os.path.exists("DummyClassifier/1.csv")
  assert os.path.exists("DummyClassifier/DummyClassifier_fold_0_score.csv")
  assert os.path.exists("DummyClassifier/DummyClassifier_roc_curves.png")

  # Remove temporary files
  for filename in os.listdir("DummyClassifier"):
    os.remove(os.path.join("DummyClassifier", filename))
  os.rmdir("DummyClassifier")

  # Re-enable logging
  logging.enable(logging.CRITICAL)


def test_test_classifiers_with_optimization():
  """
  Tests the test_classifiers_with_optimization function with mock data.
  """

  # Create mock dataframes
  df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": ["A", "A", "B"]})
  test_df = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [10, 11, 12], "target": ["A", "B", "B"]})
  classes = ["A", "B"]
  testclasses = ["A", "B", "B"]
  classifiers = [DummyClassifier]  # Replace with actual classifiers
  clf_options = {"C": [0.001, 0.1, 1], "kernel": ["linear", "rbf"]}
  clf_names = ["Dummy Classifier"]

  # Mock functions (replace with actual implementations)
  def minmaxscale(data):
    return data

  def grid_search_classifier_parameters(clf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=5, name=""):
    # Mock grid search, return dummy parameters
    return {"C": 0.1, "kernel": "linear"}

  # Suppress logging
  logging.disable(logging.CRITICAL)

  # Run the test function
  test_classifiers_with_optimization(df.copy(), test_df.copy(), classes, testclasses, classifiers, clf_options, overwrite=False, scale=True, cv=2, n_repeats=1, random_seed=107901, clf_names=clf_names, no_train_output=True, class_labels=(0, 1))

  # Assertions
  assert os.path.exists("Dummy_Classifier/fold_0.csv")  # Check fold data is saved
  assert os.path.exists("Dummy_Classifier/model_Dummy_Classifier.sav")  # Check model is saved
  os.remove("Dummy_Classifier/fold_0.csv")  # Remove temporary files
  os.remove("Dummy_Classifier/model_Dummy_Classifier.sav")

  # Test functionality without saving models
  test_classifiers_with_optimization(df.copy(), test_df.copy(), classes, testclasses, classifiers, clf_options, overwrite=False, scale=True, cv=2, n_repeats=1, random_seed=107901, clf_names=clf_names, no_train_output=True, class_labels=(0, 1))

  # Test functionality with overwriting directory
  test_classifiers_with_optimization(df.copy(), test_df.copy(), classes, testclasses, classifiers, clf_options, overwrite=True, scale=True, cv=2, n_repeats=1, random_seed=107901, clf_names=clf_names, no_train_output=True, class_labels=(0, 1))

  # Remove leftover directory
  os.rmdir("Dummy_Classifier")

  # Re-enable logging
  logging.enable(logging.CRITICAL)

def test_build_data_from_directory():
  """
  Tests the build_data_from_directory function with a mock directory structure.
  """

  # Create a mock directory with some CSV files
  data_directory = "test_data"
  max_folds = 2
  os.makedirs(data_directory, exist_ok=True)
  for i in range(max_folds):
    data_fold = pd.DataFrame({"m_index": [1, 2, 3], "known": [4, 5, 6], "prediction": [7, 8, 9], "prob0": [0.1, 0.2, 0.3], "prob1": [0.9, 0.8, 0.7]})
    data_fold.to_csv(os.path.join(data_directory, f"{i}.csv"), index=False)

  # Capture logging output
  log_catcher = logging.capturer()
  with log_catcher:
    data = build_data_from_directory(data_directory, max_folds)

  # Assertions for data processing
  assert len(data.columns) == 5  # Check if there are 5 columns
  assert data.shape[0] == max_folds * 3  # Check if all data is concatenated without duplicates
  assert list(data) == ["m_index", "known", "prediction", "prob0", "prob1"]  # Check column order
  assert data.index.dtype == int  # Check if index is integer
  assert not data.duplicated(subset=["m_index"]).any()  # Check for duplicates

  # Test functionality with save option (separate test)
  data = build_data_from_directory(data_directory, max_folds, save=True)
  assert os.path.exists(os.path.join(data_directory, "all_predictions.csv"))  # Check if file is saved
  os.remove(os.path.join(data_directory, "all_predictions.csv"))  # Remove saved file

  # Check logging messages
  logs = log_catcher.getvalue()
  for i in range(max_folds):
    assert f"Reading {i}.csv" in logs  # Check if each file is read

  logging.disable(logging.CAPTURE)  # Disable logging capture

def test_build_data_from_directory_regr():
  """
  Tests the build_data_from_directory_regr function with a mock directory structure.
  """

  # Create a mock directory with some CSV files
  data_directory = "test_data"
  max_folds = 3
  os.makedirs(data_directory, exist_ok=True)
  for i in range(max_folds):
    data_fold = pd.DataFrame({"known": [1, 2, 3], "prediction": [4, 5, 6]})
    data_fold.to_csv(os.path.join(data_directory, f"{i}.csv"), index=False)

  # Capture logging output
  log_catcher = logging. capturarer()
  with log_catcher:
    data = build_data_from_directory_regr(data_directory, max_folds)

  # Assertions
  assert len(data.columns) == 3  # Check if there are 3 columns
  assert data.shape[0] == max_folds * 3  # Check if all data is concatenated
  assert list(data) == ["known", "prediction", "index"]  # Check column order
  assert data.index.dtype == int  # Check if index is integer

  # Remove mock directory
  os.rmdir(data_directory)

  # Check logging messages
  logs = log_catcher.getvalue()
  for i in range(max_folds):
    assert f"Reading {i}.csv" in logs  # Check if each file is read

  logging.disable(logging.CAPTURE)  # Disable logging capture


class TestMetricsForRegression(unittest.TestCase):

    @patch('matplotlib.pyplot.savefig')  # Mock savefig from plt
    @patch('workflow_functions.build_data_from_directory_regr')  # Mock the data building function
    def test_metrics_for_regression(self, mock_build_data, mock_savefig):
        # Mock data
        mock_build_data.return_value = {
            'known': [1.0, 2.0, 3.0],
            'prediction': [1.1, 1.9, 3.2]
        }

        # Call the function
        directories = ['test_directory']
        metrics_for_regression(directories=directories)

        # Check if metrics.txt was created and contains the expected content
        metrics_path = "test_directory/metrics.txt"
        self.assertTrue(os.path.isfile(metrics_path), "metrics.txt was not created")

        with open(metrics_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5, "metrics.txt does not have the expected number of lines")

            # Update these values based on the actual output from your function
            self.assertAlmostEqual(float(lines[0].strip()), 0.9767, places=2, msg="Variance is incorrect")  # Update if needed
            self.assertAlmostEqual(float(lines[1].strip()), 0.1333, places=2, msg="MAE is incorrect")  # Updated
            self.assertAlmostEqual(float(lines[2].strip()), 0.02, places=2, msg="MSE is incorrect")  # Update if needed
            self.assertAlmostEqual(float(lines[3].strip()), 0.14, places=2, msg="RMSE is incorrect")  # Update if needed
            self.assertAlmostEqual(float(lines[4].strip()), 0.97, places=2, msg="R2 is incorrect")  # Updated

        # Ensure savefig was not called
        mock_savefig.assert_not_called()

class TestMetricsForAllClasses(unittest.TestCase):

    def setUp(self):
        # Set up a stream to capture log output
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def tearDown(self):
        # Remove the log handler and close the stream
        logging.getLogger().removeHandler(self.log_handler)
        self.log_handler.close()
        self.log_stream.close()

    @patch('workflow_functions.build_data_from_directory')
    @patch('workflow_functions.cmetrics')
    @patch('builtins.open', new_callable=mock_open, create=True)
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('os.makedirs')

    def test_metrics_for_all_classes(self, mock_makedirs, mock_os_path_join, mock_open, mock_cmetrics,
                                     mock_build_data_from_directory):

        # Mock return values for the functions
        mock_build_data_from_directory.return_value = pd.DataFrame({
        'known': [1, 0, 1, 1, 0],
        'prediction': [1, 1, 0, 1, 0],
        'names': ['mol1', 'mol2', 'mol3', 'mol4', 'mol5'],
        'smiles': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'prob0': [0.9, 0.6, 0.8, 0.7, 0.5],
        'prob1': [0.1, 0.4, 0.2, 0.3, 0.5]
        })

        mock_cmetrics.get_multi_label_confusion_matrix.return_value = {'metric': 'dummy_value'}

        mock_cmetrics.calculate_multi_label_confusion_based_metrics.return_value = [
            {'tpr': 0.8, 'fpr': 0.2, 'tnr': 0.8, 'fnr': 0.2, 'f_half': 0.7, 'f1': 0.75, 'f2': 0.72,
            'matthews_correlation_coefficient': 0.6, 'precision': 0.7, 'recall': 0.8, 'roc_auc': 0.8, 'pr_auc': 0.75},

            {'tpr': 0.6, 'fpr': 0.3, 'tnr': 0.7, 'fnr': 0.3, 'f_half': 0.65, 'f1': 0.68, 'f2': 0.66,
            'matthews_correlation_coefficient': 0.5, 'precision': 0.6, 'recall': 0.6, 'roc_auc': 0.7, 'pr_auc': 0.65}
        ]
        mock_cmetrics.calculate_confusion_based_metrics.return_value = {'dummy_confusion_metric': 'value'}

        # Run the function
        from workflow_functions import metrics_for_all_classes
        metrics_for_all_classes(directories=["test_directory"])

        # Get log output
        log_output = self.log_stream.getvalue()
        self.assertIn('Analyzing predictions for model test_directory', log_output)
        self.assertIn('Over all data points including smote points', log_output)
        self.assertIn('Over all REAL data points NOT including smote points', log_output)

        # Capture written data
        written_files = {call[0][0]: call[1] for call in mock_open().write.call_args_list}

        # Convert bytes to strings if necessary
        for filename, content in written_files.items():
            if isinstance(content, bytes):
                written_files[filename] = content.decode('utf-8')

                # Expected contents
                expected_csv_content = 'known,prediction,names,smiles\n1,1,mol1,C1\n0,1,mol2,C2\n1,0,mol3,C3\n1,1,mol4,C4\n0,0,mol5,C5\n'
                expected_metrics_content = "[{'tpr': 0.8, 'fpr': 0.2, 'tnr': 0.8, 'fnr': 0.2, 'f_half': 0.7, 'f1': 0.75, 'f2': 0.72, 'matthews_correlation_coefficient': 0.6, 'precision': 0.7, 'recall': 0.8, 'roc_auc': 0.8, 'pr_auc': 0.75'}, {'tpr': 0.6, 'fpr': 0.3, 'tnr': 0.7, 'fnr': 0.3, 'f_half': 0.65, 'f1': 0.68, 'f2': 0.66, 'matthews_correlation_coefficient': 0.5, 'precision': 0.6, 'recall': 0.6, 'roc_auc': 0.7, 'pr_auc': 0.65'}]"
                expected_tex_content = ('\\begin{tabular}{lllllllllllll}\n'
                                        '\\toprule\n'
                                        ' &  tpr &  fpr &  tnr &  fnr & f\\_half &    f1 &    f2 &  MCC & precision & recall & roc\\_auc & pr\\_auc \\\\\n'
                                        '\\midrule\n'
                                        '0 &  0.8 &  0.2 &  0.8 &  0.2 &    0.7 &  0.75 &  0.72 &  0.6 &       0.7 &    0.8 &     0.8 &   0.75 \\\\\n'
                                        '1 &  0.6 &  0.3 &  0.7 &  0.3 &   0.65 &  0.68 &  0.66 &  0.5 &       0.6 &    0.6 &     0.7 &   0.65 \\\\\n'
                                        '\\bottomrule\n'
                                        '\\end{tabular}\n')

                # Assertions
                self.assertTrue(any(expected_csv_content in content for filename, content in written_files.items() if
                                    filename.endswith('metrics.csv')),
                                "CSV content not found in written data")
                self.assertTrue(any(
                    expected_metrics_content in content for filename, content in written_files.items() if
                    filename.endswith('multi_metrics.txt')),
                                "Metrics content not found in written data")
                self.assertTrue(any(expected_tex_content in content for filename, content in written_files.items() if
                                    filename.endswith('metric.tex')),
                                "TeX content not found in written data")

    
class TestWhichAreMisclassified(unittest.TestCase):

    def setUp(self):
        # Setup a DataFrame with various columns for testing
        self.df = pd.DataFrame({
            'known': [1, 0, 1, 1, 0],
            'prediction': [1, 1, 0, 1, 0],
            'names': ['mol1', 'mol2', 'mol3', 'mol4', 'mol5'],
            'smiles': ['C1', 'C2', 'C3', 'C4', 'C5']
        })

        # Setup logging to capture log messages
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def tearDown(self):
        # Remove the log handler and close the stream
        logging.getLogger().removeHandler(self.log_handler)
        self.log_handler.close()
        self.log_stream.close()

    def test_no_misclassified(self):
        # All predictions are correct
        df_no_misclassified = self.df.copy()
        df_no_misclassified['prediction'] = df_no_misclassified['known']
        result_df = which_are_misclassified(df_no_misclassified, return_indx=False)
        self.assertTrue(result_df.empty)

    def test_some_misclassified(self):
        # Test with some misclassified entries
        result_df = which_are_misclassified(self.df, return_indx=False)
        expected_df = self.df[self.df['known'] != self.df['prediction']]
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_with_names_and_smiles(self):
        # Test when both 'names' and 'smiles' columns are present
        which_are_misclassified(self.df, return_indx=False)
        log_output = self.log_stream.getvalue()
        self.assertIn('The molecules which are misclassified are:', log_output)
        self.assertIn('mol2 C2', log_output)
        self.assertIn('mol3 C3', log_output)

    def test_with_only_names(self):
        # Test when only 'names' column is present
        df_names_only = self.df.drop(columns='smiles')
        which_are_misclassified(df_names_only, return_indx=False)
        log_output = self.log_stream.getvalue()
        self.assertIn('The molecules which are misclassified are:', log_output)
        self.assertIn('mol2', log_output)
        self.assertIn('mol3', log_output)

    def test_with_only_smiles(self):
        # Test when only 'smiles' column is present
        df_smiles_only = self.df.drop(columns='names')
        which_are_misclassified(df_smiles_only, return_indx=False)
        log_output = self.log_stream.getvalue()
        self.assertIn('The molecules which are misclassified are:', log_output)
        self.assertIn('C2', log_output)
        self.assertIn('C3', log_output)

    def test_return_indices(self):
        # Test when return_indx is True
        result_df = which_are_misclassified(self.df, return_indx=True)
        expected_indices = self.df[self.df['known'] != self.df['prediction']].index.tolist()
        self.assertListEqual(result_df.index.tolist(), expected_indices)

class TestGetFeatureNamesFromColumnTransformers(unittest.TestCase):
    def setUp(self):
        # Setup a DataFrame for testing
        self.df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['cat1', 'cat2', 'cat1'],
            'C': [1.5, 2.5, 3.5],
            'D': [np.nan, 2.0, 3.0]
        })

        # Define transformers for testing
        self.transformers = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['C']),
                ('cat', OneHotEncoder(), ['B']),
                ('miss', SimpleImputer(strategy='mean'), ['D'])
            ]
        )

        # Fit the ColumnTransformer with the DataFrame
        self.transformers.fit(self.df)


    def test_feature_names(self):
        # Create an instance of the ColumnTransformer
        transformer = self.transformers

        # Expected feature names
        expected_feature_names = [
            'C',                # StandardScaler
            'B_cat1',           # OneHotEncoder
            'B_cat2',           # OneHotEncoder
            'D'                 # SimpleImputer (does not modify the feature name)
        ]

        # Run the function
        result_feature_names = get_feature_names_from_column_transformers(transformer)

        # Test if the result matches the expected feature names
        self.assertEqual(result_feature_names, expected_feature_names)

def test_feature_categorization():
  """
  Tests the feature_categorization function for various data and parameter combinations
  """

  # Sample data with different feature types
  data = {'categorical_feature': ['0', '1', '0', '1'],
          'numerical_feature': [1.3, 2.4, 3.5, 4.6],
          'boolean_feature': [True, False, True, False]}
  df = pd.DataFrame(data)

  # Test with all categorical features (feature_types="categorical")
  scaled_df, categorical_idxs, numerical_transformer, categorical_transformer = feature_categorization(df.copy(), feature_types="categorical")
  assert categorical_idxs is None  # No need for categorical indices when all are categorical
  assert isinstance(numerical_transformer, MinMaxScaler) is False  # No numerical scaling applied
  assert isinstance(categorical_transformer, OneHotEncoder) is True  # OneHotEncoder used for categorical features
  assert len(scaled_df.columns) > len(df.columns)  # More columns after one-hot encoding

  # Test with some categorical features (feature_types="some_categorical")
  categorical_idxs = [0, 2]  # Specify indices of categorical features
  scaled_df, categorical_idxs, numerical_transformer, categorical_transformer = feature_categorization(df.copy(), feature_types="some_categorical", categorical_indxs=categorical_idxs)
  assert categorical_idxs == [0, 2]  # Check if categorical indices are preserved
  assert isinstance(numerical_transformer, MinMaxScaler) is True  # MinMaxScaler used for numerical features
  assert isinstance(categorical_transformer, OneHotEncoder) is True  # OneHotEncoder used for categorical features
  assert len(scaled_df.columns) > len(df.columns)  # More columns after one-hot encoding for categorical features

  # Test with no categorical features (feature_types="no_categorical")
  scaled_df, categorical_idxs, numerical_transformer, categorical_transformer = feature_categorization(df.copy(), feature_types="no_categorical")
  assert categorical_idxs is None  # No categorical indices
  assert isinstance(numerical_transformer, MinMaxScaler) is True  # MinMaxScaler used for numerical features
  assert isinstance(categorical_transformer, OneHotEncoder) is False  # No categorical encoding applied
  assert len(scaled_df.columns) == len(df.columns)  # No change in column number

  # Test with no scaling or encoding (feature_types="none")
  scaled_df, categorical_idxs, numerical_transformer, categorical_transformer = feature_categorization(df.copy(), feature_types="none")
  assert scaled_df.equals(df)  # Dataframe remains unchanged
  assert categorical_idxs is None  # No categorical indices
  assert numerical_transformer is None  # No numerical transformer
  assert categorical_transformer is None  # No categorical transformer

  # Test handling boolean data (automatic conversion to integer)
  df["boolean_feature2"] = [False, True, False, True]
  scaled_df, categorical_idxs, numerical_transformer, categorical_transformer = feature_categorization(df.copy(), feature_types="some_categorical", categorical_indxs=[0, 2])
  assert df["boolean_feature2"].dtype == np.int64  # Boolean converted to integer

  # Test automatic detection of categorical features
  scaled_df, categorical_idxs, numerical_transformer, categorical_transformer = feature_categorization(df.copy())
  assert len(categorical_idxs) == 3  # Three categorical features detected automatically (including boolean)

  # Test with mismatched feature names and encoded data shape
  categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  categorical_transformer.fit([[1], [2]])  # Fit with only two unique values
  with warnings.catch_warnings(record=True) as warning_messages:
      warnings.simplefilter("always")  # Ensure all warnings are captured
      feature_categorization(df.copy(), categorical_transformer=categorical_transformer)
      assert any(msg.message.startswith("Feature names length") for msg in warning_messages)

def test_scale_test_set():
    # Create a sample training dataset
    data_train = {
        'numerical1': [1, 2, 3, 4, 5],
        'numerical2': [10, 20, 30, 40, 50],
        'categorical': ['A', 'B', 'A', 'B', 'A']
    }
    features_df_train = pd.DataFrame(data_train)

    # Create a sample test dataset
    data_test = {
        'numerical1': [2, 3],
        'numerical2': [15, 35],
        'categorical': ['A', 'B']
    }
    features_df_test = pd.DataFrame(data_test)

    # Apply feature categorization on the training dataset
    features_df_train, categorical_indxs, numerical_trans, categorical_trans = feature_categorization(
        features_df_train, feature_types="some_categorical"
    )

    # Scale and transform the test dataset
    scaled_test_set = scale_test_set(features_df_test, categorical_indxs, numerical_trans, categorical_trans)

    # Print the scaled and transformed test set
    print("Scaled and Transformed Test Set:")
    print(scaled_test_set)

    # Verify the transformations
    # The numerical features should be scaled between 0 and 1
    assert (scaled_test_set[['numerical1', 'numerical2']].min().min() >= 0), "Numerical features are not scaled properly"
    assert (scaled_test_set[['numerical1', 'numerical2']].max().max() <= 1), "Numerical features are not scaled properly"

    # The categorical features should be one-hot encoded
    assert 'categorical_A' in scaled_test_set.columns, "Categorical feature 'A' is not one-hot encoded properly"
    assert 'categorical_B' in scaled_test_set.columns, "Categorical feature 'B' is not one-hot encoded properly"

    print("All tests passed successfully!")


class TestEnsembleFunction(unittest.TestCase):
   def setUp(self):
       # Create sample CSV data
       self.csv_data_1 = StringIO("""actual,predicted
       1.0,1.1
       2.0,1.9
       3.0,3.2
       4.0,3.8
       """)

       self.csv_data_2 = StringIO("""actual,predicted
       1.0,1.0
       2.0,2.1
       3.0,2.9
       4.0,4.1
       """)

       # Create temporary CSV files
       self.csv_files = ['test1.csv', 'test2.csv']
       with open(self.csv_files[0], 'w') as f:
           f.write(self.csv_data_1.getvalue())
       with open(self.csv_files[1], 'w') as f:
           f.write(self.csv_data_2.getvalue())

   def tearDown(self):
       # Remove temporary CSV files
       for file in self.csv_files:
           os.remove(file)

   def test_ensemble(self):
       metrics = ensemble(self.csv_files)

       self.assertIn('mae', metrics)
       self.assertIn('mse', metrics)
       self.assertIn('rmse', metrics)
       self.assertIn('r2', metrics)

       # Recalculate the expected metrics
       actual_values = np.array([1.0, 2.0, 3.0, 4.0])
       predictions_1 = np.array([1.1, 1.9, 3.2, 3.8])
       predictions_2 = np.array([1.0, 2.1, 2.9, 4.1])
       mean_predictions = np.mean([predictions_1, predictions_2], axis=0)

       expected_mae = mean_absolute_error(actual_values, mean_predictions)
       expected_mse = mean_squared_error(actual_values, mean_predictions)
       expected_rmse = np.sqrt(expected_mse)
       expected_r2 = r2_score(actual_values, mean_predictions)

       self.assertAlmostEqual(metrics['mae'], expected_mae)
       self.assertAlmostEqual(metrics['mse'], expected_mse)
       self.assertAlmostEqual(metrics['rmse'], expected_rmse)
       self.assertAlmostEqual(metrics['r2'], expected_r2)

if __name__ == '__main__':
    unittest.main()

