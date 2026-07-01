############################################################################
## (C)Copyright 2021-2026 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################

##################################################################
# Fraud Detection example using a CUSTOM Scikit-Learn model
# with Swarm Learning.
#
# This demonstrates how to use the 'weight_attrs' parameter
# to integrate a custom sklearn-compatible model that is NOT
# in the built-in _SKLEARN_WEIGHT_REGISTRY.
#
# The custom model (OnlineLogisticRegression) implements:
#   - partial_fit() for incremental/online learning
#   - predict() and predict_proba() for evaluation
#   - Custom weight attributes: 'theta_' and 'bias_'
#
# Integration uses the weight_attrs parameter:
#   SwarmCallback(
#       model=model,
#       weight_attrs=['theta_', 'bias_'],
#       ...
#   )
#
# This tells SwarmCallback which attributes to extract, merge,
# and inject during swarm sync rounds — without requiring the
# model to be registered in _SKLEARN_WEIGHT_REGISTRY.
##################################################################

import os
import sys
import numpy as np
import csv
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelBinarizer
from swarmlearning.sklearn import SwarmCallback, SwarmSklearnTrainer


# ================================================================== #
#  Custom Model: OnlineLogisticRegression                             #
#  A minimal logistic regression trained via SGD with partial_fit().  #
#  Uses custom weight attribute names (theta_, bias_) to exercise     #
#  the weight_attrs code path in SwarmCallback.                       #
# ================================================================== #

class OnlineLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    A custom online logistic regression classifier.

    Unlike sklearn's SGDClassifier, this model stores its weights
    as 'theta_' and 'bias_' instead of 'coef_' and 'intercept_'.
    This makes it a perfect test case for the weight_attrs feature
    in SwarmCallback.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for SGD updates.
    random_state : int or None, default=None
        Seed for reproducibility.
    """

    def __init__(self, learning_rate=0.01, random_state=None):
        self.learning_rate = learning_rate
        self.random_state = random_state

    def partial_fit(self, X, y, classes=None):
        """
        Perform one pass of SGD over the given mini-batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        classes : array-like of shape (n_classes,), optional
            Required on the first call to allocate weight arrays.
        """
        if classes is not None:
            self.classes_ = np.array(classes)

        # Lazy initialization of weights
        if not hasattr(self, 'theta_'):
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state)
            else:
                rng = np.random.RandomState()
            n_features = X.shape[1]
            # Small random initialization to break symmetry
            self.theta_ = rng.randn(1, n_features).astype(np.float64) * 0.01
            self.bias_ = np.zeros(1, dtype=np.float64)
            # Initialize a list of arrays representing extra model parameters to test 'is_list' support
            self.extra_params_ = [
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([0.3], dtype=np.float64)
            ]

        # SGD update for each sample in the mini-batch
        for xi, yi in zip(X, y):
            xi = xi.reshape(1, -1)
            # Forward pass: sigmoid
            z = xi @ self.theta_.T + self.bias_
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            # Gradient of binary cross-entropy
            error = prob.ravel()[0] - yi
            # Weight update
            self.theta_ -= self.learning_rate * error * xi
            self.bias_ -= self.learning_rate * error
            # Update the custom list of arrays to ensure their values shift and synchronize
            self.extra_params_[0] -= self.learning_rate * error * 0.1
            self.extra_params_[1] -= self.learning_rate * error * 0.2

        return self

    def predict_proba(self, X):
        """Return probability estimates for each class."""
        z = X @ self.theta_.T + self.bias_
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        prob_pos = prob_pos.ravel()
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X):
        """Return hard class predictions."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ================================================================== #
#  Data Loading                                                       #
# ================================================================== #

def get_xy(dataSet):
    """Shuffle dataset and split into features (X) and labels (y)."""
    np.random.shuffle(dataSet)
    length = np.size(dataSet, 0)
    X = dataSet[0:length, :-1]
    y = dataSet[0:length, -1].astype(int)  # flatten to 1-D for sklearn
    return X, y


# Constants
testFileName = 'SB19_CCFDUBL_TEST.csv'
trainFileName = 'SB19_CCFDUBL_TRAIN.csv'

BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 100
DEFAULT_MIN_PEERS = 2
DEFAULT_SYNC_FREQUENCY = 128
CLASSES = np.array([0, 1])


def main():
    modelName = 'fraud-detection-skl-custom'
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch = int(os.getenv('MAX_EPOCHS', str(DEFAULT_MAX_EPOCHS)))
    minPeers = int(os.getenv('MIN_PEERS', str(DEFAULT_MIN_PEERS)))
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', str(DEFAULT_SYNC_FREQUENCY)))
    os.makedirs(scratchDir, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_file = os.path.join(scratchDir, "model_output.log")
    log_f = open(log_file, 'w', buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    print('***** Starting model =', modelName)

    # ================== Load test and train Data =========================
    print('-' * 64)

    trainFile = dataDir + '/' + trainFileName
    print("loading train dataset %s .." % trainFile)
    try:
        with open(trainFile, 'r') as f:
            # first line is the header row so remove it
            trainData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
            print('size of training Data set : %s' % np.size(trainData, 0))
    except FileNotFoundError:
        print(f"Error: Train data file not found at {trainFile}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing train data file {trainFile}: {e}", file=sys.stderr)
        sys.exit(1)

    print('-' * 64)
    testFile = dataDir + '/' + testFileName
    print("loading test dataset %s .." % testFile)
    try:
        with open(testFile, 'r') as f:
            # first line is the header row so remove it
            testData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
            print('size of test Data set : %s' % np.size(testData, 0))
    except FileNotFoundError:
        print(f"Error: Test data file not found at {testFile}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing test data file {testFile}: {e}", file=sys.stderr)
        sys.exit(1)

    print('-' * 64)

    # ================== Prepare data ======================================
    x_train, y_train = get_xy(trainData)
    x_test = testData[:, :-1]
    y_test = testData[:, -1].astype(int)

    # ================== Model to train and evaluate =======================
    # Use our custom OnlineLogisticRegression instead of SGDClassifier.
    # This model uses theta_ and bias_ as weight attributes,
    # which are NOT in the built-in _SKLEARN_WEIGHT_REGISTRY.
    model = OnlineLogisticRegression(
        learning_rate=0.01,
        random_state=42,
    )
    print("Model: OnlineLogisticRegression (custom) — Logistic Regression via SGD")
    print("Weight attributes: ['theta_', 'bias_', 'extra_params_'] (registered via weight_attrs)")

    # ================== Swarm Callback Setup ==============================
    # Use a small sample from training data for lazy weight initialization.
    initSample = (x_train[:1], y_train[:1])

    swarmCallback = SwarmCallback(
        syncFrequency=syncFrequency,
        minPeers=minPeers,
        adsValData=(x_test, y_test),
        mergeMethod='mean',
        totalEpochs=maxEpoch,
        model=model,
        initData=initSample,
        classes=CLASSES,
        # === KEY DIFFERENCE FROM BUILT-IN EXAMPLE ===
        # Register custom weight attributes explicitly since
        # OnlineLogisticRegression is not in _SKLEARN_WEIGHT_REGISTRY.
        # Uses 'extra_params_' (list of arrays) and flags list serialization via 'weight_attrs_is_list'.
        weight_attrs=['theta_', 'bias_', 'extra_params_'],
        weight_attrs_is_list=True,
        lossFunction='log_loss',
        metricFunction='roc_auc_score',
    )

    # ================== Training Loop via Trainer =========================
    print('Starting training with SwarmSklearnTrainer ...')
    trainer = SwarmSklearnTrainer(swarmCallback=swarmCallback, model=model)
    trainer.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=maxEpoch, classes=CLASSES)
    print('Training done!')

    # ================== Evaluate ==========================================
    y_pred_proba = model.predict_proba(x_test)

    finalLoss = log_loss(y_test, y_pred_proba)
    finalAuc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print('***** Test loss:', finalLoss)
    print('***** Test auc:', finalAuc)

    # ================== Save ==============================================
    model_path = os.path.join(scratchDir, modelName + '.joblib')
    joblib.dump(model, model_path)
    print('Saved the trained model to', model_path)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()


if __name__ == '__main__':
    main()
