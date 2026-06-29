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
# Fraud Detection example using Scikit-Learn with Swarm Learning.
#
# This is the Scikit-Learn equivalent of the TensorFlow-based
# fraud-detection example. It uses SGDClassifier with log_loss
# (logistic regression) to match the original single-layer
# Dense(1, activation='sigmoid') model.
#
# This example uses one of the models in _SKLEARN_WEIGHT_REGISTRY
# built-in to the Scikit-Learn Swarm client (sklearn.py). This is
# the default way as explained in the README.md file.
##################################################################

import os
import sys
import numpy as np
import csv
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
from swarmlearning.sklearn import SwarmCallback, SwarmSklearnTrainer


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
    modelName = 'fraud-detection-skl'
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
    # SGDClassifier with loss='log_loss' is equivalent to logistic regression
    # trained via SGD, matching the original TF model:
    #   Dense(1, activation='sigmoid') + binary_crossentropy + SGD optimizer
    model = SGDClassifier(
        loss='log_loss',
        learning_rate='constant',
        eta0=0.01,
        random_state=42,
    )
    print("Model: SGDClassifier(loss='log_loss') — Logistic Regression via SGD")

    # ================== Swarm Callback Setup ==============================
    # Use a small sample from training data for lazy weight initialization.
    # The dummy partial_fit() inside SwarmCallback will allocate coef_ and
    # intercept_ attributes so they exist before the first swarm sync.
    initSample = (x_train[:1], y_train[:1])

    # In SwarmCallBack following parameter is provided to enable displaying
    # training progress or ETA of training on the SLM UI.
    # 'totalEpochs' - Total epochs used in local training.
    swarmCallback = SwarmCallback(
        syncFrequency=syncFrequency,
        minPeers=minPeers,
        adsValData=(x_test, y_test),
        mergeMethod='mean',
        totalEpochs=maxEpoch,
        model=model,
        initData=initSample,
        classes=CLASSES,
        lossFunction='log_loss',
        metricFunction='roc_auc_score',
    )

    # ================== Step 3-6: Training Loop via Trainer ===============
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
