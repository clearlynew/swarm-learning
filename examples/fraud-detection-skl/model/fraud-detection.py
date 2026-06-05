############################################################################
## (C)Copyright 2021-2025 Hewlett Packard Enterprise Development LP
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
# Integration uses Approach 1 (Direct Callback):
#   1. Initialize model
#   2. Create SwarmCallback with initData
#   3. swCb.on_train_begin()   ← initial sync
#   4. Manual batch loop with partial_fit + on_batch_end
#   5. swCb.on_epoch_end()
#   6. swCb.on_train_end()     ← final merge
##################################################################

import os
import numpy as np
import csv
import logging
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
from swarmlearning.sklearn import SwarmCallback, SwarmSklearnTrainer


def getXY(dataSet):
    """Shuffle dataset and split into features (X) and labels (y)."""
    np.random.shuffle(dataSet)
    length = np.size(dataSet, 0)
    X = dataSet[0:length, :-1]
    y = dataSet[0:length, -1].astype(int)  # flatten to 1-D for sklearn
    return X, y


# Constants
testFileName = 'SB19_CCFDUBL_TEST.csv'
trainFileName = 'SB19_CCFDUBL_TRAIN.csv'

batchSize = 32
defaultMaxEpoch = 100
defaultMinPeers = 2


def main():
    modelName = 'fraud-detection-skl'
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)

    log_file = os.path.join(scratchDir, "model_output.log")
    import sys
    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

    # ================== Load test and train Data =========================
    print('-' * 64)

    trainFile = dataDir + '/' + trainFileName
    print("loading train dataset %s .." % trainFile)
    with open(trainFile, 'r') as f:
        # first line is the header row so remove it
        trainData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
        print('size of training Data set : %s' % np.size(trainData, 0))

    print('-' * 64)
    testFile = dataDir + '/' + testFileName
    print("loading test dataset %s .." % testFile)
    with open(testFile, 'r') as f:
        # first line is the header row so remove it
        testData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
        print('size of test Data set : %s' % np.size(testData, 0))

    print('-' * 64)

    # ================== Prepare data ======================================
    x_train, y_train = getXY(trainData)
    x_test, y_test = getXY(testData)

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
        syncFrequency=128,
        minPeers=minPeers,
        adsValData=(x_test, y_test),
        mergeMethod='mean',
        totalEpochs=maxEpoch,
        model=model,
        initData=initSample,
        classes=np.array([0, 1]),
        lossFunction='log_loss',
        metricFunction='roc_auc_score',
    )

    # ================== Step 3-6: Training Loop via Trainer ===============
    print('Starting training with SwarmSklearnTrainer ...')
    trainer = SwarmSklearnTrainer(swarmCallback=swarmCallback, model=model)
    trainer.fit(x_train, y_train, batch_size=batchSize, epochs=maxEpoch, classes=np.array([0, 1]))
    print('Training done!')

    # ================== Evaluate ==========================================
    y_pred_proba = model.predict_proba(x_test)
    y_pred = model.predict(x_test)

    finalLoss = log_loss(y_test, y_pred_proba)
    finalAuc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print('***** Test loss:', finalLoss)
    print('***** Test auc:', finalAuc)

    # ================== Save ==============================================
    model_path = os.path.join(scratchDir, modelName + '.joblib')
    joblib.dump(model, model_path)
    print('Saved the trained model to', model_path)


if __name__ == '__main__':
    main()
