Credit card fraud detection (Without SWOP/SWCI)
================================================

This example runs a Credit Card Fraud Detection algorithm [1] on the Swarm Learning platform. It uses Keras and TensorFlow.

This example uses a subset of the data from [1] for each node. These subset datasets are biased with respect to the class and the volume of data.
This example uses two training datasets and one test dataset, located in the respective `workspace/fraud-detection/data-and-scratch<n>` directories.

> **_NOTE :_** Refer [Data license](/examples/fraud-detection/Data_license.md/) associated with this dataset.

The ML program, after conversion to Swarm Learning, is in `workspace/fraud-detection-without-swop-swci/model` and is called `fraud-detection.py`.

This example shows the Swarm training of a Credit Card Fraud Detection model using two Machine Learning (ML) nodes. Unlike the standard example, this variant runs **without SWOP or SWCI** — Swarm Learning (SL) nodes are launched directly, and training is initiated manually. All nodes run on a single host. This example also shows how private data, private scratch area and shared model can be mounted to Machine Learning nodes for Swarm training.



## Cluster Setup

The cluster setup for this example uses only one host, as shown in the figure below:
- host-1: 172.1.1.1

<img width="1376" height="783" alt="Gemini_Generated_Image_jlw7q7jlw7q7jlw7" src="https://github.com/user-attachments/assets/c30d1d1c-9407-4e2b-a019-87c1f815997b" />


1. This example uses one Swarm Network (SN) node. The name of the docker container representing this node is **sn1**. sn1 is also the Sentinel Node. sn1 runs on host 172.1.1.1.
2. Two Swarm Learning (SL) and two Machine Learning (ML) nodes are launched directly using `run-sl`. The names of the docker containers representing these nodes are **sl1** and **sl2**. Both run on host 172.1.1.1.
3. Training begins automatically once both SL nodes are up and `MIN_PEERS` is satisfied — no SWCI node is required.
4. This example assumes that a License Server (APLS) already runs on host 172.1.1.1. All Swarm nodes connect to the License Server on its default port 5814.



## Running the Credit card fraud detection example (without SWOP/SWCI)

1. *On host-1*:
   Clone the project repository into the workspace directory.
   ```
   cd ~/swarm-learning/workspace/
   git clone https://github.com/clearlynew/Fraud-Detection-Without-SWOP-SWCI.git fraud-detection
   ```

2. *On host-1*:
   Copy the `gen-cert` utility into the cloned project and generate certificates for each Swarm component.
   ```
   cd ~/swarm-learning/
   cp -r examples/utils/gen-cert workspace/fraud-detection/
   ./workspace/fraud-detection/gen-cert -e fraud-detection -i 1
   ./workspace/fraud-detection/gen-cert -e fraud-detection -i 2
   ```

3. *On host-1*:
   Remove the SWOP and SWCI certificates that were auto-generated but are not needed for this setup.
   ```
   cd workspace/fraud-detection/cert
   rm swop-* swci-*
   cd ../../../
   ```

4. *On host-1*:
   Create a network called `host-1-net` using the docker network create command. This network will be used for SN, SL, and ML containers. Please ignore this step if this network is already created.
   ```
   docker network create host-1-net
   ```

5. *On host-1*:
   Create separate temporary mount directories for each SL node and set appropriate permissions.
   ```
   mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl1
   mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl2
   chmod -R 777 ~/swarm-learning/workspace/fraud-detection/tmp
   ```

6. *On host-1*:
   Copy the SwarmLearning wheel file into the ML Docker build context and remove any duplicate wheel files.
   ```
   cp ~/swarm-learning/lib/swarmlearning-*.whl \
   ~/swarm-learning/workspace/fraud-detection/ml-context/
   rm workspace/fraud-detection/ml-context/swarmlearning-client-*.whl 2>/dev/null
   ```

7. *On host-1*:
   Build the ML Docker image that will be used to run the fraud detection model inside the ML containers.
   ```
   docker build -t fraud-ml-env ~/swarm-learning/workspace/fraud-detection/ml-context
   ```

8. *On host-1*:
   Run the APLS license server container if it is not already running or not connected.
   ```
   docker run -d \
   --name apls \
   --network host-1-net \
   -v apls-volume:/hpe \
   -p 5814:5814 \
   --restart unless-stopped \
   hub.myenterpriselicense.hpe.com/hpe_eval/autopass/apls:9.19
   ```

9. *On host-1*:
   Declare and assign values to the environment variables. The values mentioned here are for illustration purposes only. Use appropriate values as per your swarm network (check your machine's IP using `hostname -I`).
   ```
   export HOST_IP=172.1.1.1
   export SN_IP=172.1.1.1
   export APLS_IP=172.1.1.1
   export SN_API_PORT=30304
   ```

10. *On host-1*:
    Run the Swarm Network node (sn1) — this is the Sentinel node.
    ```
    cd ~/swarm-learning
    ./scripts/bin/run-sn -d --name=sn1 \
    --network=host-1-net \
    --host-ip=${HOST_IP} \
    --sentinel \
    --sn-api-port=${SN_API_PORT} \
    --key=workspace/fraud-detection/cert/sn-1-key.pem \
    --cert=workspace/fraud-detection/cert/sn-1-cert.pem \
    --capath=workspace/fraud-detection/cert/ca/capath \
    --apls-ip=${APLS_IP}
    ```
    Use the docker logs command to monitor the Sentinel SN node and wait for the node to finish initializing. The Sentinel node is ready when this message appears in the log output:
    ```
    swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
    ```

11. *On host-1*:
    Run Swarm Learning node 1 (sl1) along with its associated ML container (ml1).
    ```
    ./scripts/bin/run-sl -d --name=sl1 \
    --network=host-1-net \
    --host-ip=${HOST_IP} \
    --sn-ip=${SN_IP} \
    --sn-api-port=${SN_API_PORT} \
    --sl-fs-port=16000 \
    --key=workspace/fraud-detection/cert/sl-1-key.pem \
    --cert=workspace/fraud-detection/cert/sl-1-cert.pem \
    --capath=workspace/fraud-detection/cert/ca/capath \
    --ml-image=fraud-ml-env \
    --ml-name=ml1 \
    --ml-entrypoint=python3 \
    --ml-cmd=/tmp/test/model/fraud-detection.py \
    -v ~/workspace/fraud-detection/tmp/sl1:/tmp/hpe-swarm \
    --ml-v workspace/fraud-detection/model:/tmp/test/model \
    --ml-v workspace/fraud-detection/data-and-scratch1/app-data:/app-data \
    --ml-e DATA_DIR=/app-data \
    --ml-e SCRATCH_DIR=/tmp/scratch \
    --ml-e MIN_PEERS=2 \
    --ml-e MAX_EPOCHS=16 \
    --apls-ip=${APLS_IP}
    ```

12. *On host-1*:
    Run Swarm Learning node 2 (sl2) along with its associated ML container (ml2).
    ```
    ./scripts/bin/run-sl -d --name=sl2 \
    --network=host-1-net \
    --host-ip=${HOST_IP} \
    --sn-ip=${SN_IP} \
    --sn-api-port=${SN_API_PORT} \
    --sl-fs-port=17000 \
    --key=workspace/fraud-detection/cert/sl-2-key.pem \
    --cert=workspace/fraud-detection/cert/sl-2-cert.pem \
    --capath=workspace/fraud-detection/cert/ca/capath \
    --ml-image=fraud-ml-env \
    --ml-name=ml2 \
    --ml-entrypoint=python3 \
    --ml-cmd=/tmp/test/model/fraud-detection.py \
    -v ~/workspace/fraud-detection/tmp/sl2:/tmp/hpe-swarm \
    --ml-v workspace/fraud-detection/model:/tmp/test/model \
    --ml-v workspace/fraud-detection/data-and-scratch2/app-data:/app-data \
    --ml-e DATA_DIR=/app-data \
    --ml-e SCRATCH_DIR=/tmp/scratch \
    --ml-e MIN_PEERS=2 \
    --ml-e MAX_EPOCHS=3 \
    --apls-ip=${APLS_IP}
    ```

13. *On host-1*:
    Two-node Swarm training will begin automatically once both SL nodes are up and the peer quorum is reached. Open separate terminals to monitor the ML node logs for training progress. Swarm training will end with the following log message:
    `SwarmCallback : INFO : All peers and Swarm training rounds finished. Final Swarm model was loaded.`

    To monitor training:
    ```
    docker logs -f sl1
    docker logs -f sl2
    ```

    Final Swarm model will be saved in each node's scratch directory. To clean up, stop and remove all containers, delete the docker network (`host-1-net`) and docker volume (`sl-cli-lib`), and remove the `workspace` directory.



## References
1. M. L. G. - ULB, "Credit Card Fraud Detection," [Online]. Available: [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
