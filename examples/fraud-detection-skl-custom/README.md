Credit card fraud detection — Custom Scikit-Learn Model
=======================================================

This example runs a Credit Card Fraud Detection algorithm on the Swarm Learning platform using a **custom Scikit-Learn model** that is NOT in the built-in weight registry. It demonstrates the `weight_attrs` feature for integrating arbitrary sklearn-compatible models with Swarm Learning.

## What makes this different from `fraud-detection-skl`

| Feature | `fraud-detection-skl` | `fraud-detection-skl-custom` |
|---------|----------------------|------------------------------|
| Model | `SGDClassifier` (built-in registry) | `OnlineLogisticRegression` (custom) |
| Weight attributes | `coef_`, `intercept_` (auto-detected) | `theta_`, `bias_`, `extra_params_` (registered via `weight_attrs`) |
| Registration | Automatic via `_SKLEARN_WEIGHT_REGISTRY` | Manual via `weight_attrs=['theta_', 'bias_', 'extra_params_']` and `weight_attrs_is_list=True` |

## Custom Model: `OnlineLogisticRegression`

The model in `model/fraud-detection.py` defines a custom `OnlineLogisticRegression` class that:
- Inherits from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`
- Implements `partial_fit()` for online/incremental learning
- Implements `predict()` and `predict_proba()` for evaluation
- Stores weights as `theta_` (coefficients), `bias_` (intercept), and `extra_params_` (list of arrays) — **not** the standard sklearn names

This is integrated with Swarm Learning by passing `weight_attrs` and `weight_attrs_is_list`:
```python
swarmCallback = SwarmCallback(
    syncFrequency=128,
    minPeers=minPeers,
    model=model,
    weight_attrs=['theta_', 'bias_', 'extra_params_'],   # <-- tells Swarm which attributes to sync
    weight_attrs_is_list=True,                           # <-- signals custom list-of-arrays layout is present
    ...
)
```

This example uses a subset of the data from [1] for each node. These subset datasets are biased with respect to the class and the volume of data.
This example uses four training batches and one test batch. These files are located in the respective `examples/fraud-detection-skl-custom/data-and-scratch<n>` directories.
>  **_NOTE :_** Refer [Data license](/examples/fraud-detection-skl-custom/Data_license.md/) associated with this dataset.


The ML program, after conversion to Swarm Learning, is in `examples/fraud-detection-skl-custom/model` and is called `fraud-detection.py`.

This example shows the Swarm training of Credit Card Fraud Detection model using four Machine Learning (ML) nodes. Machine Learning nodes along with Swarm Learning (SL) nodes are automatically spawned by Swarm Operators (SWOP) nodes - all running on single host. Swarm training gets initiated by Swarm Command Interface (SWCI) node and orchestrated by one Swarm Network (SN) nodes running on the same host. This example also shows how private data, private scratch area and shared model can be mounted to Machine Learning nodes for Swarm training.



## Cluster Setup

The cluster setup for this example uses only one host, as shown in the figure below:
- host-1: 172.1.1.1

1. This example uses one Swarm Network (SN) nodes. The names of the docker containers representing this node is **sn1**. sn1 is also the Sentinel Node. sn1 runs on host 172.1.1.1.
2. Four Swarm Learning (SL) and four Machine Learning (ML) nodes are automatically spawned by Swarm Operators (SWOP) node during training and removed after training. Example uses one SWOP node that connects to the SN node. The names of the docker containers representing this SWOP node is **swop1**. swop1 runs on host 172.1.1.1.
3. Training is initiated by SWCI node (**swci1**) that runs on host 172.1.1.1
4. Example assumes that License Server already runs on host 172.1.1.1. All Swarm nodes connect to the License Server, on its default port 5814.



## Running the Custom Fraud Detection example

1. *On host-1*:
   cd to `swarm-learning` folder (i.e. parent to examples directory)
   ```
   cd swarm-learning
   ```

2. *On host-1*:
   Create a temporary `workspace` directory and copy `fraud-detection-skl-custom` example and `gen-cert` utility there as follows.
   ```
   mkdir workspace
   cp -r examples/fraud-detection-skl-custom workspace/
   cp -r examples/utils/gen-cert workspace/fraud-detection-skl-custom/
   ```

3. *On host-1*:
   Run the `gen-cert` utility to generate certificates for each Swarm component using the command: `gen-cert -e <EXAMPLE-NAME> -i <HOST-INDEX>`
   ```
   ./workspace/fraud-detection-skl-custom/gen-cert -e fraud-detection-skl-custom -i 1
   ```

4. *On host-1*:
   Create a network called `host-1-net` using docker network create command. This network will be used for SN, SWOP, SWCI, SL and user containers. Please ignore this step if this network is already created.
   ```
   docker network create host-1-net
   ```

5. *On host-1*:
   Declare and assign values to the variables like APLS_IP, HOST_IP, SN_IP and SN_API_PORT. The values mentioned here are for illustration purpose only. Use appropriate values as per your swarm network.
   ```
   APLS_IP=172.1.1.1
   HOST_IP=172.1.1.1
   SN_IP=172.1.1.1
   SN_API_PORT=30304
   ```

6. *On host-1*:
   Search and replace all occurrences of placeholders and replace them with appropriate values.
   ```
   sed -i "s+<PROJECT-MODEL>+$(pwd)/workspace/fraud-detection-skl-custom/model+g" workspace/fraud-detection-skl-custom/swci/taskdefs/swarm_fd_custom_task.yaml

   sed -i "s+<SWARM-NETWORK>+host-1-net+g" workspace/fraud-detection-skl-custom/swop/swop*_profile.yaml
   sed -i "s+<CURRENT-PATH>/examples+$(pwd)/workspace+g" workspace/fraud-detection-skl-custom/swop/swop*_profile.yaml
   sed -i "s+<LICENSE-SERVER-ADDRESS>+${APLS_IP}+g" workspace/fraud-detection-skl-custom/swop/swop*_profile.yaml
   sed -i "s+<PROJECT-CERTS>+$(pwd)/workspace/fraud-detection-skl-custom/cert+g" workspace/fraud-detection-skl-custom/swop/swop*_profile.yaml
   sed -i "s+<PROJECT-CACERTS>+$(pwd)/workspace/fraud-detection-skl-custom/cert/ca/capath+g" workspace/fraud-detection-skl-custom/swop/swop*_profile.yaml
   ```

7. *On host-1*:
   Rebuild the SwarmLearning wheel from source (requires Docker). This ensures the wheel includes any source code changes (e.g., sklearn custom model support):
   ```
   docker run --rm -v $(pwd):/workspace -w /workspace/lib/src python:3.8-slim sh -c \
     "pip install build && export SWARM_VER=client && python3 -m build --wheel -n --outdir ../ python-client"
   ```

8. *On host-1*:
   Create a docker volume and copy the rebuilt SwarmLearning wheel file there:
   ```
   docker volume rm sl-cli-lib
   docker volume create sl-cli-lib
   docker container create --name helper -v sl-cli-lib:/data hello-world
   docker cp lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data
   docker rm helper
   ```

9. *On host-1*:
   Run Swarm Network node (sn1) - sentinel node
   ```
   ./scripts/bin/run-sn -d --rm --name=sn1 --network=host-1-net --host-ip=${HOST_IP} --sentinel --sn-api-port=${SN_API_PORT}     \
   --key=workspace/fraud-detection-skl-custom/cert/sn-1-key.pem --cert=workspace/fraud-detection-skl-custom/cert/sn-1-cert.pem   \
   --capath=workspace/fraud-detection-skl-custom/cert/ca/capath --apls-ip=${APLS_IP}
   ```
   Use the docker logs command to monitor the Sentinel SN node and wait for the node to finish initializing. The Sentinel node is ready when these messages appear in the log output:
   `swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304`

10. *On host-1*:
   Run Swarm Operator node (swop1)

   Note: If required, modify proxy, according to environment, either in the below command or in the swop profile file under `workspace/fraud-detection-skl-custom/swop` folder.
   **Important:** The `--swop-uid 0` flag is required so that SWOP can access the Docker socket to build user containers.
   ```
   ./scripts/bin/run-swop -d --rm --name=swop1 --network=host-1-net --usr-dir=workspace/fraud-detection-skl-custom/swop                        \
   --profile-file-name=swop1_profile.yaml --sn-ip=${SN_IP} --sn-api-port=${SN_API_PORT} --key=workspace/fraud-detection-skl-custom/cert/swop-1-key.pem     \
   --cert=workspace/fraud-detection-skl-custom/cert/swop-1-cert.pem --capath=workspace/fraud-detection-skl-custom/cert/ca/capath                \
   -e http_proxy= -e https_proxy= --apls-ip=${APLS_IP} \
   --swop-uid 0
   ```

11. *On host-1*:
    Run Swarm Command Interface node (swci1). It will create, finalize and assign below tasks to task-framework for sequential execution –
    - user_env_skl_custom_build_task: Builds Python-based docker image for ML node to run custom model training
    - swarm_fd_custom_task: Create containers out of ML image and mount model and data path to run Swarm training

    Note: If required, modify SN IP, according to environment, in `workspace/fraud-detection-skl-custom/swci/swci-init` file
    ```
    ./scripts/bin/run-swci --rm --name=swci1 --network=host-1-net --usr-dir=workspace/fraud-detection-skl-custom/swci         \
    --init-script-name=swci-init --key=workspace/fraud-detection-skl-custom/cert/swci-1-key.pem                               \
    --cert=workspace/fraud-detection-skl-custom/cert/swci-1-cert.pem --capath=workspace/fraud-detection-skl-custom/cert/ca/capath        \
    -e http_proxy= -e https_proxy= --apls-ip=${APLS_IP}
    ```

12. *On host-1*:
    Four node Swarm training will be automatically started when the run task (swarm_fd_custom_task) gets assigned and executed. User can open a new terminal on host-1 and monitor the docker logs of ML nodes for Swarm training. Swarm training will end with the following log message at the end –
    `SwarmCallback : INFO : All peers and Swarm training rounds finished. Final Swarm model was loaded.`

    Final Swarm model will be saved in each node specific scratch directory, which is `workspace/fraud-detection-skl-custom/data-and-scratch<n>/user<n>` directory. All the dynamically spawned SL and ML nodes will exit after Swarm training. The SN and SWOP nodes continue running.

13. *On host-1*:
    To clean-up, run the `scripts/bin/stop-swarm` script on all the systems to stop and remove the container nodes of the previous run. If needed, take backup of the container logs. Finally remove docker network (`host-1-net`) and docker volume (`sl-cli-lib`) and delete the `workspace` directory.


## How to adapt this for your own custom model

To use your own custom model with Swarm Learning:

1. **Implement the sklearn interface**: Your model must have `partial_fit()`, `predict()`, and optionally `predict_proba()`.

2. **Identify weight attributes**: Determine which attributes hold the trainable parameters (e.g., `my_weights_`, `my_bias_`).

3. **Pass `weight_attrs`**:
   ```python
   swarmCallback = SwarmCallback(
       model=your_model,
       weight_attrs=['my_weights_', 'my_bias_'],
       ...
   )
   ```

4. **Use `initData`** if weights are lazily initialized (i.e., they don't exist until `partial_fit()` is called):
   ```python
   swarmCallback = SwarmCallback(
       model=your_model,
       weight_attrs=['my_weights_', 'my_bias_'],
       initData=(X_sample, y_sample),
       classes=np.array([0, 1]),
       ...
   )
   ```


## References
1. M. L. G. - ULB, "Credit Card Fraud Detection," [Online]. Available: [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
