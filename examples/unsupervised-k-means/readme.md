# NSL-KDD Binary Intrusion Detection

This example demonstrates how a binary network intrusion detection model — built over the [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) using a deep autoencoder with a Deep Embedded Clustering (K-Means) head — runs and performs on the Swarm Learning platform.

The model classifies network traffic records into two classes, `normal` and `attack`, using unsupervised clustering in a learned embedding space rather than a directly supervised classifier.

To download the dataset, follow the steps below:

Click on "Download" on the official dataset page — [NSL-KDD on the Canadian Institute for Cybersecurity site](https://www.unb.ca/cic/datasets/nsl.html) — or use a verified mirror of the original files.

After downloading, copy `KDDTrain+.txt` and `KDDTest+.txt` to the workspace data directory:

```
cp /Downloads/KDDTrain+.txt /Downloads/KDDTest+.txt workspace/nsl-kdd-binary/data/
```

No file path needs to be edited in the training script — `train_binary.py` and `prepare_data.py` both read their input location from the `DATA_DIR` environment variable, which is set automatically by the launch commands in this README.

The following describes the cluster setup for this example:

- This example uses **one SN node**. `sn1` is the name of the Docker container, running on host `10.160.0.2`.
- **Three SL and ML nodes** are manually spawned by running the `run-sl` script. Swarm training is invoked once the ML nodes are started. The SL containers are named `sl1`, `sl2`, `sl3`. The ML containers are named `ml1`, `ml2`, `ml3`.
- All three SL/ML pairs run on the **same host**, `10.160.0.2` — this example simulates a 3-node swarm on a single GCP Compute Engine VM, rather than across separate physical hosts.
- This example assumes the HPE AutoPass License Server (APLS) is already running on `10.160.0.2`. All Swarm nodes connect to the License Server on its default port, `5814`.
- Each of the three nodes trains on a different, deliberately non-IID partition of `KDDTrain+.txt`:
  - **Node 1** — normal-dominant (65% normal / 35% attack)
  - **Node 2** — attack-dominant (65% attack / 35% normal)
  - **Node 3** — balanced (50% / 50%)

## Running the NSL-KDD Binary example

1. Navigate to the `swarm-learning` folder (that is, parent to the `examples` directory).

```
cd swarm-learning
```

2. Create the workspace directory structure.

```
mkdir -p workspace/nsl-kdd-binary/{model,data,ml-context}
mkdir -p workspace/nsl-kdd-binary/hpe-swarm-sl1
mkdir -p workspace/nsl-kdd-binary/hpe-swarm-sl2
mkdir -p workspace/nsl-kdd-binary/hpe-swarm-sl3
mkdir -p workspace/nsl-kdd-binary/saved_models/node1
mkdir -p workspace/nsl-kdd-binary/saved_models/node2
mkdir -p workspace/nsl-kdd-binary/saved_models/node3
```

3. Copy the `gen-cert` utility into the workspace and run it once per node index to generate certificates for each Swarm component, using `gen-cert -e <EXAMPLE-NAME> -i <HOST-INDEX>`:

```
cp examples/utils/gen-cert workspace/nsl-kdd-binary/
chmod +x workspace/nsl-kdd-binary/gen-cert
```

```
./gen-cert -e nsl-kdd-binary -i 1
./gen-cert -e nsl-kdd-binary -i 2
./gen-cert -e nsl-kdd-binary -i 3
```

Since all three nodes run on the same host in this setup, there is no need to copy CA certificates between separate hosts — all generated certificates already share the same `workspace/nsl-kdd-binary/cert/` directory.

4. Split the raw NSL-KDD training set into three non-IID partitions, one per node.

```
python3 workspace/nsl-kdd-binary/model/prepare_data.py
```

This reads `KDDTrain+.txt` from the data directory and writes `node1/KDDTrain+.txt`, `node2/KDDTrain+.txt`, and `node3/KDDTrain+.txt`, each with a different normal/attack ratio as described above.

5. Copy the Swarm Learning wheel file into the build context and build the Docker image for ML, containing the environment used to run Swarm training of the user model.

```
cp -L lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl workspace/nsl-kdd-binary/ml-context/
docker build -t nsl-kdd-binary-env workspace/nsl-kdd-binary/ml-context
```

You may need to specify the correct `https_proxy` for the Docker build if you are behind a firewall. For example:

```
docker build -t nsl-kdd-binary-env --build-arg https_proxy=http://<your-proxy-server-ip>:<port> workspace/nsl-kdd-binary/ml-context
```
Set the following variables according to your environment:

```bash
HOST_1_IP=<host-ip-address>
APLS_IP=${HOST_1_IP}
SN_IP=${HOST_1_IP}
SN_API_PORT=30304
```

6. Run the Swarm Network node (sentinel node).

```
./scripts/bin/run-sn -d --rm --name=sn1 --host-ip=${HOST_1_IP} \
--sentinel --sn-api-port=${SN_API_PORT} --key=workspace/nsl-kdd-binary/cert/sn-1-key.pem \
--cert=workspace/nsl-kdd-binary/cert/sn-1-cert.pem \
--capath=workspace/nsl-kdd-binary/cert/ca/capath --apls-ip=${APLS_IP}
```

Use the Docker logs command to monitor this sentinel SN node and wait for the node to finish initializing. The sentinel node is ready when this message appears in the log output:

```
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

7. Run Swarm Learning node 1 and Machine Learning node 1 (as a side-car). Set the proxy server as appropriate.

```
./scripts/bin/run-sl --name=sl1 --host-ip=${HOST_1_IP} \
--sn-ip=${SN_IP} --sn-api-port=${SN_API_PORT} --sl-fs-port=16000 \
--key=workspace/nsl-kdd-binary/cert/sl-1-key.pem \
--cert=workspace/nsl-kdd-binary/cert/sl-1-cert.pem \
--capath=workspace/nsl-kdd-binary/cert/ca/capath \
--volume=workspace/nsl-kdd-binary/hpe-swarm-sl1:/tmp/hpe-swarm \
--ml-image=nsl-kdd-binary-env --ml-name=ml1 \
--ml-w=/tmp/nsl-kdd --ml-entrypoint=python3 --ml-cmd=model/train_binary.py \
--ml-v=workspace/nsl-kdd-binary/model:/tmp/nsl-kdd/model \
--ml-v=workspace/nsl-kdd-binary/saved_models/node1:/tmp/nsl-kdd/model_out \
--ml-v=workspace/nsl-kdd-binary/data/node1:/tmp/nsl-kdd/data \
--ml-e DATA_DIR=/tmp/nsl-kdd/data --ml-e MODEL_DIR=/tmp/nsl-kdd/model_out \
--ml-e MAX_EPOCHS=50 --ml-e MIN_PEERS=3 \
--ml-e https_proxy=http://<your-proxy-server-ip>:<port-number> \
--apls-ip=${APLS_IP}
```

8. Run Swarm Learning node 2 and Machine Learning node 2 (as a side-car). Set the proxy server as appropriate.

```
./scripts/bin/run-sl --name=sl2 --host-ip=${HOST_1_IP} \
--sn-ip=${SN_IP} --sn-api-port=${SN_API_PORT} --sl-fs-port=17000 \
--key=workspace/nsl-kdd-binary/cert/sl-2-key.pem \
--cert=workspace/nsl-kdd-binary/cert/sl-2-cert.pem \
--capath=workspace/nsl-kdd-binary/cert/ca/capath \
--volume=workspace/nsl-kdd-binary/hpe-swarm-sl2:/tmp/hpe-swarm \
--ml-image=nsl-kdd-binary-env --ml-name=ml2 \
--ml-w=/tmp/nsl-kdd --ml-entrypoint=python3 --ml-cmd=model/train_binary.py \
--ml-v=workspace/nsl-kdd-binary/model:/tmp/nsl-kdd/model \
--ml-v=workspace/nsl-kdd-binary/saved_models/node2:/tmp/nsl-kdd/model_out \
--ml-v=workspace/nsl-kdd-binary/data/node2:/tmp/nsl-kdd/data \
--ml-e DATA_DIR=/tmp/nsl-kdd/data --ml-e MODEL_DIR=/tmp/nsl-kdd/model_out \
--ml-e MAX_EPOCHS=50 --ml-e MIN_PEERS=3 \
--ml-e https_proxy=http://<your-proxy-server-ip>:<port-number> \
--apls-ip=${APLS_IP}
```

9. Run Swarm Learning node 3 and Machine Learning node 3 (as a side-car). Set the proxy server as appropriate.

```
./scripts/bin/run-sl --name=sl3 --host-ip=${HOST_1_IP} \
--sn-ip=${SN_IP} --sn-api-port=${SN_API_PORT} --sl-fs-port=18000 \
--key=workspace/nsl-kdd-binary/cert/sl-3-key.pem \
--cert=workspace/nsl-kdd-binary/cert/sl-3-cert.pem \
--capath=workspace/nsl-kdd-binary/cert/ca/capath \
--volume=workspace/nsl-kdd-binary/hpe-swarm-sl3:/tmp/hpe-swarm \
--ml-image=nsl-kdd-binary-env --ml-name=ml3 \
--ml-w=/tmp/nsl-kdd --ml-entrypoint=python3 --ml-cmd=model/train_binary.py \
--ml-v=workspace/nsl-kdd-binary/model:/tmp/nsl-kdd/model \
--ml-v=workspace/nsl-kdd-binary/saved_models/node3:/tmp/nsl-kdd/model_out \
--ml-v=workspace/nsl-kdd-binary/data/node3:/tmp/nsl-kdd/data \
--ml-e DATA_DIR=/tmp/nsl-kdd/data --ml-e MODEL_DIR=/tmp/nsl-kdd/model_out \
--ml-e MAX_EPOCHS=50 --ml-e MIN_PEERS=3 \
--ml-e https_proxy=http://<your-proxy-server-ip>:<port-number> \
--apls-ip=${APLS_IP}
```

10. Three nodes of Swarm training are now started. Monitor the Docker logs of the ML nodes (`ml1`, `ml2`, `ml3` containers) for Swarm training progress:

```
docker logs -f ml1
docker logs -f ml2
docker logs -f ml3
```

Training ends with each node logging its final epoch and cluster distribution, for example:

```
Ep  50/50 | Loss:0.2506 | NMI:0.3461
```

Each node's final Swarm model, encoder, preprocessing pipeline, and cluster map are saved inside that node's own model directory — `workspace/nsl-kdd-binary/saved_models/node{1,2,3}/`. SL and ML nodes exit but are not removed after Swarm training completes.

> Rather than running steps 6–9 by hand each time, the included `workspace/nsl-kdd-binary/run.sh` script automates the full sequence above — cleanup, SN startup, launching all 3 SL/ML pairs, and live log monitoring — and can be run with:
>
> ```
> MAX_EPOCHS=50 bash workspace/nsl-kdd-binary/run.sh
> ```

## Evaluating the trained model

To evaluate a node's trained model against the held-out `KDDTest+.txt` set:

```
docker run --rm \
  -v $(pwd)/workspace/nsl-kdd-binary/saved_models/node1:/tmp/nsl-kdd/model \
  -v $(pwd)/workspace/nsl-kdd-binary/data:/tmp/nsl-kdd/data \
  -e DATA_DIR=/tmp/nsl-kdd/data \
  -e MODEL_DIR=/tmp/nsl-kdd/model \
  nsl-kdd-binary-env \
  python3 /tmp/nsl-kdd/model/evaluate_binary.py
```

This reports overall accuracy along with accuracy broken out separately for the `normal` and `attack` classes.

## Cleaning up

To clean up, run the `scripts/bin/stop-swarm` script to stop and remove the container nodes of the previous run. If required, back up the container logs and delete the workspace directory.

```
./scripts/bin/stop-swarm
```
## References

- Xie, J., Girshick, R., & Farhadi, A. (2016). **Unsupervised Deep Embedding for Clustering Analysis.** *Proceedings of the 33rd International Conference on Machine Learning (ICML).* https://arxiv.org/abs/1511.06335
- Maziar M. Fard. **Deep k-Means Reference Implementation (GitHub).** https://github.com/MaziarMF/deep-k-means

- NSL-KDD Dataset: Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009). **A Detailed Analysis of the KDD CUP 99 Data Set.** *IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA).* https://www.unb.ca/cic/datasets/nsl.html
