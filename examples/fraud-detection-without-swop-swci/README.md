# 1. Clone Project Repository into Workspace

```bash
cd ~/swarm-learning/workspace/
git clone https://github.com/clearlynew/Fraud-Detection-Without-SWOP-SWCI.git fraud-detection
```

---

# 2. Generate Certificates

```bash
cd ~/swarm-learning/
cp -r examples/utils/gen-cert workspace/fraud-detection/

./workspace/fraud-detection/gen-cert -e fraud-detection -i 1
./workspace/fraud-detection/gen-cert -e fraud-detection -i 2
```

---

# 3. Delete certificates with "swop" and "swci" in their name

```bash
cd workspace/fraud-detection/cert
rm swop-* swci-*
cd ../../../
```

---

# 4. Create Docker Network (if not already created)

```bash
docker network create host-1-net
```

---

# 5. Create Separate Mount Directory

```bash
mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl1
mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl2

chmod -R 777 ~/swarm-learning/workspace/fraud-detection/tmp
```

---

# 6. Copy SwarmLearning Wheel and delete duplicate

```bash
cp ~/swarm-learning/lib/swarmlearning-*.whl \
~/swarm-learning/workspace/fraud-detection/ml-context/

rm workspace/fraud-detection/ml-context/swarmlearning-client-*.whl 2>/dev/null

```

---

# 7. Build ML Docker Image

```bash
docker build -t fraud-ml-env ~/swarm-learning/workspace/fraud-detection/ml-context
```

---

# 8. Run APLS (only if not running or not connected)

```bash
docker run -d \
--name apls \
--network host-1-net \
-v apls-volume:/hpe \
-p 5814:5814 \
--restart unless-stopped \
hub.myenterpriselicense.hpe.com/hpe_eval/autopass/apls:9.19
```

---

# Set Environment Variables (according to hostname -I)

```bash
export HOST_IP=172.1.1.1
export SN_IP=172.1.1.1
export APLS_IP=172.1.1.1
export SN_API_PORT=30304
```

---

# 9. Run SN (Swarm Network Node)

```bash
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

---

# 10. Monitor SN until ready

```bash
docker logs -f sn1
```

Wait until you see:

```
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

---

# 11. Run SL1

```bash
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

---

# 12. Run SL2

```bash
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

---

# 13. Monitor Training

```bash
docker logs -f sl1
docker logs -f sl2
```
