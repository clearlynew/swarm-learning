############################################################################
## (C)Copyright 2026 Hewlett Packard Enterprise Development LP
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


# ---------------
# Prepares the NSL-KDD dataset for 3-node Swarm Learning training by splitting
# the full KDDTrain+.txt into 3 deliberately non-IID (non-identically distributed)
# partitions — one per Swarm node. Non-IID splits simulate realistic federated
# learning conditions where each participant's local data has a meaningfully
# different class distribution, rather than each node holding a random,
# representative sample of the same overall population.
#
# Data distribution strategy:
#   Node 1 — Normal-dominant  : 65% normal / 35% attack  (13,000 + 7,000 rows)
#   Node 2 — Attack-dominant  : 65% attack / 35% normal  (13,000 + 7,000 rows)
#   Node 3 — Balanced         : 50% normal / 50% attack  (10,000 + 10,000 rows)
#
# This tests whether Swarm Learning's weight-averaging can produce a robust
# global model even when individual nodes have strongly skewed local views
# of the data — without any node ever sharing its raw network logs.
import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "/platform/swarmml/data")

# NSL-KDD column schema — the raw .txt files have no header row, so column
# names must be supplied manually. 41 network-traffic features + label + difficulty
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

ATTACK_MAP = {attack: "attack" for attack in [
    "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
    "apache2", "processtable", "udpstorm", "ipsweep", "nmap", "portsweep",
    "satan", "mscan", "saint", "ftp_write", "guess_passwd", "imap",
    "multihop", "phf", "spy", "warezclient", "warezmaster", "sendmail",
    "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop",
    "httptunnel", "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm"
]}
ATTACK_MAP["normal"] = "normal"

print("Loading full training data...")

df = pd.read_csv(
    os.path.join(DATA_DIR, "KDDTrain+.txt"),
    names=COL_NAMES,
    low_memory=False
)

df["label"] = df["label"].map(ATTACK_MAP).fillna("attack")
df = (
    df[df["label"].isin(["normal", "attack"])]
    .drop(columns="difficulty")
    .reset_index(drop=True)
)

print(f"Total rows: {len(df)}")
print(df["label"].value_counts())

normal_df = df.query("label == 'normal'")
attack_df = df.query("label == 'attack'")

configs = [
    (13000, 7000, 1),
    (7000, 13000, 2),
    (10000, 10000, 3),
]

for idx, (normal_n, attack_n, seed) in enumerate(configs, start=1):

    node = pd.concat([
        normal_df.sample(normal_n, random_state=seed),
        attack_df.sample(attack_n, random_state=seed)
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    out_dir = os.path.join(DATA_DIR, f"node{idx}")
    os.makedirs(out_dir, exist_ok=True)

    node.to_csv(os.path.join(out_dir, "KDDTrain+.txt"), index=False)

    print(
        f"Node {idx}: {len(node)} rows | "
        f"{node['label'].value_counts().to_dict()}"
    )

print("Done!")
