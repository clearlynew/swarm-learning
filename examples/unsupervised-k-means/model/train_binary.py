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

##################################################################
# NSL-KDD Binary Intrusion Detection using Deep Embedded
# Clustering (DEC) with HPE Swarm Learning.
#
# This example demonstrates unsupervised binary classification
# (normal vs attack) on the NSL-KDD network intrusion dataset
# using a Deep Autoencoder combined with a K-Means clustering
# head, trained collaboratively across 3 Swarm nodes without
# any node sharing its raw network traffic data.
#
# Model Architecture:
#   - Encoder : Dense(128) -> BN -> ReLU -> Dropout(0.2)
#               -> Dense(64) -> BN -> ReLU -> Dense(32, linear)
#   - Decoder : Dense(64) -> Dense(128) -> Dense(input_dim)
#   - Clustering head: ClusteringLayer implementing Student's
#     t-distribution soft assignment (K=10 clusters).
#
# Training pipeline:
#   Phase 1 - Autoencoder pretraining (MSE reconstruction loss)
#   Phase 2 - Deep K-Means via joint KLD + MSE optimization,
#             Swarm-synchronized via SwarmCallback
#
# Swarm Learning integration note:
#   SwarmCallback must wrap a SINGLE continuous model.fit() call.
#   Calling fit() multiple times re-triggers on_train_end() each
#   time, signaling the SL container that training is complete
#   prematurely and causing a broken-pipe error on the next sync.
#   The periodic target-distribution refresh required by DEC is
#   handled inside DeepKMeansCallback using a tf.data.Dataset
#   generator, allowing P to update each epoch within one fit().
#
# Reference:
#   Xie, J., Girshick, R., & Farhadi, A. (2016).
#   "Unsupervised Deep Embedding for Clustering Analysis."
#   ICML 2016. https://arxiv.org/abs/1511.06335
##################################################################
import os, logging, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.cluster import KMeans
from collections import Counter
from imblearn.over_sampling import SMOTE

def safe_save(model, path):
    try: model.save(path)
    except: pass

def safe_dump(obj, path):
    try:
        with open(path, 'wb') as f: pickle.dump(obj, f)
    except: pass

data_dir   = os.getenv('DATA_DIR',   '/platform/swarmml/data')
model_dir  = os.getenv('MODEL_DIR',  '/platform/swarmml/model')
max_epochs = int(os.getenv('MAX_EPOCHS', '50'))
min_peers  = int(os.getenv('MIN_PEERS',  '3'))

PROTOCOL_CATS = ['icmp','tcp','udp']
SERVICE_CATS  = ['IRC','X11','Z39_50','aol','auth','bgp','courier','csnet_ns',
                 'ctf','daytime','discard','domain','domain_u','echo','eco_i',
                 'ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest',
                 'hostnames','http','http_2784','http_443','http_8001','imap4',
                 'iso_tsap','klogin','kshell','ldap','link','login','mtp','name',
                 'netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private',
                 'red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc',
                 'supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois']
FLAG_CATS     = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']

col_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

attack_map = {
    'normal':'normal',
    'back':'attack','land':'attack','neptune':'attack','pod':'attack',
    'smurf':'attack','teardrop':'attack','mailbomb':'attack',
    'apache2':'attack','processtable':'attack','udpstorm':'attack',
    'ipsweep':'attack','nmap':'attack','portsweep':'attack',
    'satan':'attack','mscan':'attack','saint':'attack',
    'ftp_write':'attack','guess_passwd':'attack','imap':'attack',
    'multihop':'attack','phf':'attack','spy':'attack',
    'warezclient':'attack','warezmaster':'attack','sendmail':'attack',
    'named':'attack','snmpgetattack':'attack','snmpguess':'attack',
    'xlock':'attack','xsnoop':'attack','httptunnel':'attack',
    'buffer_overflow':'attack','loadmodule':'attack','perl':'attack',
    'rootkit':'attack','ps':'attack','sqlattack':'attack','xterm':'attack'
}

print("Loading dataset...")
fp = os.path.join(data_dir, 'KDDTrain+.txt')
with open(fp) as f: first = f.readline().strip()

if first.startswith('duration'):
    df    = pd.read_csv(fp, low_memory=False)
    y_str = df['label'].values
    df    = df.drop(['label'], axis=1)
else:
    df          = pd.read_csv(fp, names=col_names, low_memory=False)
    df['label'] = df['label'].map(attack_map).fillna('attack')
    y_str       = df['label'].values
    df          = df.drop(['label','difficulty'], axis=1)

print(f"Shape: {df.shape}")
print(f"Labels: {Counter(y_str)}")

cat_cols = ['protocol_type','service','flag']
num_cols = [c for c in df.columns if c not in cat_cols]

prep = ColumnTransformer([
    ('cat', OneHotEncoder(
        categories=[PROTOCOL_CATS,SERVICE_CATS,FLAG_CATS],
        sparse_output=False, handle_unknown='ignore'
    ), cat_cols),
    ('num', StandardScaler(), num_cols)
])
X = prep.fit_transform(df).astype('float32')
os.makedirs(model_dir, exist_ok=True)
safe_dump(prep, os.path.join(model_dir, 'preprocessor.pkl'))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

in_dim  = X.shape[1]
enc_dim = 32
BS      = 256
K       = 10  # 10 clusters for binary: some normal, some attack

ei  = keras.Input(shape=(in_dim,))
e   = layers.Dense(128)(ei); e = layers.BatchNormalization()(e); e = layers.Activation('relu')(e); e = layers.Dropout(0.2)(e)
e   = layers.Dense(64)(e);   e = layers.BatchNormalization()(e); e = layers.Activation('relu')(e)
emb = layers.Dense(enc_dim, activation='linear')(e)
encoder = Model(ei, emb, name='encoder')

di  = keras.Input(shape=(enc_dim,))
d   = layers.Dense(64, activation='relu')(di)
d   = layers.Dense(128, activation='relu')(d)
rec = layers.Dense(in_dim, activation='linear')(d)
decoder = Model(di, rec, name='decoder')

ai          = keras.Input(shape=(in_dim,))
autoencoder = Model(ai, decoder(encoder(ai)))

class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, embedding_dim, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters=n_clusters; self.embedding_dim=embedding_dim; self.alpha=alpha
    def build(self, s):
        self.clusters = self.add_weight('clusters',(self.n_clusters,self.embedding_dim),
                        initializer='glorot_uniform',trainable=True)
        super().build(s)
    def call(self, x):
        z=tf.expand_dims(x,1); mu=tf.expand_dims(self.clusters,0)
        d=tf.reduce_sum(tf.square(z-mu),axis=2)
        q=1.0/(1.0+d/self.alpha); q=q**((self.alpha+1.0)/2.0)
        return q/tf.reduce_sum(q,axis=1,keepdims=True)
    def get_config(self):
        c=super().get_config()
        c.update({'n_clusters':self.n_clusters,'embedding_dim':self.embedding_dim,'alpha':self.alpha})
        return c

clust_out = ClusteringLayer(K, enc_dim, name='clustering')(encoder(ai))
model     = Model(ai, [clust_out, decoder(encoder(ai))])

# ── PHASE 1: Pretrain ────────────────────────────────────────────────────
print('\n'+'='*60+'\nPHASE 1: AUTOENCODER PRETRAINING\n'+'='*60)
autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
autoencoder.fit(X, X, epochs=30, batch_size=BS, validation_split=0.1, verbose=1)
safe_save(encoder, os.path.join(model_dir, 'encoder.keras'))

# ── SMOTE BALANCING ──────────────────────────────────────────────────────
print('\n'+'='*60+'\nSMOTE BALANCING\n'+'='*60)
le    = LabelEncoder()
y_int = le.fit_transform(y_str)
counts = Counter(y_int)
print(f"Before SMOTE: {dict(zip(le.classes_, counts.values()))}")

target   = min(max(counts.values()), 15000)
sampling = {cls: max(cnt, target) for cls, cnt in counts.items()}
try:
    smote = SMOTE(sampling_strategy=sampling, random_state=42,
                  k_neighbors=min(3, min(counts.values())-1))
    X_bal, y_bal = smote.fit_resample(X, y_int)
    print(f"After SMOTE: {Counter(y_bal)}")
except Exception as ex:
    print(f"SMOTE skipped: {ex}")
    X_bal, y_bal = X, y_int

# ── PHASE 2: Deep K-Means (FIXED: ONE continuous fit() call) ─────────────
# Root cause of earlier "Broken pipe" failures: Keras fires on_train_end()
# exactly once per model.fit() call, no matter how many epochs that call
# covers. SwarmCallback hooks on_train_end() to tell the SL container this
# node is done. Calling fit() more than once (whether once per epoch, or
# once per block of 5 epochs) re-fires on_train_end() each time, and after
# the first firing the SL container begins its shutdown handshake — so the
# *next* fit() call's attempt to sync hits a pipe that's already closing.
#
# Fix: call model.fit() EXACTLY ONCE for the full max_epochs. The K-Means
# target distribution P (which DEC needs refreshed periodically from the
# current Q) is refreshed via a callback's on_epoch_begin, feeding a
# tf.data.Dataset generator that always reads the latest P from a mutable
# dict. This was verified standalone before use here: a single fit() call
# can refresh epoch-to-epoch targets and on_train_end still fires only once
# at the true end of training.
print('\n'+'='*60+f'\nPHASE 2: DEEP K-MEANS (K={K}, Binary)\n'+'='*60)
embs = encoder.predict(X_bal, verbose=0)
km   = KMeans(n_clusters=K, n_init=20, random_state=42).fit(embs)
model.get_layer('clustering').set_weights([km.cluster_centers_])

n_samples = len(X_bal)
steps_per_epoch = (n_samples + BS - 1) // BS
total_steps = steps_per_epoch * max_epochs
lr = keras.optimizers.schedules.CosineDecay(1e-3, total_steps, alpha=1e-5)
model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss=['kld', 'mse'], loss_weights=[1.0, 0.5]
)

def target_dist(q):
    w = q**2 / q.sum(axis=0)
    return (w.T / w.sum(axis=1)).T

# Mutable container holding the current target distribution P. Updated by
# the callback at the start of every epoch; read fresh by the generator
# that feeds each batch within that epoch.
_state = {'p': None}

q0, _ = model.predict(X_bal, verbose=0)
_state['p'] = target_dist(q0).astype('float32')

def batch_gen():
    idx = np.arange(n_samples)
    for start in range(0, n_samples, BS):
        end = min(start + BS, n_samples)
        b = idx[start:end]
        yield X_bal[b], (_state['p'][b], X_bal[b])

def make_epoch_dataset():
    return tf.data.Dataset.from_generator(
        batch_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, in_dim), dtype=tf.float32),
            (tf.TensorSpec(shape=(None, K), dtype=tf.float32),
             tf.TensorSpec(shape=(None, in_dim), dtype=tf.float32)),
        )
    )

# Infinite dataset: a fresh pass over batch_gen() each "epoch" the
# Keras training loop asks for, so the generator picks up whatever
# _state['p'] currently holds (refreshed by the callback below) at
# the moment that epoch's batches are pulled.
train_ds = tf.data.Dataset.range(1).flat_map(lambda _: make_epoch_dataset()).repeat()

try:
    from swarmlearning.tf import SwarmCallback
    swCb = SwarmCallback(syncFrequency=512, minPeers=min_peers, useAdaptiveSync=False)
    swCb.logger.setLevel(logging.DEBUG)
    print('SwarmCallback OK — syncFrequency=512')
except Exception as ex:
    swCb = None
    print(f'Local mode: {ex}')

best_nmi = 0.0
LOG_EVERY = 5

class DeepKMeansCallback(keras.callbacks.Callback):
    """Refreshes the K-Means target distribution P at the start of each
    epoch and logs NMI / checkpoints the best model at the end of each
    epoch. Does NOT call fit() itself and does NOT trigger any extra
    on_train_begin/on_train_end firings — those happen exactly once for
    the whole training run, driven by Keras itself."""

    def on_epoch_begin(self, epoch, logs=None):
        q, _ = self.model.predict(X_bal, batch_size=BS, verbose=0)
        _state['p'] = target_dist(q).astype('float32')

    def on_epoch_end(self, epoch, logs=None):
        global best_nmi
        q_eval, _ = self.model.predict(X, batch_size=BS, verbose=0)
        pr = np.argmax(q_eval, axis=1)
        nmi = normalized_mutual_info_score(y_int, pr)

        if nmi > best_nmi:
            best_nmi = nmi
            safe_save(self.model, os.path.join(model_dir, 'best_model.keras'))

        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0 or (epoch + 1) == max_epochs:
            loss_val = logs.get('loss', float('nan')) if logs else float('nan')
            dist = ' | '.join([f'C{c}:{n}' for c, n in zip(*np.unique(pr, return_counts=True))])
            print(f'  Ep {epoch+1:3d}/{max_epochs} | Loss:{loss_val:.4f} | NMI:{nmi:.4f}\n      {dist}')

cbs = [DeepKMeansCallback()]
if swCb is not None:
    cbs.append(swCb)

print(f'\nTraining ({max_epochs} epochs, ONE continuous fit() call, '
      f'{steps_per_epoch} steps/epoch)...\n')

model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=max_epochs,
    verbose=0,
    callbacks=cbs
)

safe_save(model, os.path.join(model_dir, 'final_model.keras'))

# ── Cluster Map ──────────────────────────────────────────────────────────
q_f, _      = model.predict(X, batch_size=BS, verbose=0)
final_preds = np.argmax(q_f, axis=1)
cluster_map = {}
for i in range(K):
    idx = np.where(final_preds == i)[0]
    cluster_map[i] = Counter(y_str[idx]).most_common(1)[0][0] if len(idx) > 0 else 'normal'

print('\nFinal Cluster Map:')
for k, v in cluster_map.items():
    print(f"  C{k} -> {v.upper()}")

safe_dump(cluster_map, os.path.join(model_dir, 'cluster_map.pkl'))
print('='*60+'\nDONE\n'+'='*60)
