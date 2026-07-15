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

# evaluate_binary.py
# ------------------
# Evaluates a trained Swarm node's model against the held-out KDDTest+.txt set.
# Loads the fitted preprocessor and final_model.keras saved by train_binary.py,
# maps each of the 10 unsupervised clusters to its majority true label
# (normal/attack), and reports overall accuracy plus per-class breakdown.
import os, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, normalized_mutual_info_score
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_dir  = os.getenv('DATA_DIR',  '/tmp/nsl-kdd/data')
model_dir = os.getenv('MODEL_DIR', '/tmp/nsl-kdd/model') #loading model

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

class ClusteringLayer(layers.Layer):
    def __init__(self,n_clusters,embedding_dim,alpha=1.0,**kwargs):
        super().__init__(**kwargs)
        self.n_clusters=n_clusters; self.embedding_dim=embedding_dim; self.alpha=alpha
    def build(self,s):
        self.clusters=self.add_weight('clusters',(self.n_clusters,self.embedding_dim),
                      initializer='glorot_uniform',trainable=True); super().build(s)
    def call(self,x):
        z=tf.expand_dims(x,1); mu=tf.expand_dims(self.clusters,0)
        d=tf.reduce_sum(tf.square(z-mu),axis=2)
        q=1.0/(1.0+d/self.alpha); q=q**((self.alpha+1.0)/2.0)
        return q/tf.reduce_sum(q,axis=1,keepdims=True)
    def get_config(self):
        c=super().get_config()
        c.update({'n_clusters':self.n_clusters,'embedding_dim':self.embedding_dim,'alpha':self.alpha})
        return c

print('\n'+'='*65+'\nBINARY EVALUATION: Normal vs Attack\n'+'='*65)

with open(os.path.join(model_dir,'preprocessor.pkl'),'rb') as f: prep=pickle.load(f)
model=keras.models.load_model(
    os.path.join(model_dir,'final_model.keras'),
    custom_objects={'ClusteringLayer':ClusteringLayer})
print("Model loaded!")

ts=pd.read_csv(os.path.join(data_dir,'KDDTest+.txt'),names=col_names,low_memory=False)
ts['label']=ts['label'].map(attack_map).fillna('attack')
true_labels=ts['label'].values
ts=ts.drop(['label','difficulty'],axis=1)
X_test=prep.transform(ts).astype('float32')
print(f"Test shape: {X_test.shape}")
print(f"True distribution: {Counter(true_labels)}")

q,_=model.predict(X_test,verbose=0)
raw_preds=np.argmax(q,axis=1)

# Optimal cluster map from test set
cmap={}
for i in range(model.get_layer('clustering').n_clusters):
    idx=np.where(raw_preds==i)[0]
    cmap[i]=Counter(true_labels[idx]).most_common(1)[0][0] if len(idx)>0 else 'normal'

final_preds=np.array([cmap[int(c)] for c in raw_preds])

acc=accuracy_score(true_labels, final_preds)
nmi=normalized_mutual_info_score(true_labels, raw_preds)

normal_mask=true_labels=='normal'
attack_mask=true_labels=='attack'
normal_acc=(final_preds[normal_mask]=='normal').mean()
attack_acc=(final_preds[attack_mask]=='attack').mean()
#logging the outuput with nmi
print(f'\n{"="*65}')
print(f'BINARY CLASSIFICATION RESULTS')
print(f'{"="*65}')
print(f'Overall ACC    : {acc*100:.1f}%')

print(f'Normal  ACC    : {normal_acc*100:.1f}%  ({normal_mask.sum()} samples)')
print(f'Attack  ACC    : {attack_acc*100:.1f}%  ({attack_mask.sum()} samples)')
print(f'{"="*65}')
print(f'\nCluster Map: {cmap}')
print(f'\nPrediction distribution: {Counter(final_preds)}')
print('='*65)
