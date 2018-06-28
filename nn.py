#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 00:25:39 2018

@author: Mauricio de Oliveira

"""
#test['fault_severity'] = None
#test['location_id'] = test.location.apply(lambda x: int(x.split('location ')[1]))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read data
train = pd.read_csv('datasets/train.csv', index_col=0)
test  = pd.read_csv('datasets/test.csv',  index_col=0)
resource_type = pd.read_csv('datasets/resource_type.csv', index_col=0)
severity_type = pd.read_csv('datasets/severity_type.csv', index_col=0)
event_type = pd.read_csv('datasets/event_type.csv', index_col=0)
log_feature = pd.read_csv('datasets/log_feature.csv', index_col=0)

# make sets cleaner
test["fault_severity"] = -1
train = train.append(test)
train['location'] = train.location.apply(lambda x: int(x.split('location ')[1]))
#test['location'] = test.location.apply(lambda x: int(x.split('location ')[1]))
#train['time_id'] = train.index
resource_type['resource_type'] = resource_type.resource_type.apply(lambda x: int(x.split('resource_type ')[1]))
severity_type['severity_type'] = severity_type.severity_type.apply(lambda x: int(x.split('severity_type ')[1]))
event_type['event_type'] = event_type.event_type.apply(lambda x: int(x.split('event_type ')[1]))
log_feature['log_feature'] = log_feature.log_feature.apply(lambda x: int(x.split('feature ')[1]))
log_feature['volume'] = log_feature.volume.apply(lambda x: int(x))

# each features unique values
print("different locations: ", train.location.unique().size)
#print("different time_id: ", train.time_id.unique().size)
print("different resource_types: ", resource_type.resource_type.unique().size)
print("different severity_types: ", severity_type.severity_type.unique().size)
print("different event_types: ", event_type.event_type.unique().size)
print("different log_features: ", log_feature.log_feature.unique().size)
print("different volumes: ", log_feature.volume.unique().size)

print("max volume: ", log_feature.volume.max(), "; min volume: ", log_feature.volume.min())

#fault_sev = train.fault_severity.value_counts().plot.bar()
#fault_sev.set_title("Fault severity")
#fault_sev.set_ylabel("Training set occurrences")
#fault_sev = fault_sev.get_figure()
#fault_sev.savefig("fault_sev.png")

# analyzing feature distribution among overall data, examples, and for each fault severity
resource_train = resource_type.loc[resource_type.index.isin(train.index)]
severity_train = severity_type.loc[severity_type.index.isin(train.index)]
event_train = event_type.loc[event_type.index.isin(train.index)]
log_train = log_feature.loc[log_feature.index.isin(train.index)]

fault_sev_0 = train[train.fault_severity == 0]
fault_sev_1 = train[train.fault_severity == 1]
fault_sev_2 = train[train.fault_severity == 2]

#print("max time_id train: ", train.time_id.max(), "; min time_id: ", train.time_id.min())
#print("max time_id 0: ", fault_sev_0.time_id.max(), "; min time_id: ", fault_sev_0.time_id.min())
#print("max time_id 1: ", fault_sev_1.time_id.max(), "; min time_id: ", fault_sev_1.time_id.min())
#print("max time_id 2: ", fault_sev_2.time_id.max(), "; min time_id: ", fault_sev_2.time_id.min())

resource_train_0 = resource_type.loc[resource_type.index.isin(fault_sev_0.index)]
severity_train_0 = severity_type.loc[severity_type.index.isin(fault_sev_0.index)]
event_train_0 = event_type.loc[event_type.index.isin(fault_sev_0.index)]
log_train_0 = log_feature.loc[log_feature.index.isin(fault_sev_0.index)]

resource_train_1 = resource_type.loc[resource_type.index.isin(fault_sev_1.index)]
severity_train_1 = severity_type.loc[severity_type.index.isin(fault_sev_1.index)]
event_train_1 = event_type.loc[event_type.index.isin(fault_sev_1.index)]
log_train_1 = log_feature.loc[log_feature.index.isin(fault_sev_1.index)]

resource_train_2 = resource_type.loc[resource_type.index.isin(fault_sev_2.index)]
severity_train_2 = severity_type.loc[severity_type.index.isin(fault_sev_2.index)]
event_train_2 = event_type.loc[event_type.index.isin(fault_sev_2.index)]
log_train_2 = log_feature.loc[log_feature.index.isin(fault_sev_2.index)]

#plt.figure()
#resource_type.resource_type.value_counts().plot.bar(title = "resource_type")
#plt.figure()
#resource_train.resource_type.value_counts().plot.bar(title = "resource_train")
#plt.figure()
#resource_train_0.resource_type.value_counts().plot.bar(title = "resource_train_0")
#plt.figure()
#resource_train_1.resource_type.value_counts().plot.bar(title = "resource_train_1")
#plt.figure()
#resource_train_2.resource_type.value_counts().plot.bar(title = "resource_train_2")
#plt.figure()
#severity_type.severity_type.value_counts().plot.bar(title = "severity_type")
#plt.figure()
#severity_train.severity_type.value_counts().plot.bar(title = "severity_train")
#plt.figure()
#severity_train_0.severity_type.value_counts().plot.bar(title = "severity_train_0")
#plt.figure()
#severity_train_1.severity_type.value_counts().plot.bar(title = "severity_train_1")
#plt.figure()
#severity_train_2.severity_type.value_counts().plot.bar(title = "severity_train_2")

res_idx = resource_type.resource_type.unique()
sev_idx = severity_type.severity_type.unique()

# for event  type its best to analyse the value_counts output.

#plt.figure()
#event_type.event_type.value_counts().head(10).plot.bar(title = "event_type")
#event_type.event_type.value_counts().tail(40).plot.bar(title = "event_type")
#event_type.event_type.value_counts()
#plt.figure()
#event_train.event_type.value_counts().head(10).plot.bar(title = "event_train")
#event_type.event_type.value_counts().tail(40).plot.bar(title = "event_train")
#event_train.event_type.value_counts()
#plt.figure()
#event_train_0.event_type.value_counts().head(10).plot.bar(title = "event_train_0")
#event_type.event_type.value_counts().tail(40).plot.bar(title = "event_train_0")
#event_train_0.event_type.value_counts()
#plt.figure()
#event_train_1.event_type.value_counts().head(10).plot.bar(title = "event_train_1")
#event_type.event_type.value_counts().tail(40).plot.bar(title = "event_train_1")
#event_train_1.event_type.value_counts()
#plt.figure()
#event_train_2.event_type.value_counts().head(10).plot.bar(title = "event_train_2")
#event_type.event_type.value_counts().tail(40).plot.bar(title = "event_train_2")
#event_train_2.event_type.value_counts()
"""
event_idxs_train = event_train.event_type.value_counts().head(20).index
#event_idxs_1 = event_train_1.event_type.value_counts().head(20).index                             
#event_idxs_2 = event_train_2.event_type.value_counts().head(20).index                             
#event_idxs = event_idxs_2.append(event_idxs_train)
#event_idxs = event_idxs.append(event_idxs_1)
#event_idxs = event_idxs.drop_duplicates()
event_idxs = event_idxs_train.drop_duplicates()
"""

#event_idxs_train = event_train.event_type.value_counts().head(20).index
event_idxs_0 = event_train_0.event_type.value_counts().head(30).index
event_idxs_1 = event_train_1.event_type.value_counts().head(30).index                             
event_idxs_2 = event_train_2.event_type.value_counts().head(30).index        
event_idxs = event_idxs_2.append(event_idxs_0)                     
#event_idxs = event_idxs_2.append(event_idxs_train)
event_idxs = event_idxs.append(event_idxs_1)
event_idxs = event_idxs.drop_duplicates()


# volume

#plt.figure()
#log_feature.volume.value_counts().head(15).plot.bar(title = "log_feature (volume)")
#log_feature.volume.apply(lambda x: np.log2(x)).value_counts().head(300).plot.kde(title = "log_feature (volume)")
#plt.figure()
#log_train.volume.value_counts().head(15).plot.bar(title = "log_train (volume)")
#log_train.volume.apply(lambda x: np.log2(x)).value_counts().head(300).plot.kde(title = "log_train (volume)")
#plt.figure()
#log_train_0.volume.value_counts().head(15).plot.bar(title = "log_train_0 (volume)")
#log_train_0.volume.apply(lambda x: np.log2(x)).value_counts().head(300).plot.kde(title = "log_train_0 (volume)")
#plt.figure()
#log_train_1.volume.value_counts().head(15).plot.bar(title = "log_train_1 (volume)")
#log_train_1.volume.apply(lambda x: np.log2(x)).value_counts().head(300).plot.kde(title = "log_train_1 (volume)")
#plt.figure()
#log_train_2.volume.value_counts().head(15).plot.bar(title = "log_train_2 (volume)")
#log_train_2.volume.apply(lambda x: np.log2(x)).value_counts().head(300).plot.kde(title = "log_train_2 (volume)")
"""
vol_idxs_train = log_train.volume.value_counts().head(34).index
#vol_idxs_1 = log_train_1.volume.value_counts().head(34).index                             
#vol_idxs_2 = log_train_2.volume.value_counts().head(34).index                             
#vol_idxs = vol_idxs_2.append(vol_idxs_train)
#vol_idxs = vol_idxs.append(vol_idxs_1)
#vol_idxs = vol_idxs.drop_duplicates()
vol_idxs = vol_idxs_train.drop_duplicates()
"""

#vol_idxs_train = log_train.volume.value_counts().head(34).index
vol_idxs_0 = log_train_0.volume.value_counts().head(28).index                             
vol_idxs_1 = log_train_1.volume.value_counts().head(60).index                             
vol_idxs_2 = log_train_2.volume.value_counts().head(70).index    
vol_idxs = vol_idxs_2.append(vol_idxs_0)                         
#vol_idxs = vol_idxs_2.append(vol_idxs_train)
vol_idxs = vol_idxs.append(vol_idxs_1)
vol_idxs = vol_idxs.drop_duplicates()

#vol_12 = train[train.index.isin(log_feature.volume.isin([1,2]).index)]
#vol_12.fault_severity.value_counts().plot.bar(title = "crazy histogram")

# log feature

#plt.figure()
#log_feature.log_feature.value_counts().head(15).plot.bar(title = "log_feature")
#plt.figure()
#log_train.log_feature.value_counts().head(15).plot.bar(title = "log_train")
#plt.figure()
#log_train_0.log_feature.value_counts().head(15).plot.bar(title = "log_train_0")
#plt.figure()
#log_train_1.log_feature.value_counts().head(15).plot.bar(title = "log_train_1")
#plt.figure()
#log_train_2.log_feature.value_counts().head(15).plot.bar(title = "log_train_2")

"""
log_idxs_train = log_train.log_feature.value_counts().head(60).index
#log_idxs_1 = log_train_1.log_feature.value_counts().head(60).index                             
#log_idxs_2 = log_train_2.log_feature.value_counts().head(60).index                             
#log_idxs = log_idxs_2.append(log_idxs_train)
#log_idxs = log_idxs.append(log_idxs_1)
#log_idxs = log_idxs.drop_duplicates()
log_idxs = log_idxs_train.drop_duplicates()
"""


#log_idxs_train = log_train.log_feature.value_counts().head(60).index
log_idxs_0 = log_train_0.log_feature.value_counts().head(80).index                             
log_idxs_1 = log_train_1.log_feature.value_counts().head(100).index                             
log_idxs_2 = log_train_2.log_feature.value_counts().head(110).index         
log_idxs = log_idxs_2.append(log_idxs_0)                    
#log_idxs = log_idxs_2.append(log_idxs_train)
log_idxs = log_idxs.append(log_idxs_1)
log_idxs = log_idxs.drop_duplicates()

# location

#plt.figure()
#train.location.value_counts().head(100).plot.bar(title = "location")
#plt.figure()
#train.location.value_counts().tail(100).plot.bar(title = "location")
#plt.figure()
#fault_sev_0.location.value_counts().head(100).plot.bar(title = "location_0")
#plt.figure()
#fault_sev_1.location.value_counts().head(100).plot.bar(title = "location_1")
#plt.figure()
#fault_sev_2.location.value_counts().head(100).plot.bar(title = "location_2")

#print("different locations: ", train.location.unique().size)
#print("different locations 0: ", fault_sev_0.location.unique().size)
#print("different locations 1: ", fault_sev_1.location.unique().size)
#print("different locations 2: ", fault_sev_2.location.unique().size)
"""
loc_idxs_train = train.location.value_counts().head(350).index
#loc_idxs_1 = fault_sev_1.location.value_counts().head(100).index                             
#loc_idxs_2 = fault_sev_2.location.value_counts().head(100).index                             
#loc_idxs = loc_idxs_2.append(loc_idxs_train)
#loc_idxs = loc_idxs.append(loc_idxs_1)
#loc_idxs = loc_idxs.drop_duplicates()
loc_idxs = loc_idxs_train.drop_duplicates()
"""

#loc_idxs_train = train.location.value_counts().head(200).index
loc_idxs_0 = fault_sev_1.location.value_counts().head(400).index                             
loc_idxs_1 = fault_sev_1.location.value_counts().head(400).index                             
loc_idxs_2 = fault_sev_2.location.value_counts().head(150).index   
loc_idxs = loc_idxs_2.append(loc_idxs_0)
#loc_idxs = loc_idxs_2.append(loc_idxs_train)
loc_idxs = loc_idxs.append(loc_idxs_1)
loc_idxs = loc_idxs.drop_duplicates()   

vol_sum = log_feature.groupby(log_feature.index).volume.apply(lambda x: sum(x))
copy_vol = vol_sum
vol_sum = (vol_sum - vol_sum.min())/(vol_sum.max() - vol_sum.min())
vol_sum = vol_sum[vol_sum.index.isin(train.index)]
train = train.join(vol_sum).rename(columns={'volume': 'volume_sum'})

vol_2 = log_train_2.groupby(log_train_2.index).volume.apply(lambda x: sum(x))
vol_1 = log_train_1.groupby(log_train_1.index).volume.apply(lambda x: sum(x))
vol_0 = log_train_0.groupby(log_train_0.index).volume.apply(lambda x: sum(x))
vol_train = log_train.groupby(log_train.index).volume.apply(lambda x: sum(x))

vol_prob_0 = np.exp(-((vol_train - vol_0.mean())**2)/(2*vol_0.std()*vol_0.std()))/(np.sqrt(2*np.pi*vol_0.std()*vol_0.std()))
vol_prob_1 = np.exp(-((vol_train - vol_1.mean())**2)/(2*vol_1.std()*vol_1.std()))/(np.sqrt(2*np.pi*vol_1.std()*vol_1.std()))
vol_prob_2 = np.exp(-((vol_train - vol_2.mean())**2)/(2*vol_2.std()*vol_2.std()))/(np.sqrt(2*np.pi*vol_2.std()*vol_2.std()))

#vol_prob_0 = (vol_prob_0 - vol_prob_0.min())/(vol_prob_0.max() - vol_prob_0.min())
#vol_prob_1 = (vol_prob_1 - vol_prob_1.min())/(vol_prob_1.max() - vol_prob_1.min())
#vol_prob_2 = (vol_prob_2 - vol_prob_2.min())/(vol_prob_2.max() - vol_prob_2.min())

vol_prob_0 = vol_prob_0 * 0.648
vol_prob_1 = vol_prob_1 * 0.253
vol_prob_2 = vol_prob_2 * 0.099

total = (vol_prob_0+vol_prob_1+vol_prob_2)
vol_prob_0 = (vol_prob_0)/ total
vol_prob_1 = (vol_prob_1)/ total
vol_prob_2 = (vol_prob_2)/ total

vol_prob_0 = vol_prob_0
vol_prob_1 = vol_prob_1
vol_prob_2 = vol_prob_2

vol_prob_0 = vol_prob_0.rename("vol_prob_0")
vol_prob_1 = vol_prob_1.rename("vol_prob_1")
vol_prob_2 = vol_prob_2.rename("vol_prob_2")

train = train.join(vol_prob_0)
train = train.join(vol_prob_1)
train = train.join(vol_prob_2)

log_sum = log_feature.index.value_counts()
log_sum = (log_sum - log_sum.min())/(log_sum.max() - log_sum.min())
log_sum = log_sum[log_sum.index.isin(train.index)]
log_sum = log_sum.rename("log_sum")
train = train.join(log_sum)

ev_sum = event_type.index.value_counts()
ev_sum = (ev_sum - ev_sum.min())/(ev_sum.max() - ev_sum.min())
ev_sum = ev_sum[ev_sum.index.isin(train.index)]
ev_sum = ev_sum.rename("ev_sum")
train = train.join(ev_sum)

res_num = resource_type.index.value_counts()
res_num = (res_num - res_num.min())/(res_num.max() - res_num.min())
res_num = res_num[res_num.index.isin(train.index)]
res_num = res_num.rename("res_num")
train = train.join(res_num)

log_fet = log_feature.log_feature.value_counts()

train["locc"] = train.location
train.locc = (train.locc - train.locc.min()) / (train.locc.max() - train.locc.min())

# one hot encoding
"""
loc = train[train.location.isin(loc_idxs)].copy()
loc["placeholder"] = 1
loc = loc.pivot(columns='location', values='placeholder')
loc.columns = ['location_%i' % col for col in loc.columns]
train = pd.merge(train, loc, how='left', right_index=True, left_index=True).fillna(0)
"""
res = resource_type[resource_type.resource_type.isin(res_idx)].copy()
res["placeholder"] = 1
res = res.pivot(columns='resource_type', values='placeholder')
res = res[res.index.isin(train.index)]
res.columns = ['resource_type_%i' % col for col in res.columns]
train = pd.merge(train, res, how='left', right_index=True, left_index=True).fillna(0)

sev = severity_type[severity_type.severity_type.isin(sev_idx)].copy()
sev["placeholder"] = 1
sev = sev.pivot(columns='severity_type', values='placeholder')
sev = sev[sev.index.isin(train.index)]
sev.columns = ['severity_type_%i' % col for col in sev.columns]
train = pd.merge(train, sev, how='left', right_index=True, left_index=True).fillna(0)

ev = event_type[event_type.event_type.isin(event_idxs)].copy()
ev["placeholder"] = 1
ev = ev.pivot(columns='event_type', values='placeholder')
ev = ev[ev.index.isin(train.index)]
ev.columns = ['event_type_%i' % col for col in ev.columns]
train = pd.merge(train, ev, how='left', right_index=True, left_index=True).fillna(0)

"""
log = log_feature[log_feature.log_feature.isin(log_idxs)].copy()
log["placeholder"] = 1
log = log.pivot(columns='log_feature', values='placeholder')
log = log[log.index.isin(train.index)]
log.columns = ['log_feature_%i' % col for col in log.columns]
train = pd.merge(train, log, how='left', right_index=True, left_index=True).fillna(0)
"""
log = log_feature[log_feature.log_feature.isin(log_idxs)].copy()
#log["placeholder"] = 1
log = log.pivot(columns='log_feature', values='volume')
log = log[log.index.isin(train.index)]
log.columns = ['log_feature_%i' % col for col in log.columns]
train = pd.merge(train, log, how='left', right_index=True, left_index=True).fillna(0)

for col in log.columns:
    
    train[col] = (train[col] - train[col].min()) / (train[col].max() - train[col].min())

vol = log_feature[log_feature.volume.isin(vol_idxs)].copy()
vol = vol.drop(columns = "log_feature")
vol = pd.get_dummies(vol, prefix = "vol", columns = ["volume",],  dummy_na=False)
vol = vol.groupby("id").agg(lambda x: 1 if sum(list(x)) >= 1 else 0)
vol = vol[vol.index.isin(train.index)]

#vol["placeholder"] = 1
#vol = vol.pivot(columns='volume', values='placeholder')
#vol.columns = ['volume_%i' % col for col in vol.columns]
#vol = vol[vol.index.isin(train.index)]

train = train.drop(columns = "location")
#train = pd.merge(train, vol, how='left', right_index=True, left_index=True).fillna(0)
"""
vol_sum = log_feature.groupby(log_feature.index).volume.apply(lambda x: sum(x))
vol_sum = (vol_sum - vol_sum.min())/(vol_sum.max() - vol_sum.min())
vol_sum = vol_sum[vol_sum.index.isin(train.index)]
train = train.join(vol_sum).rename(columns={'volume': 'volume_sum'})
"""

#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#vol_sum = min_max_scaler.fit_transform()
#train = pd.merge(train, vol_sum, how='left', right_index=True, left_index=True).fillna(0)

#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler((0, 295))
#scaler.fit(train)
#scaler.transform(train)

temp  = train
test  = train[train.fault_severity == -1]
train = train[train.fault_severity != -1]
test_x = test.loc[:, test.columns != "fault_severity"]
x_test = test_x.values

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from timeit import default_timer as timer

X = train.loc[:, train.columns != "fault_severity"]
X = X.values
Y = train.fault_severity.values
#y_dict = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]),
#          2: np.array([0, 0, 1])}
y_dict = {0: np.array([1, 0, 0]), 
          1: np.array([0, 1, 0]),
          2: np.array([0, 0, 1])}

Y = np.array([y_dict[y] for y in Y])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

"""
params = {0:{"hidden_layer_sizes":(100,), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
          1:{"hidden_layer_sizes":(100,100), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
          2:{"hidden_layer_sizes":(100,100,100), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
             
          3:{"hidden_layer_sizes":(200,), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
          4:{"hidden_layer_sizes":(200,200), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
          5:{"hidden_layer_sizes":(200,200,200), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
             
          6:{"hidden_layer_sizes":(300,), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
          7:{"hidden_layer_sizes":(300,300), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
          8:{"hidden_layer_sizes":(300,300,300), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True}
         }
"""
params = {#0: {"hidden_layer_sizes":(50,20,10), "warm_start" : False},
          1: {"hidden_layer_sizes":(300,300), "warm_start" : False,
              "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
              "early_stopping": True},
              #"learning_rate_init": 0.001},
          #2: {"hidden_layer_sizes":(200,60), "warm_start" : False,
          #    "solver": 'adam', "tol":0.00005, "alpha": 0.001}, 
          #2: {"hidden_layer_sizes":(200,50), "warm_start" : False}, 
          #3: {"hidden_layer_sizes":(300,100), "warm_start" : False},
          #4: {"hidden_layer_sizes":(500,200), "warm_start" : False},
          #4: {"hidden_layer_sizes":(300,300), "warm_start" : False,
          #    "solver": 'adam', "tol":0.00005, "alpha": 0.001},
          #5: {"hidden_layer_sizes":(400,200), "warm_start" : False,
          #    "solver": 'adam', "tol":0.00005, "alpha": 0.001},    
          #6: {"hidden_layer_sizes":(600,300), "warm_start" : False,
          #    "solver": 'adam', "tol":0.00005, "alpha": 0.001,
          #    "early_stopping": True, "random_state": 300},
          #6: {"hidden_layer_sizes":(200,100,30), "warm_start" : False},
          #7: {"hidden_layer_sizes":(500,300,100), "warm_start" : False},
          #7: {"hidden_layer_sizes":(400,200,50), "warm_start" : False,
          #     "solver": 'adam', "tol":0.00005, "alpha": 0.001, "random_state": 20,
          #     "early_stopping": True},
          #8: {"hidden_layer_sizes":(600,300,100), "warm_start" : False,
          #     "solver": 'adam', "tol":0.00005, "alpha": 0.001,
          #     "early_stopping": True, "random_state": 300},
          #9: {"hidden_layer_sizes":(800,200,50), "warm_start" : False,
          #     "solver": 'adam', "tol":0.00005, "alpha": 0.001,
          #     "early_stopping": True, "random_state": 300},
          #10: {"hidden_layer_sizes":(800,200,50), "warm_start" : False,
          #     "solver": 'adam', "tol":0.00005, "alpha": 0.01,
          #     "early_stopping": True, "random_state": 300},
          #11: {"hidden_layer_sizes":(800,200,50), "warm_start" : False,
          #     "solver": 'adam', "tol":0.00005, "alpha": 0.1,
          #     "early_stopping": True, "random_state": 300}
          #9: {"hidden_layer_sizes":(3000,2000,1000), "warm_start" : False,
          #    "solver": 'lbfgs', "tol":0.00005, "alpha": 0.001}
          }
          
print("started")

def logloss(x,y,ann):
    
    logprob = ann.predict_log_proba(x)
    logprob = logprob*y
    logprob = (logprob.sum()*-1)/x.shape[0]
    
    return logprob

train_logloss = {}
cv_logloss    = {}


for k in range(2):
    
    print(k)
    start = timer()
    ann  = MLPClassifier(hidden_layer_sizes=(200,200),warm_start = False,
                         solver= 'adam', tol=0.00005, alpha=0.001, 
                         early_stopping = True).fit(X_train, Y_train)
    print("training error: ", timer()-start)
    print(str(ann.hidden_layer_sizes))
    print("training error: ", 1 - ann.score(X_train, Y_train))
    print("validation error: ", 1 - ann.score(X_test, Y_test))
    #print()
    print("time:", str(timer()-start))
    print("log loss 1", logloss(X_train, Y_train,ann))
    print("log loss 2", logloss(X_test, Y_test,ann))
    print("log loss 1", log_loss(Y_train, ann.predict_proba(X_train)))
    print("log loss 2", log_loss(Y_test, ann.predict_proba(X_test)))

    
    train_logloss[k] = logloss(X_train, Y_train,ann)
    cv_logloss[k] = logloss(X_test, Y_test,ann)
    
    pred = ann.predict(X_test)

    zero = 0
    one  = 0
    two  = 0

    for k in range(len(pred)):
    
        if False in (pred[k] == Y_test[k]): 

            if Y_test[k][0]: zero+=1
            elif Y_test[k][1]: one+=1
            elif Y_test[k][2]: two+=1

    total_error=zero+one+two
    print("zero error: ", zero/total_error)
    print("one error: ", one/total_error)
    print("two error: ", two/total_error)
    
    

#wrong = {}

"""
for param in params:
    
    print(str(param))
    start = timer()
    ann  = MLPClassifier(**params[param]).fit(X_train, Y_train)
    print(str(ann.hidden_layer_sizes))
    print("training error: ", 1 - ann.score(X_train, Y_train))
    print("validation error: ", 1 - ann.score(X_test, Y_test))
    #print()
    print("time:", str(timer()-start))
    print("log loss 1", logloss(X_train, Y_train))
    print("log loss 2", logloss(X_test, Y_test))
    
    pred = ann.predict(X_test)

    zero = 0
    one  = 0
    two  = 0

    for k in range(len(pred)):
    
        if False in (pred[k] == Y_test[k]): 

            if Y_test[k][0]: zero+=1
            elif Y_test[k][1]: one+=1
            elif Y_test[k][2]: two+=1

    total_error=zero+one+two
    print("zero error: ", zero/total_error)
    print("one error: ", one/total_error)
    print("two error: ", two/total_error)

"""
"""
pred = ann.predict(X_test)

zero = 0
one  = 0
two  = 0

for k in range(len(pred)):
    
    if False in (pred[k] == Y_test[k]): 

        if Y_test[k][0]: zero+=1
        elif Y_test[k][1]: one+=1
        elif Y_test[k][2]: two+=1

total_error=zero+one+two
print("zero error: ", zero/total_error)
print("one error: ", one/total_error)
print("two error: ", two/total_error)

"""
    
"""
from sklearn.tree import DecisionTreeClassifier as bestDecisionStump

decision_stumps = []
accuracy = []
t = 1000
hypotheses = dict()
N = X_train.shape[0]
w = np.ones(N) / N

for it in range(t):
    
    f = bestDecisionStump(max_depth=1)
    f.fit(X_train, Y_train, sample_weight = w)
    pred = f.predict(X_train)
    eps = w.dot(pred != Y_train)
    #print("erro")
    #eps = 1 - f.score(X_train, Y_train, sample_weight = w)
    #print(eps)
    alpha = (np.log(1 - eps) - np.log(eps)) / 2
    #alpha = (np.log((1 - eps)/eps)) / 2
    w = w * np.exp(- alpha * Y_train * pred)
    w = w / w.sum()
    
    hypotheses[it] = {"function":f,"weight":alpha}
    
y = np.zeros(X_test.shape[0])

for it in hypotheses:
    y = y + (hypotheses[it]["weight"] * hypotheses[it]["function"].predict(X_test))  
y = np.sign(y)
accuracy.append(accuracy_score(y,Y_test))


log_feature_num = log_feature.groupby('id').count()[["log_feature"]]
log_feature_num.columns = ["log_feature_num"]

volume_num = log_feature.groupby('id').count()[["volume"]]
volume_num.columns = ["volume_num"]

resource_type_num = resource_type.groupby('id').count()[["resource_type"]]
resource_type_num.columns = ["resource_type_num"]

event_type_num = event_type.groupby('id').count()[["event_type"]]
event_type_num.columns = ["event_type_num"]

train = pd.merge(train, log_feature_num, how='inner', right_index=True, left_index=True)
train = pd.merge(train, volume_num, how='inner', right_index=True, left_index=True)
train = pd.merge(train, resource_type_num, how='inner', right_index=True, left_index=True)
train = pd.merge(train, event_type_num, how='inner', right_index=True, left_index=True)

fault_sev_0 = train[train.fault_severity == 0]
fault_sev_1 = train[train.fault_severity == 1]
fault_sev_2 = train[train.fault_severity == 2]

# log feature and volume

#plt.figure()
#fault_sev_0.log_feature_num.value_counts().plot.bar()
#plt.figure()
#fault_sev_1.log_feature_num.value_counts().plot.bar()
#plt.figure()
#fault_sev_2.log_feature_num.value_counts().plot.bar()

#resource_type
#severity_type
#event_type
#log_feature

resource_type = pd.get_dummies(resource_type.resource_type)
resource_type.columns = ["resource_" + str(col) for col in resource_type.columns]
severity_type = pd.get_dummies(severity_type.severity_type)
severity_type.columns = ["severity_" + str(col) for col in severity_type.columns]
event_type = pd.get_dummies(event_type.event_type)
event_type.columns = ["event_type_" + str(col) for col in event_type.columns]
log_feature = pd.get_dummies(log_feature.log_feature)
log_feature.columns = ["feature_" + str(col) for col in log_feature.columns]

#train = train.merge(resource_type, left_on='id', right_on='id', how = "outer")
#train = train.merge(severity_type, left_on='id', right_on='id', how = "outer")
#train = train.merge(event_type, left_on='id', right_on='id', how = "outer")
#train = train.merge(log_feature, left_on='id', right_on='id', how = "outer")

"""
"""
plt.figure()
fault_sev_0.resource_type_num.value_counts().plot.bar(title = "resource")
plt.figure()
fault_sev_1.resource_type_num.value_counts().plot.bar(title = "resource")
plt.figure()
fault_sev_2.resource_type_num.value_counts().plot.bar(title = "resource")

plt.figure()
fault_sev_0.event_type_num.value_counts().plot.bar(title = "event")
plt.figure()
fault_sev_1.event_type_num.value_counts().plot.bar(title = "event")
plt.figure()
fault_sev_2.event_type_num.value_counts().plot.bar(title = "event")
"""


def create_submission(file_name):

    predprob  = ann.predict_proba(x_test)
    predprob2 = np.zeros((11171,4)).tolist()

    for k in range(predprob.shape[0]):
        predprob2[k][0] = int(test.index[k])
        predprob2[k][1:] = predprob[k][:]
    
    #print(predprob2)
    import csv
    
    with open(file_name + ".csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["id","predict_0","predict_1","predict_2"])
        wr.writerows(predprob2)