#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 26 23:54:54 2018

@author: maurice
"""

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

#train = train.append(test)
train['location'] = train.location.apply(lambda x: int(x.split('location ')[1]))
#test['location'] = test.location.apply(lambda x: int(x.split('location ')[1]))
#train['time_id'] = train.index
resource_type['resource_type'] = resource_type.resource_type.apply(lambda x: int(x.split('resource_type ')[1]))
severity_type['severity_type'] = severity_type.severity_type.apply(lambda x: int(x.split('severity_type ')[1]))
event_type['event_type'] = event_type.event_type.apply(lambda x: int(x.split('event_type ')[1]))
log_feature['log_feature'] = log_feature.log_feature.apply(lambda x: int(x.split('feature ')[1]))
log_feature['volume'] = log_feature.volume.apply(lambda x: int(x))

# analyzing feature distribution among overall data, examples, and for each fault severity
resource_train = resource_type.loc[resource_type.index.isin(train.index)]
severity_train = severity_type.loc[severity_type.index.isin(train.index)]
event_train = event_type.loc[event_type.index.isin(train.index)]
log_train = log_feature.loc[log_feature.index.isin(train.index)]

train["time"] = np.linspace(train.location.min(),train.location.max(),7381)
fault_sev_0 = train[train.fault_severity == 0]
fault_sev_1 = train[train.fault_severity == 1]
fault_sev_2 = train[train.fault_severity == 2]

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

vol_2 = log_train_2.groupby(log_train_2.index).volume.apply(lambda x: sum(x))
vol_1 = log_train_1.groupby(log_train_1.index).volume.apply(lambda x: sum(x))
vol_0 = log_train_0.groupby(log_train_0.index).volume.apply(lambda x: sum(x))
vol_train = log_train.groupby(log_train.index).volume.apply(lambda x: sum(x))

#vol_prob_train = np.exp(-((vol_train - vol_train.mean())**2)/(2*vol_train.std()*vol_train.std()))/(np.sqrt(2*np.pi*vol_train.std()*vol_train.std()))
vol_prob_0 = np.exp(-((vol_0 - vol_0.mean())**2)/(2*vol_0.std()*vol_0.std()))/(np.sqrt(2*np.pi*vol_0.std()*vol_0.std()))
vol_prob_1 = np.exp(-((vol_1 - vol_1.mean())**2)/(2*vol_1.std()*vol_1.std()))/(np.sqrt(2*np.pi*vol_1.std()*vol_1.std()))
vol_prob_2 = np.exp(-((vol_2 - vol_2.mean())**2)/(2*vol_2.std()*vol_2.std()))/(np.sqrt(2*np.pi*vol_2.std()*vol_2.std()))

#total = (vol_prob_0+vol_prob_1+vol_prob_2)
#vol_prob_0 = (vol_prob_0)/ total
#vol_prob_1 = (vol_prob_1)/ total
#vol_prob_2 = (vol_prob_2)/ total

fig0 = plt.figure()
#vol_2 = (vol_2 - vol_2.min())/(vol_2.max() - vol_2.min())
#vol_1 = (vol_1 - vol_1.min())/(vol_1.max() - vol_1.min())
#vol_0 = (vol_0 - vol_0.min())/(vol_0.max() - vol_0.min())
#vol_prob_train.plot.kde()
vol_prob_2.plot.kde()
vol_prob_1.plot.kde()
vol_prob_0.plot.kde()
plt.xlim(0,0.0155)
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("(prob) Sum of volume for log occurences for each id")

fig1 = plt.figure()
#vol_prob_train.plot.kde()
vol_2.plot.kde()
vol_1.plot.kde()
vol_0.plot.kde()
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Sum of volume for log occurences for each id")
plt.xlim(-100,200)
plt.ylim(0,0.04)
plt.show()
#plt.figure()
#log_train_0.volume.value_counts().plot.kde()

fig2 = plt.figure()
#train.location.plot.kde()
#print(fault_sev_2.location.value_counts().head(150))
fault_sev_2.location.plot.kde()
#plt.figure()
#print(fault_sev_1.location.value_counts().head(150))
fault_sev_1.location.plot.kde()
#plt.figure()
#print(fault_sev_0.location.value_counts().head(150))
fault_sev_0.location.plot.kde()
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Location number")


fig3 = plt.figure()
severity_train_2.severity_type.plot.kde()
severity_train_1.severity_type.plot.kde()
severity_train_0.severity_type.plot.kde()
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Severity type")


fig4 = plt.figure()
resource_train_2.resource_type.plot.kde()
resource_train_1.resource_type.plot.kde()
resource_train_0.resource_type.plot.kde()
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Resource type")


fig5 = plt.figure()
log_train_2.volume.plot.kde()
log_train_1.volume.plot.kde()
log_train_0.volume.plot.kde()
plt.xlim(-30,50)
plt.ylim(0,0.15)
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Volume")


fig6 = plt.figure()
log_train_2.log_feature.plot.kde()
log_train_1.log_feature.plot.kde()
log_train_0.log_feature.plot.kde()
plt.xlim(0,450)
plt.ylim(0,0.015)
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Log feature")

"""
fig7 = plt.figure()
fault_sev_2["ind"] = fault_sev_2.index
fault_sev_1["ind"] = fault_sev_1.index
fault_sev_0["ind"] = fault_sev_0.index
fault_sev_2.ind.plot.kde()
fault_sev_1.ind.plot.kde()
fault_sev_0.ind.plot.kde()
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
"""

#log_feature_count = log_feature.groupby(log_feature.index).count()[['log_feature']]
"""
fig8 = plt.figure()
log_train_2.log_feature.groupby(log_train_2.index).count().plot.kde()
log_train_1.log_feature.groupby(log_train_1.index).count().plot.kde()
log_train_0.log_feature.groupby(log_train_0.index).count().plot.kde()
plt.xlim(-10,30)
plt.ylim(0,0.5)
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Log feature")
"""
fig8 = plt.figure()
event_train_2.event_type.plot.kde()
event_train_1.event_type.plot.kde()
event_train_0.event_type.plot.kde()
plt.xlim(-10,30)
plt.ylim(0,0.2)
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Event type")

#train_copy = train.join(severity_train.severity_type)
#loc_sev = 
"""
fig9 = plt.figure()
tl = fault_sev_2.location
train["id"] = train.index
ti = train.id
tl = (tl - tl.min())/(tl.max() - tl.min())
ti = (ti - ti.min())/(ti.max() - ti.min())
tl.plot.kde()
ti.plot.kde()
"""
#train["time"] = np.linspace(train.location.min(),train.location.max(),7381)
#train_copy = train.join(severity_train.severity_type)
#train_copy = train_copy.reindex(train_copy.time)
#fig10 = plt.figure()
#plt.plot(train_copy.time, train_copy.severity_type.)
#plt.bar(train_copy[train_copy.severity_type==1].location.values)#,train_copy[train_copy.severity_type==1].severity_type.values)
#plt.plot(train_copy.location.values,train_copy[].severity_type.values)
#plt.plot(train_copy.location.values,train_copy[].severity_type.values)
#plt.plot(train_copy.location.values,train_copy[].severity_type.values)
#plt.plot(train_copy.location.values,train_copy[].severity_type.values)

fig9 = plt.figure()
fault_sev_2.time.plot.kde()
fault_sev_1.time.plot.kde()
fault_sev_0.time.plot.kde()
#plt.xlim(-10,30)
#plt.ylim(0,0.2)
plt.legend(["Fault severity 2","Fault severity 1","Fault severity 0"])
plt.xlabel("Time")

fig10 = plt.figure()

log_train_0.index.value_counts().plot.kde()
log_train_1.index.value_counts().plot.kde()
log_train_2.index.value_counts().plot.kde()
plt.legend(["Fault severity 0","Fault severity 1","Fault severity 2"])
plt.xlabel("log occurences")

log_idxs_0 = log_train_0.log_feature.value_counts().head(80).index                             
log_idxs_1 = log_train_1.log_feature.value_counts().head(100).index                             
log_idxs_2 = log_train_2.log_feature.value_counts().head(110).index         
log_idxs = log_idxs_2.append(log_idxs_0)                    
#log_idxs = log_idxs_2.append(log_idxs_train)
log_idxs = log_idxs.append(log_idxs_1)
log_idxs = log_idxs.drop_duplicates()

log = log_feature[log_feature.log_feature.isin(log_idxs)].copy()
#log["placeholder"] = 1
log = log.pivot(columns='log_feature', values='volume')
log = log[log.index.isin(train.index)]
log.columns = ['log_feature_%i' % col for col in log.columns]
#train = pd.merge(train, log, how='left', right_index=True, left_index=True).fillna(0)