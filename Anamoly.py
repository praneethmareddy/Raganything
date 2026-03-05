# ==========================================================
# REALISTIC TELEMETRY ANOMALY DETECTION PIPELINE
# (Pods + Events + CPU + Memory)
# ==========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

print("\nSTEP 1 — Simulating realistic telemetry data (>2000 rows)\n")

# ----------------------------------------------------------
# Realistic pods
# ----------------------------------------------------------

pods = ["lcm-pod","monitor-pod","analytics-pod"]

# ----------------------------------------------------------
# Realistic operations
# ----------------------------------------------------------

events = [
"app_install",
"app_monitoring",
"app_uninstall",
"log_retrieval",
"log_update",
"push_image",
"pull_image"
]

eventType_map = {
"app_install":"peak",
"push_image":"peak",
"pull_image":"peak",
"log_retrieval":"peak",
"app_monitoring":"steady",
"app_uninstall":"steady",
"log_update":"steady"
}

rows=[]
time_steps=2500

for t in range(time_steps):

    pod=np.random.choice(pods)
    event=np.random.choice(events)
    eventType=eventType_map[event]

    # realistic telemetry patterns
    if pod=="lcm-pod":
        base_cpu=35
        base_mem=400
    elif pod=="monitor-pod":
        base_cpu=30
        base_mem=300
    else:
        base_cpu=45
        base_mem=450

    if eventType=="steady":
        cpu=np.random.normal(base_cpu,4)
        mem=np.random.normal(base_mem,30)
    else:
        cpu=np.random.normal(base_cpu+30,6)
        mem=np.random.normal(base_mem+400,60)

    rows.append([t,pod,event,eventType,cpu,mem,0])

df=pd.DataFrame(rows,columns=["time","pod","event","eventType","cpuUsage","memoryUsage","anomaly"])

# ----------------------------------------------------------
# Inject anomalies (simulate real failures)
# ----------------------------------------------------------

# CPU runaway process
df.loc[800:820,"cpuUsage"]+=70
df.loc[800:820,"anomaly"]=1

# memory leak
df.loc[1500:1520,"memoryUsage"]+=500
df.loc[1500:1520,"anomaly"]=1

# abnormal monitoring spike
df.loc[2000:2010,"cpuUsage"]+=60
df.loc[2000:2010,"anomaly"]=1

print("Dataset size:",len(df))
print(df.head())

# ----------------------------------------------------------
# STEP 2 — Encode categorical features
# ----------------------------------------------------------

print("\nSTEP 2 — Encoding categorical features")

enc_pod=LabelEncoder()
enc_event=LabelEncoder()
enc_type=LabelEncoder()

df["pod"]=enc_pod.fit_transform(df["pod"])
df["event"]=enc_event.fit_transform(df["event"])
df["eventType"]=enc_type.fit_transform(df["eventType"])

# ----------------------------------------------------------
# STEP 3 — Normalize features
# ----------------------------------------------------------

print("\nSTEP 3 — Normalizing telemetry metrics")

features=["pod","event","eventType","cpuUsage","memoryUsage"]

scaler=StandardScaler()
data=scaler.fit_transform(df[features])

# ----------------------------------------------------------
# STEP 4 — Create time windows
# ----------------------------------------------------------

print("\nSTEP 4 — Creating sliding windows")

window=20
X=[]
labels=[]

for i in range(len(data)-window):
    X.append(data[i:i+window])
    labels.append(df["anomaly"].iloc[i+window])

X=np.array(X)
labels=np.array(labels)

print("Windowed dataset shape:",X.shape)

# ----------------------------------------------------------
# STEP 5 — Train / Test split
# ----------------------------------------------------------

X_train,X_test,y_train,y_test=train_test_split(
    X,labels,test_size=0.3,shuffle=False
)

X_train_tensor=torch.tensor(X_train,dtype=torch.float32)
X_test_tensor=torch.tensor(X_test,dtype=torch.float32)

# ----------------------------------------------------------
# STEP 6 — Transformer Anomaly Model
# ----------------------------------------------------------

print("\nSTEP 6 — Building Transformer Autoencoder")

class TransformerAE(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.encoder_layer=nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=64,
            batch_first=True
        )

        self.encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=2)

        self.decoder=nn.Linear(input_dim,input_dim)

    def forward(self,x):

        x=self.encoder(x)
        x=self.decoder(x)

        return x


model=TransformerAE(X.shape[2])

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# ----------------------------------------------------------
# STEP 7 — Training
# ----------------------------------------------------------

print("\nSTEP 7 — Training anomaly detection model")

epochs=15

for epoch in range(epochs):

    optimizer.zero_grad()

    output=model(X_train_tensor)

    loss=criterion(output,X_train_tensor)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch,"Loss:",round(loss.item(),4))

# ----------------------------------------------------------
# STEP 8 — Compute anomaly scores
# ----------------------------------------------------------

print("\nSTEP 8 — Computing anomaly scores")

model.eval()

with torch.no_grad():
    recon_train=model(X_train_tensor)
    recon_test=model(X_test_tensor)

train_errors=torch.mean((X_train_tensor-recon_train)**2,dim=(1,2)).numpy()
test_errors=torch.mean((X_test_tensor-recon_test)**2,dim=(1,2)).numpy()

threshold=np.mean(train_errors)+3*np.std(train_errors)

train_pred=(train_errors>threshold).astype(int)
test_pred=(test_errors>threshold).astype(int)

# ----------------------------------------------------------
# STEP 9 — Metrics
# ----------------------------------------------------------

print("\nSTEP 9 — Model evaluation")

precision=precision_score(y_test,test_pred)
recall=recall_score(y_test,test_pred)
f1=f1_score(y_test,test_pred)

print("Precision:",round(precision,3))
print("Recall:",round(recall,3))
print("F1 Score:",round(f1,3))

# ----------------------------------------------------------
# STEP 10 — Visualization (Grafana style)
# ----------------------------------------------------------

print("\nSTEP 10 — Grafana-style dashboard")

fig,axs=plt.subplots(3,1,figsize=(12,10))

axs[0].plot(df["cpuUsage"])
axs[0].set_title("CPU Usage")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("CPU")

axs[1].plot(df["memoryUsage"])
axs[1].set_title("Memory Usage")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Memory")

axs[2].plot(test_errors,label="Anomaly Score")
axs[2].axhline(threshold,color="red",label="Threshold")
axs[2].scatter(np.where(test_pred==1),test_errors[test_pred==1],color="red")
axs[2].set_title("Detected Anomalies")
axs[2].set_xlabel("Time Window")
axs[2].set_ylabel("Reconstruction Error")

axs[2].legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# STEP 11 — Inference examples
# ----------------------------------------------------------

print("\nSTEP 11 — Running inference on 5 realistic examples\n")

examples=[
["lcm-pod","app_monitoring","steady",40,420],
["monitor-pod","app_monitoring","steady",35,310],
["analytics-pod","push_image","peak",85,950],
["lcm-pod","app_monitoring","steady",120,450], # abnormal CPU
["monitor-pod","app_monitoring","steady",30,900] # memory leak
]

for e in examples:

    pod,event,etype,cpu,mem=e

    x=[enc_pod.transform([pod])[0],
       enc_event.transform([event])[0],
       enc_type.transform([etype])[0],
       cpu,
       mem]

    x=scaler.transform([x])

    seq=np.tile(x,(window,1))
    seq=torch.tensor([seq],dtype=torch.float32)

    with torch.no_grad():
        recon=model(seq)

    err=torch.mean((seq-recon)**2).item()

    result="ANOMALY" if err>threshold else "NORMAL"

    print("Input:",e,"->",result,"score:",round(err,4))"app_monitoring":"steady",
"app_uninstall":"steady",
"log_update":"steady"
}

rows=[]
time_steps=1500

for t in range(time_steps):

    pod=np.random.choice(pods)
    event=np.random.choice(events)
    eventType=eventType_map[event]

    if eventType=="steady":
        cpu=np.random.normal(40,4)
        mem=np.random.normal(500,40)
    else:
        cpu=np.random.normal(70,6)
        mem=np.random.normal(900,80)

    rows.append([t,pod,event,eventType,cpu,mem])

df=pd.DataFrame(rows,columns=["time","pod","event","eventType","cpuUsage","memoryUsage"])

# Inject anomalies
df.loc[500:520,"cpuUsage"]+=60
df.loc[900:910,"memoryUsage"]+=400

print("Example data rows:")
print(df.head())

# ----------------------------------------------------------
# STEP 2 — Visualize metrics
# ----------------------------------------------------------

print("\nSTEP 2 — Visualizing raw metrics")

plt.figure(figsize=(10,4))
plt.plot(df["cpuUsage"])
plt.title("CPU Usage")
plt.xlabel("Time")
plt.ylabel("CPU")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(df["memoryUsage"])
plt.title("Memory Usage")
plt.xlabel("Time")
plt.ylabel("Memory")
plt.show()

# ----------------------------------------------------------
# STEP 3 — Encode categorical features
# ----------------------------------------------------------

print("\nSTEP 3 — Encoding categorical variables")

enc_pod=LabelEncoder()
enc_event=LabelEncoder()
enc_type=LabelEncoder()

df["pod"]=enc_pod.fit_transform(df["pod"])
df["event"]=enc_event.fit_transform(df["event"])
df["eventType"]=enc_type.fit_transform(df["eventType"])

print(df.head())

# ----------------------------------------------------------
# STEP 4 — Normalization
# ----------------------------------------------------------

print("\nSTEP 4 — Normalizing features")

features=["pod","event","eventType","cpuUsage","memoryUsage"]

scaler=StandardScaler()
data=scaler.fit_transform(df[features])

print("Normalized sample:")
print(data[:5])

# ----------------------------------------------------------
# STEP 5 — Create time windows
# ----------------------------------------------------------

print("\nSTEP 5 — Creating sliding windows")

window=20
X=[]

for i in range(len(data)-window):
    X.append(data[i:i+window])

X=np.array(X)

print("Data shape:",X.shape)

X_tensor=torch.tensor(X,dtype=torch.float32)

# ----------------------------------------------------------
# STEP 6 — Transformer Anomaly Model
# ----------------------------------------------------------

print("\nSTEP 6 — Building Transformer model")

class TransformerAE(nn.Module):

    def __init__(self,input_dim):
        super().__init__()

        self.encoder_layer=nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=64,
            batch_first=True
        )

        self.encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=2)

        self.decoder=nn.Linear(input_dim,input_dim)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

model=TransformerAE(X.shape[2])

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# ----------------------------------------------------------
# STEP 7 — Train model
# ----------------------------------------------------------

print("\nSTEP 7 — Training model")

epochs=10

for epoch in range(epochs):

    optimizer.zero_grad()

    output=model(X_tensor)

    loss=criterion(output,X_tensor)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch,"Loss:",loss.item())

# ----------------------------------------------------------
# STEP 8 — Compute anomaly score
# ----------------------------------------------------------

print("\nSTEP 8 — Computing anomaly scores")

model.eval()

with torch.no_grad():
    recon=model(X_tensor)

errors=torch.mean((X_tensor-recon)**2,dim=(1,2)).numpy()

threshold=np.mean(errors)+3*np.std(errors)

anomalies=errors>threshold

print("Threshold:",threshold)
print("Anomalies detected:",anomalies.sum())

# ----------------------------------------------------------
# STEP 9 — Visualization dashboard
# ----------------------------------------------------------

print("\nSTEP 9 — Grafana-style visualization")

fig,axs=plt.subplots(3,1,figsize=(12,10))

axs[0].plot(df["cpuUsage"])
axs[0].set_title("CPU Usage")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("CPU")

axs[1].plot(df["memoryUsage"])
axs[1].set_title("Memory Usage")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Memory")

axs[2].plot(errors,label="Anomaly Score")
axs[2].axhline(threshold,color="red",label="Threshold")
axs[2].scatter(np.where(anomalies)[0],errors[anomalies],color="red")
axs[2].set_title("Detected Anomalies")
axs[2].set_xlabel("Time Window")
axs[2].set_ylabel("Error")

axs[2].legend()

plt.tight_layout()
plt.show()

print("\nPipeline complete.")
