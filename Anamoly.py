# ==========================================================
# COMPLETE PROMETHEUS-STYLE ANOMALY DETECTION PIPELINE
# Beginner Friendly – Step-By-Step Explanations
# ==========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)

print("\n================ STEP 0 : SYSTEM PIPELINE OVERVIEW ================\n")

print("""
Real Monitoring Pipeline (similar to Prometheus + ML):

        Pods / Containers
               │
               ▼
        Metrics Collection
   (CPU, Memory, Network, Events)
               │
               ▼
        Data Preprocessing
     - Encoding categorical data
     - Normalization
               │
               ▼
       Time-Series Window Creation
               │
               ▼
      Transformer Anomaly Detection Model
               │
               ▼
        Anomaly Score Calculation
               │
               ▼
        Alert / Grafana Dashboard
""")

print("\n===================================================================\n")


# ==========================================================
# STEP 1 : SIMULATE PROMETHEUS METRICS
# ==========================================================

print("STEP 1 : Simulating Prometheus-like monitoring data\n")

pods = ["pod1","pod2","pod3"]

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

# inject anomalies
df.loc[500:520,"cpuUsage"]+=60
df.loc[900:910,"memoryUsage"]+=400


print("Sample Input Data (first 10 rows):\n")
print(df.head(10))

print("\nColumns meaning:")
print("time → timestamp")
print("pod → which container/pod generated metric")
print("event → activity happening in pod")
print("eventType → workload type (steady or peak)")
print("cpuUsage → CPU utilization")
print("memoryUsage → memory consumption\n")


# ==========================================================
# STEP 2 : VISUALIZE RAW METRICS
# ==========================================================

print("STEP 2 : Visualizing raw telemetry metrics\n")

plt.figure(figsize=(12,6))
plt.plot(df["cpuUsage"])
plt.title("CPU Usage Over Time")
plt.xlabel("Time")
plt.ylabel("CPU Usage")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df["memoryUsage"])
plt.title("Memory Usage Over Time")
plt.xlabel("Time")
plt.ylabel("Memory Usage (MB)")
plt.show()

print("""
X-axis : Time
Y-axis : Metric value

Spikes in these graphs represent abnormal system behavior.
""")


# ==========================================================
# STEP 3 : ENCODE CATEGORICAL FEATURES
# ==========================================================

print("\nSTEP 3 : Encoding categorical variables\n")

enc_pod=LabelEncoder()
enc_event=LabelEncoder()
enc_type=LabelEncoder()

df["pod"]=enc_pod.fit_transform(df["pod"])
df["event"]=enc_event.fit_transform(df["event"])
df["eventType"]=enc_type.fit_transform(df["eventType"])

print("Encoded values example:\n")
print(df.head())


# ==========================================================
# STEP 4 : NORMALIZE FEATURES
# ==========================================================

print("\nSTEP 4 : Normalizing data\n")

features=["pod","event","eventType","cpuUsage","memoryUsage"]

scaler=StandardScaler()

data=scaler.fit_transform(df[features])

print("Normalized feature sample:\n")
print(data[:5])


# ==========================================================
# STEP 5 : CREATE TIME-SERIES WINDOWS
# ==========================================================

print("\nSTEP 5 : Creating sliding windows\n")

window=20

X=[]

for i in range(len(data)-window):
    X.append(data[i:i+window])

X=np.array(X)

print("Windowed data shape:",X.shape)
print("""
Meaning:
samples = number of sequences
window size = 20 timestamps
features = pod,event,eventType,cpu,memory
""")


X_tensor=torch.tensor(X,dtype=torch.float32)


# ==========================================================
# STEP 6 : SOTA TRANSFORMER ANOMALY MODEL
# (Inspired by TranAD / Anomaly Transformer)
# ==========================================================

print("\nSTEP 6 : Building Transformer-based anomaly detection model\n")

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


# ==========================================================
# STEP 7 : TRAIN MODEL
# ==========================================================

print("\nSTEP 7 : Training anomaly detection model\n")

epochs=10

for epoch in range(epochs):

    optimizer.zero_grad()

    output=model(X_tensor)

    loss=criterion(output,X_tensor)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch,"Training Loss:",loss.item())


# ==========================================================
# STEP 8 : COMPUTE ANOMALY SCORE
# ==========================================================

print("\nSTEP 8 : Computing anomaly score\n")

model.eval()

with torch.no_grad():
    recon=model(X_tensor)

errors=torch.mean((X_tensor-recon)**2,dim=(1,2)).numpy()

threshold=np.mean(errors)+3*np.std(errors)

anomalies=errors>threshold

print("Threshold value:",threshold)
print("Number of anomalies detected:",anomalies.sum())


# ==========================================================
# STEP 9 : GRAFANA-STYLE DASHBOARD
# ==========================================================

print("\nSTEP 9 : Grafana-style anomaly dashboard\n")

fig,axs=plt.subplots(3,1,figsize=(14,10))

axs[0].plot(df["cpuUsage"])
axs[0].set_title("CPU Usage Monitoring")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("CPU Usage")

axs[1].plot(df["memoryUsage"])
axs[1].set_title("Memory Usage Monitoring")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Memory Usage")

axs[2].plot(errors,label="Anomaly Score")
axs[2].axhline(threshold,color="red",label="Threshold")
axs[2].scatter(np.where(anomalies)[0],errors[anomalies],color="red")
axs[2].set_title("Detected Anomalies")
axs[2].set_xlabel("Time Window")
axs[2].set_ylabel("Reconstruction Error")

axs[2].legend()

plt.tight_layout()
plt.show()


print("""
Dashboard Explanation:

Graph 1 : CPU Usage over time
Graph 2 : Memory Usage over time
Graph 3 : Anomaly Score

Red horizontal line = anomaly threshold
Red points = detected anomalies
""")
