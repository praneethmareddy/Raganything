# ==========================================================
# FULL ANOMALY DETECTION PIPELINE (TELEMETRY + EVENTS)
# ==========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)

print("\nSTEP 1 — Simulating monitoring data\n")

# ----------------------------------------------------------
# Simulate input data similar to your screenshot
# ----------------------------------------------------------

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
