import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)

# ===============================
# STEP 1 — Simulate Pod Events
# ===============================

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

rows = []
time_steps = 2000

for t in range(time_steps):

    pod = np.random.choice(pods)
    event = np.random.choice(events)

    eventType = eventType_map[event]

    if eventType == "steady":
        cpu = np.random.normal(40,4)
        mem = np.random.normal(500,40)
    else:
        cpu = np.random.normal(70,6)
        mem = np.random.normal(900,80)

    rows.append([t,pod,event,eventType,cpu,mem])

df = pd.DataFrame(rows,columns=[
    "time","pod","event","eventType","cpuUsage","memoryUsage"
])

# Inject anomalies
df.loc[800:820,"cpuUsage"] += 50
df.loc[1500:1510,"memoryUsage"] += 400


# ===============================
# STEP 2 — Encode categorical features
# ===============================

enc_pod = LabelEncoder()
enc_event = LabelEncoder()
enc_eventType = LabelEncoder()

df["pod"] = enc_pod.fit_transform(df["pod"])
df["event"] = enc_event.fit_transform(df["event"])
df["eventType"] = enc_eventType.fit_transform(df["eventType"])


# ===============================
# STEP 3 — Feature Scaling
# ===============================

features = ["pod","event","eventType","cpuUsage","memoryUsage"]

scaler = StandardScaler()
data = scaler.fit_transform(df[features])


# ===============================
# STEP 4 — Create Sliding Windows
# ===============================

window = 20
X = []

for i in range(len(data)-window):
    X.append(data[i:i+window])

X = np.array(X)

X_tensor = torch.tensor(X,dtype=torch.float32)


# ===============================
# STEP 5 — Transformer Autoencoder
# ===============================

class TransformerAE(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=64,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=2)

        self.decoder = nn.Linear(input_dim,input_dim)

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = TransformerAE(input_dim=X.shape[2])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# ===============================
# STEP 6 — Train Model
# ===============================

epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X_tensor)

    loss = criterion(output,X_tensor)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch,"Loss",loss.item())


# ===============================
# STEP 7 — Compute Anomaly Scores
# ===============================

model.eval()

with torch.no_grad():
    recon = model(X_tensor)

errors = torch.mean((X_tensor-recon)**2,dim=(1,2)).numpy()

threshold = np.mean(errors) + 3*np.std(errors)

anomalies = errors > threshold


# ===============================
# STEP 8 — Visualization
# ===============================

fig,axs = plt.subplots(3,1,figsize=(12,10))

axs[0].plot(df["cpuUsage"],label="CPU")
axs[0].set_title("CPU Usage")

axs[1].plot(df["memoryUsage"],label="Memory")
axs[1].set_title("Memory Usage")

axs[2].plot(errors,label="Anomaly Score")
axs[2].axhline(threshold,color="red")
axs[2].scatter(np.where(anomalies)[0],errors[anomalies],color="red")
axs[2].set_title("Detected Anomalies")

plt.tight_layout()
plt.show()
