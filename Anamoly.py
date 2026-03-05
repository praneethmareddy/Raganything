# ==========================================================
# REALISTIC TELEMETRY ANOMALY DETECTION PIPELINE
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

print("\nSTEP 1 — Simulating realistic telemetry data\n")

pods = ["lcm-pod", "monitor-pod", "analytics-pod"]

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
    "app_install": "peak",
    "push_image": "peak",
    "pull_image": "peak",
    "log_retrieval": "peak",
    "app_monitoring": "steady",
    "app_uninstall": "steady",
    "log_update": "steady"
}

rows = []
time_steps = 2500

for t in range(time_steps):

    pod = np.random.choice(pods)
    event = np.random.choice(events)
    eventType = eventType_map[event]

    if pod == "lcm-pod":
        base_cpu = 35
        base_mem = 400
    elif pod == "monitor-pod":
        base_cpu = 30
        base_mem = 300
    else:
        base_cpu = 45
        base_mem = 450

    if eventType == "steady":
        cpu = np.random.normal(base_cpu, 4)
        mem = np.random.normal(base_mem, 30)
    else:
        cpu = np.random.normal(base_cpu + 30, 6)
        mem = np.random.normal(base_mem + 400, 60)

    rows.append([t, pod, event, eventType, cpu, mem, 0])

df = pd.DataFrame(
    rows,
    columns=["time", "pod", "event", "eventType", "cpuUsage", "memoryUsage", "anomaly"]
)

# Inject anomalies
df.loc[800:820, "cpuUsage"] += 70
df.loc[800:820, "anomaly"] = 1

df.loc[1500:1520, "memoryUsage"] += 500
df.loc[1500:1520, "anomaly"] = 1

df.loc[2000:2010, "cpuUsage"] += 60
df.loc[2000:2010, "anomaly"] = 1

print("Dataset size:", len(df))
print(df.head())

print("\nSTEP 2 — Encoding categorical features")

enc_pod = LabelEncoder()
enc_event = LabelEncoder()
enc_type = LabelEncoder()

df["pod"] = enc_pod.fit_transform(df["pod"])
df["event"] = enc_event.fit_transform(df["event"])
df["eventType"] = enc_type.fit_transform(df["eventType"])

print("\nSTEP 3 — Normalizing features")

features = ["pod", "event", "eventType", "cpuUsage", "memoryUsage"]

scaler = StandardScaler()
data = scaler.fit_transform(df[features])

print("\nSTEP 4 — Creating sliding windows")

window = 20
X = []
labels = []

for i in range(len(data) - window):
    X.append(data[i:i + window])
    labels.append(df["anomaly"].iloc[i + window])

X = np.array(X)
labels = np.array(labels)

print("Windowed dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, shuffle=False
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

print("\nSTEP 5 — Building Transformer model")

class TransformerAE(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=64,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = TransformerAE(X.shape[2])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nSTEP 6 — Training model")

epochs = 15

for epoch in range(epochs):

    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)
    loss.backward()
    optimizer.step()

    print("Epoch", epoch, "Loss:", round(loss.item(), 4))

print("\nSTEP 7 — Computing anomaly scores")

model.eval()

with torch.no_grad():
    recon_train = model(X_train_tensor)
    recon_test = model(X_test_tensor)

train_errors = torch.mean((X_train_tensor - recon_train) ** 2, dim=(1, 2)).numpy()
test_errors = torch.mean((X_test_tensor - recon_test) ** 2, dim=(1, 2)).numpy()

threshold = np.mean(train_errors) + 3 * np.std(train_errors)

test_pred = (test_errors > threshold).astype(int)

precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)

print("\nEvaluation Metrics")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))

print("\nSTEP 8 — Visualization")

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

axs[0].plot(df["cpuUsage"])
axs[0].set_title("CPU Usage")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("CPU")

axs[1].plot(df["memoryUsage"])
axs[1].set_title("Memory Usage")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Memory")

axs[2].plot(test_errors, label="Anomaly Score")
axs[2].axhline(threshold, color="red", label="Threshold")

anomaly_points = np.where(test_pred == 1)[0]
axs[2].scatter(anomaly_points, test_errors[test_pred == 1], color="red")

axs[2].set_title("Detected Anomalies")
axs[2].set_xlabel("Time Window")
axs[2].set_ylabel("Reconstruction Error")

axs[2].legend()

plt.tight_layout()
plt.show()

print("\nPipeline complete.")
