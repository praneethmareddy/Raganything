import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)

# =====================================
# 1. Simulate realistic Prometheus data
# =====================================

time_steps = 3000
t = np.arange(time_steps)

# periodic workload patterns
cpu = 40 + 10*np.sin(t/50) + np.random.normal(0,2,time_steps)
memory = 60 + 5*np.sin(t/120) + np.random.normal(0,1.5,time_steps)
network = 100 + 20*np.sin(t/30) + np.random.normal(0,5,time_steps)
disk = 50 + 8*np.sin(t/70) + np.random.normal(0,2,time_steps)

# inject anomalies
cpu[800:820] += 40
memory[1500:1550] += np.linspace(0,30,50)   # memory leak
network[2200:2220] += 80
disk[2600:2610] += 50

df = pd.DataFrame({
    "cpu": cpu,
    "memory": memory,
    "network": network,
    "disk": disk
})

data = df.values


# =====================================
# 2. Create sliding windows
# =====================================

window_size = 30

X = []
for i in range(len(data)-window_size):
    X.append(data[i:i+window_size])

X = np.array(X)
X_tensor = torch.tensor(X,dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)


# =====================================
# 3. Transformer Autoencoder
# =====================================

class TransformerAutoencoder(nn.Module):

    def __init__(self,input_dim=4,hidden_dim=128):

        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=2,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=3)

        self.decoder = nn.Linear(input_dim,input_dim)

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = TransformerAutoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# =====================================
# 4. Train model
# =====================================

epochs = 12

for epoch in range(epochs):

    total_loss = 0

    for batch in loader:

        x = batch[0]

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output,x)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",round(total_loss,4))


# =====================================
# 5. Compute anomaly scores
# =====================================

model.eval()

with torch.no_grad():
    reconstructed = model(X_tensor)

errors = torch.mean((X_tensor - reconstructed)**2,dim=(1,2)).numpy()

threshold = np.mean(errors) + 3*np.std(errors)

anomalies = errors > threshold

print("Detected anomalies:",np.sum(anomalies))


# =====================================
# 6. Visualization
# =====================================

fig,axs = plt.subplots(5,1,figsize=(12,14))

# CPU
axs[0].plot(df["cpu"])
axs[0].set_title("CPU Usage")

# Memory
axs[1].plot(df["memory"])
axs[1].set_title("Memory Usage")

# Network
axs[2].plot(df["network"])
axs[2].set_title("Network Traffic")

# Disk
axs[3].plot(df["disk"])
axs[3].set_title("Disk IO")

# Anomaly score
axs[4].plot(errors,label="Anomaly Score")
axs[4].axhline(threshold,color="red",label="Threshold")
axs[4].scatter(np.where(anomalies)[0],errors[anomalies],color="red",s=10)
axs[4].set_title("Anomaly Detection")

axs[4].legend()

plt.tight_layout()
plt.show()for i in range(len(data)-window_size):
    X.append(data[i:i+window_size])

X = np.array(X)

# Convert to tensor
X_tensor = torch.tensor(X,dtype=torch.float32)

# DataLoader
dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)


# -----------------------------
# 3. Transformer Autoencoder
# -----------------------------
class TransformerAutoencoder(nn.Module):

    def __init__(self,input_dim=2,hidden_dim=64):

        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=2)

        self.decoder = nn.Linear(input_dim,input_dim)

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = TransformerAutoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# -----------------------------
# 4. Train model
# -----------------------------
epochs = 10

for epoch in range(epochs):

    total_loss = 0

    for batch in loader:

        x = batch[0]

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output,x)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss)


# -----------------------------
# 5. Compute anomaly scores
# -----------------------------
model.eval()

with torch.no_grad():

    reconstructed = model(X_tensor)

errors = torch.mean((X_tensor - reconstructed)**2,dim=(1,2))
errors = errors.numpy()


# -----------------------------
# 6. Detect anomalies
# -----------------------------
threshold = np.mean(errors) + 3*np.std(errors)

anomalies = errors > threshold

print("Detected anomalies:",np.sum(anomalies))


# -----------------------------
# 7. Plot anomaly scores
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(errors,label="Anomaly Score")
plt.axhline(threshold,color="red",label="Threshold")
plt.legend()
plt.title("Transformer-based Anomaly Detection")
plt.show()
