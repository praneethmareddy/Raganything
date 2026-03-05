# ===== PROMETHEUS METRIC SIMULATION + TRANSFORMER ANOMALY DETECTION =====

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 1. Simulate Prometheus metrics
# -----------------------------
np.random.seed(42)

time_steps = 2000

cpu = np.random.normal(40,5,time_steps)
memory = np.random.normal(60,4,time_steps)

# Inject anomalies
cpu[600:620] += 40
memory[1200:1220] += 30

df = pd.DataFrame({
    "cpu": cpu,
    "memory": memory
})

data = df.values


# -----------------------------
# 2. Create sliding windows
# -----------------------------
window_size = 20

X = []
for i in range(len(data)-window_size):
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
