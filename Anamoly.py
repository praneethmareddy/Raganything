def detect_anomaly(pod, event, eventType, cpuUsage, memoryUsage):
    
    import pandas as pd
    import numpy as np
    import torch
    
    print("\n===== Incoming Telemetry =====")
    print("Pod:", pod)
    print("Event:", event)
    print("EventType:", eventType)
    print("CPU:", cpuUsage)
    print("Memory:", memoryUsage)

    # Create dataframe
    sample = pd.DataFrame(
        [[pod, event, eventType, cpuUsage, memoryUsage]],
        columns=["pod","event","eventType","cpuUsage","memoryUsage"]
    )

    # Encode categorical values
    sample["pod"] = enc_pod.transform(sample["pod"])
    sample["event"] = enc_event.transform(sample["event"])
    sample["eventType"] = enc_type.transform(sample["eventType"])

    # Normalize
    sample_scaled = scaler.transform(sample)

    # Create sequence window
    seq = np.tile(sample_scaled, (window,1))

    seq = torch.tensor([seq], dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        recon, _ = model(seq)

    error = torch.mean((seq - recon)**2).item()

    print("\nAnomaly Score:", round(error,6))
    print("Threshold:", round(threshold,6))

    if error > threshold:
        print("\n🚨 ANOMALY DETECTED")
    else:
        print("\n✅ NORMAL BEHAVIOR")

    return error
detect_anomaly(
"monitor-pod",
"app_monitoring",
"steady",
200,
350
)
