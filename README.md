# ARBeaconEvaluation

Evaluation scripts and analysis for the Beacon-Assist AR Synchronization project.

## 🔍 Core Analysis Tools

### [helper.py](helper.py) 
3D transform computational utilities:
- **Pose Averaging**: Weighted position/orientation merging
- **Relative Transforms**: Homogeneous matrix operations
- **Error Metrics**:
  - Euclidean distance (meters)
  - Angular differences (radians)
  - Localization latency (seconds)

## 📊 Evaluation Notebooks

| Notebook | Description | 
|----------|-------------|
| [22Jan2025.ipynb](22Jan2025.ipynb) | BLE-assist AR Sync evaluation |
| [12Mar2025.ipynb](12Mar2025.ipynb) | UWB ranging stability analysis |
| [19Mar2025.ipynb](19Mar2025.ipynb) | UWB session-to-session position jitter and ranging stability delay |
| [21Mar2025.ipynb](21Mar2025.ipynb) | Full beacon-assist AR synchronization evaluation | 
| [24Mar2025.ipynb](24Mar2025.ipynb) | UWB-assist AR Sync reference pose accuracy |

## 📑 Raw Logs
You may find raw logs from iOS apps in [`\Logs`](Logs) directory
