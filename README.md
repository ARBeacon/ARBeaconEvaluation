# ARBeaconEvaluation
Evaluation scripts and analysis for the Beacon-Assist AR Synchronization project.
![Image-2](https://github.com/user-attachments/assets/2038191f-f487-471b-b9bc-50df341d4c16)


## 🔍 Core Analysis Tools

### [helper.py](helper.py) 
![Image](https://github.com/user-attachments/assets/bc64d2db-9575-4a6c-8844-f31d5baf5932)

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

_Note: This README.md was refined with the assistance of [DeepSeek](https://www.deepseek.com)_
