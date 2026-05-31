# Writer ID Prediction using BiLSTM and Neural ODE

This repository contains the code and resources for the project **Writer ID Prediction using BiLSTM and Neural ODE**, developed at the Machine Learning and Data Analytics (MaD) Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Technische Fakultät.

## Authors
* Vishwa Balasubramanian
* Sayeda Fatema Tuj Zohura
* Sumon Kazi
* Symon Islam

## Project Goal
The primary objective of this project is **Writer-ID classification**. We aim to identify which person wrote a handwriting sample using time-series sensor data collected from an IMU-enhanced pen (STABILO). The pen captures:
* Accelerometer (Acc)
* Gyroscope (Gyro)
* Magnetometer (Mag)
* Force/Pressure

## Datasets
We experimented with two different writer-dependent datasets:
1. **OnHW-equations_dep dataset**
2. **OnHW-words500_dep dataset**

## Models
We implemented and compared two models for this sequence classification task:

### 1. Baseline Deep Learning Model: BiLSTM
* **Limitations Addressed:** Discrete-time modeling assumptions, difficulty handling sequences of different lengths, and higher memory/computational costs.
* **Configuration:** 2 layers, hidden size 128, bidirectional, tanh activation.

### 2. Advanced Model: ODE-RNN (Neural ODE)
* **Overview:** Models hidden states as a continuous-time dynamical system defined by a differential equation. It uses GRU-based recurrent updates for observation injection and separates system dynamics from measurement updates.
* **Architecture:** Hybrid Time-Aware Model comprising an ODE-RNN Encoder, Temporal Pooling, and a Classification Head.
* **Configuration:** Hidden size 128, RK4 solver, step size 1.0, tanh activation.

## Experimental Setup
* **Input Modality:** Variable-length IMU Time-series
* **Preprocessing:** Train-only Normalization, Downsampling (5x factor), Zero padding with length masking
* **Loss Function:** Cross-Entropy
* **Optimizer:** AdamW
* **Learning Rate:** 3 × 10⁻⁴
* **Batch Size:** 32
* **Epochs:** 10
* **Gradient Clipping:** 5.0

## Results

### Validation Accuracy

| Model | Dataset | Fold 0 Val Acc | Fold 1 Val Acc |
| :--- | :--- | :--- | :--- |
| **BiLSTM** | OnHW-equations_dep | 97.83% | 98.77% |
| **ODE-RNN** | OnHW-equations_dep | 96.08% | 98.11% |
| **BiLSTM** | OnHW-words500_dep | 96.21% | 99.35% |
| **ODE-RNN** | OnHW-words500_dep | 95.64% | 99.23% |

### Comparative Study: BiLSTM vs ODE-RNN

| Aspect | BiLSTM (Baseline) | ODE-RNN (Advanced) |
| :--- | :--- | :--- |
| **Accuracy** | Higher | Slightly lower |
| **Training Stability** | High | Moderate |
| **Computational Cost** | Low (Faster training) | Higher (More training time) |
| **Time Awareness** | Implicit | Explicit |
| **Sequence Length Trend**| Performs best on short inputs | Benefits from longer sequences |
| **Modeling Generality** | Moderate | High |

#### Training Time Example (Fold 0, OnHW-equations_dep)
* **BiLSTM:** 2 mins / epoch (Total: ~20 mins)
* **ODE-RNN:** 5.6 mins / epoch (Total: ~56 mins)

## Summary & Key Takeaways
1. **BiLSTM** serves as a very strong and highly accurate baseline.
2. **ODE-RNN** provides explicit time awareness and handles longer sequences better.
3. Accuracy is not the only metric to consider; there is a clear **efficiency–robustness trade-off**.
4. The ultimate model choice depends heavily on the underlying data properties and sequence lengths.
