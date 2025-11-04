# ISAnWin — Processed Android Malware Dataset (CICMalDroid 2020) & Model Scripts

## 🧭 Overview
This repository contains a **processed version of the CICMalDroid 2020 Android malware dataset** along with the **ISAnWin** model scripts — a CNN-based backbone architecture integrated into a Siamese Neural Network.  

The repository includes:
- Processed malware image dataset derived from Android `.dex` files.
- Subfamily-level categorization using the **average hash (aHash)** similarity algorithm.
- Complete training, validation, and testing scripts for the ISAnWin model.
- The architectural diagram `ISAnWin.png`, representing the CNN backbone used in the Siamese Network.

---

## 📂 Dataset Description — Processed CICMalDroid 2020

### **Source**
The dataset is derived from the **CICMalDroid 2020** dataset originally published by the Canadian Institute for Cybersecurity (CIC).

### **Dataset Summary**
- **Families:** 4 malicious and 1 benign Android malware family.
- **Scope:** Each malware family includes image representations of Android applications converted from their `.dex` bytecode.
- **Original granularity:** Malware-family level.
- **Processed granularity:** Family and subfamily level (based on aHash similarity).

### **Processing Pipeline**
The dataset in this repository has been processed through the following stages:

1. **DEX Extraction** — Parsing the `.dex` (Dalvik Executable) files out of each APK.  
2. **Image Conversion** — Converting each `.dex` file into a grayscale image to represent the binary pattern visually.  
3. **Image Resizing** — Resizing all converted images to a uniform dimension for deep learning model compatibility.  
4. **Subfamily Grouping** — Applying the **average hash (aHash)** similarity algorithm to group visually similar samples into **subfamilies** within each malware family.

---

## 🧠 Model and Code — `Model Scripts` Folder

The `Model Scripts` directory contains the full Python implementation of the **ISAnWin** model and its supporting scripts.

| File | Description |
|------|--------------|
| `model.py` | Defines the **ISAnWin CNN backbone**, designed for feature extraction within a Siamese Neural Network. |
| `train.py` | Trains the ISAnWin model on the processed Android malware dataset. |
| `train_valid_tester.py` | Manages training, validation, and testing phases with performance metrics. |
| `Testthemodel.py` | Evaluates model accuracy and robustness on unseen samples. |

---

## 🧩 Architecture Diagram — `ISAnWin.png`

`ISAnWin.png` illustrates the **CNN-based backbone architecture** utilized within the Siamese Neural Network.  
It demonstrates:
- Sequential convolutional and pooling layers for hierarchical feature extraction.  
- A latent embedding layer for similarity computation between malware samples.  

---

## 📚 Citation

If you use this repository, dataset, or model in your research or publication, please cite both **this repository** and the work as follows:
Tayyab U, Khan FB, Khan A, Durad MH, Khan FA, Ali A. 2024. ISAnWin: inductive generalized zero-shot learning using deep CNN for malware detection across windows and android platforms. PeerJ Computer Science 10:e2604 https://doi.org/10.7717/peerj-cs.2604

### **Cite this Repository**
