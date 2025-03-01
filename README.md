# **Biofilm Antagonism Prediction (BAP)**

## **📝 A machine learning approach to predict the antagonism of beneficial strains against microbial pathogens using morphological descriptors of their biofilms.**

### **📌 Introduction**
Biofilms play a crucial role in microbial ecology, healthcare, and industry. Understanding interactions between bacterial species in biofilm environments is essential for developing novel antimicrobial strategies. This repository provides a machine learning pipeline for predicting biofilm antagonism using curated datasets. The framework supports model selection, preprocessing optimization, and performance evaluation to identify the best predictive approach.

---

## **⚡ Installation**

### **📌 Prerequisites**
- **Operating System:** Linux or Windows with WSL2 (recommended for GPU support)
- **GPU Access:** Required for optimal performance
- **CUDA-Compatible GPU:** Needed for accelerated training with LightGBM and other models
- **CUDA Built LightGBM:** Ensure you have a CUDA-enabled version of LightGBM installed

### **🔧 Setting Up the Environment**

Using **conda**:
```sh
conda env create -f environment.yaml
conda activate BAP
```

Using **pip** and virtual environments:
```sh
python -m venv BAP_env
source BAP_env/bin/activate  # Windows: BAP_env\Scripts\activate
pip install -r requirements.txt
```

To verify GPU support:
```python
import lightgbm as lgb
print(lgb.__version__)
print(lgb.__file__)
```
Ensure the output points to a CUDA-enabled LightGBM installation.

---

## **🚀 Running Experiments**

### **1️⃣ Create Hold-Out and Cross-Validation Sets**
```sh
python datasets.py --methods avg random combinatoric --mode cv
python datasets.py --methods avg random combinatoric --mode ho
```

### **2️⃣ Model Selection & Training**
```sh
python model_selection.py --run 1 --concat_results 1 --mode ho
```

### **3️⃣ Preprocessing Selection**
```sh
python preprocess_selection.py --run 1 --concat_results 1 --mode ho
```

### **4️⃣ Generate and Save Figures**
```sh
python plots.py --metrics MAE RMSE R2 --methods avg random combinatoric --plot_model_selection 1 --plot_preprocess_selection 1
```

---

## **📊 Figures**
Figures will be saved in the `Plots/` directory after running the scripts.

---

## **📫 Contact & Citation**
If you use this work, please cite our paper:

> **[Your Paper Citation Here]**

For questions or issues, open an issue in this repository or contact **[Your Name]**.

---

🎯 *Happy reviewing!* 🚀

