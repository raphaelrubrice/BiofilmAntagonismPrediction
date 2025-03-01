# **Biofilm Antagonism Prediction (BAP)**
Code for the paper :
## **A machine learning approach to predict the antagonism of beneficial strains against microbial pathogens using morphological descriptors of their biofilms.**

### **📌 Abstract**
Biofilms are structured microbial communities that promote cell interactions through close spatial organization, leading to cooperative or competitive behaviors. 
Predicting microbial interactions in biofilms could aid in developing innovative strategies to control undesirable bacteria. Here, we present a machine learning approach to predict the antagonistic effects of beneficial bacterial candidates _Bacillus_ and _Paenibacillus_ species against undesirable bacteria (_Staphylococcus aureus_, _Enterococcus cecorum_, _Escherichia coli_ and _Salmonella enterica_), based on the morphological descriptors of single-species biofilms. 
We trained the models using quantitative features (e.g. biofilm volume, thickness, roughness or substratum coverage). As a proxy for antagonism, an exclusion score was used as the supervised training target. The latter was calculated based on the ratio of biofilm volume between the undesirable bacteria growing and the beneficial strain. 
Among the tested models, the XGBoost algorithm demonstrated the highest accuracy. The resulting model analysis highlights the importance of biofilm formation context when predicting antagonism. 
Our results demonstrate that machine learning can provide an efficient, data-driven tool to predict microbial interactions within biofilms and support the selection of beneficial strains for biofilm control. This approach enables scalable screening of microbial interactions, applicable in agriculture, healthcare, and industrial microbiology.

---

## **⚡ Installation**

### **📌 Prerequisites**
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

### 🔧 **Installing LightGBM (CUDA) & cuML**

Run the following script before using the repository to ensure LightGBM and cuML are properly set up:

```bash
bash install_script.sh <path_to_clone_LightGBM>
```

Replace `<path_to_clone_LightGBM>` with the desired location where LightGBM should be cloned and built.

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

