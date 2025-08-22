Code for the paper :
# **[A machine learning approach to predict the antagonism of beneficial strains against microbial pathogens using morphological descriptors of their biofilms.](https://www.sciencedirect.com/science/article/pii/S2667318525000133?via%3Dihub)**

### **📌 Abstract**
Biofilms are structured microbial communities that promote cell interactionsthrough close spatial organization, leading to cooperative or competitive behaviors. Predicting microbial interactions in biofilms could aid in developing innovative strategies to control undesirable bacteria. Here, we present amachine learning approach to predict the antagonistic effects of beneficial bacterial candidates *Bacillus* and *Paenibacillus* species against undesirable bacteria (*Staphylococcus aureus*, *Enterococcus cecorum*, *Escherichia coli* and *Salmonella enterica*), based on the morphological descriptors of single-species biofilms. We trained the models using quantitative features (e.g. biofilmvolume, thickness, roughness or substratum coverage). As a proxy for antagonism, an exclusion score was used as the supervised training target. The latter was calculated based on the ratio of biofilm volume between the undesirable bacteria growing and the beneficial strain. We then used diverse explainability  methods to analyze the resulting model and found insights highlighting the importance of biofilm formation context when predicting antagonism. Our results demonstrate that machine learning can provide an efficient, data-driven tool to predict microbial interactions within biofilms and support the selection of beneficial strains for biofilm control. This approach enables scalable screening of microbial interactions, applicable in agriculture, healthcare, and industrial microbiology.

---

## **⚡ Installation**

### **📌 Prerequisites**
- **GPU Access:** Required for optimal performance
- **CUDA-Compatible GPU:** Needed for accelerated training and preprocessing.

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

### **Create Hold-Out and Cross-Validation Sets**
```sh
python datasets.py --methods avg random combinatoric --mode 1
```

### **Model Selection & Training**
```sh
python model_selection.py --run 1 --concat_results 1 --mode ho
python analysis_plots.py "plot_model_selection"
python analysis_plots.py "summary_model_selection"
```

### **Preprocessing Selection**
```sh
python preprocess_selection.py --run 1 --concat_results 1 --mode ho
python analysis_plots.py "summary_preprocess_selection"
```

### **Native feature selection**
```sh
python native_feature_selection.py
python analysis_plots.py "plot_native_feature_selection"
```
### **Feature engineering and selection**
```sh
python feature_engineering.py
python analysis_plots.py "plot_feature_engineering"
```

### **Hyperparameter search**
```sh
python optuna_campaign.py
python analysis_plots.py "plot_optuna_study"
```

### **Ablation study**
```sh
python ablation_study.py
python analysis_plots.py "plot_ablation_study"
```

### **Error analysis**
```sh
python analysis_plots.py "plot_err_distrib"
python analysis_plots.py "plot_err_by_org"
```
### **SHAP values**
```sh
python analysis_plots.py "plot_global_SHAP"
python analysis_plots.py "plot_local_SHAP"
```
---

## **📊 Figures**
Figures will be saved in the `Plots/` directory after running the scripts.

---

## **📫 Contact & Citation**
You can cite our work as:

> **Citation**:
> @article{RUBRICE2025100137,  
title = {A machine learning framework for the prediction and analysis of bacterial antagonism in biofilms using morphological descriptors},  
journal = {Artificial Intelligence in the Life Sciences},  
pages = {100137},  
year = {2025},  
issn = {2667-3185},  
doi = {https://doi.org/10.1016/j.ailsci.2025.100137},  
url = {https://www.sciencedirect.com/science/article/pii/S2667318525000133},  
author = {Raphaël Rubrice and Virgile Gueneau and Romain Briandet and Antoine Cornuejols and Vincent Guigue},  
keywords = {Biofilm, Beneficial biofilms, Pathogens, Machine learning, Artificial intelligence, Explainability, Interpretability},  
}  

For questions contact `raphael.rubrice@agroparistech.fr` or `vincent.guigue@agroparistech.fr`.

---
