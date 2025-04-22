
# Ligand Activity Classification and Molecular Generation

This repository contains a complete machine learning pipeline for predicting ligand activity against the AChE protein using XGBoost and generating novel molecular structures using a Variational Autoencoder (VAE). It includes data preprocessing, model training, molecule generation, visualization, and evaluation.

---

## 🧠 Project Goals

- **Classify ligands** as active or inactive based on molecular descriptors.
- **Generate new ligands** using a VAE trained on SMILES representations.
- **Evaluate** the generated ligands based on chemical similarity and visual distribution (PCA).

---

## 📁 Directory Structure

```
Drug_discovery_project/
├── AChE_dataset/
│   ├── AChE_ligands_csv.csv              # Raw dataset with ligand data
│   └── dataset_with_descriptors.csv      # Descriptor-enhanced dataset
│
├── classification model_XGBOOST/
│   ├── classification task.ipynb         # Notebook for classification task
│   └── xgboost_AChE_model_classification.pkl  # Trained model
│
├── VAE model for generation/
│   ├── VAE smile generation.ipynb        # Notebook for molecule generation
│   ├── simple_smiles_vae.pt              # Trained VAE model
│   └── vae_training_loss.png             # VAE training loss visualization
│
├── generated_active_smiles.csv           # Output of generated molecules
├── Tanimoto similarity.png               # Chemical similarity comparison
├── PCA analysis for generated and original active smiles.png  # PCA visual
└── report.pdf                            # Final project report
```

---

## 📊 Models Used

### 🔹 XGBoost Classifier

- **Goal:** Predict ligand activity.
- **Input:** Molecular descriptors.
- **Output:** Binary classification (active/inactive).
- **Performance Metrics:** Accuracy, precision, recall (see notebook for results).

### 🔹 Variational Autoencoder (VAE)

- **Goal:** Generate novel SMILES strings similar to actives.
- **Input:** SMILES of active ligands.
- **Output:** New SMILES strings.
- **Evaluation:** Tanimoto similarity + PCA visualization.

---

## 🔧 Installation & Requirements

Ensure the following Python libraries are installed:

```bash
pip install numpy pandas scikit-learn xgboost rdkit-pypi matplotlib seaborn torch
```

---

## ▶️ How to Run


1. Open and run the notebooks:
   - `classification model_XGBOOST/classification task.ipynb`
   - `VAE model for generation/VAE smile generation.ipynb`

---

## 📌 Visual Results

### PCA Distribution
![PCA](PCA%20analysis%20for%20generated%20and%20original%20active%20smiles.png)

### Tanimoto Similarity
![Tanimoto](Tanimoto%20similarity.png)

---

## 🧬 Key Technologies

- **Languages:** Python
- **Libraries:** XGBoost, PyTorch, RDKit, Scikit-learn, Matplotlib
- **ML Concepts:** Supervised classification, autoencoders, chemical similarity

---

## 📄 Report

A complete explanation of the methods and outcomes can be found in the [project report](report.pdf).

---

## 🤝 Acknowledgements

This project was developed as part of a machine learning initiative focused on ligand-based drug modeling and generative chemistry.

