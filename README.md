# 🛠️ Environment Setup Guide (Improved & GitHub‑Ready)
This guide provides a clean, structured, and GitHub‑friendly setup process for building a Python 3.12 environment with Jupyter Notebook, along with essential machine learning and graph processing libraries.

---

## 📦 1. Build the Base Environment

```bash
docker pull python:3.12
su
apt update
apt install sudo
apt install python3
apt install python3-pip
apt install nano
echo 'print("Mohammad-Rahimi")' > test.py
apt install jupyter
apt update && apt install -y git
```

---
## 🧪 2. Create and Configure a Virtual Environment
```bash
# Create the virtual environment
python3 -m venv myenv

# Activate it
source myenv/bin/activate

# Upgrade pip and install the Jupyter kernel
python3 -m pip install --upgrade pip
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=dgl_env --display-name "Python (dgl_env)"
```

---
## 📚 3. Install Required Python Packages

```bash
pip install \
  numpy==1.26.4 \
  torch==2.2.0 \
  torchvision==0.17.0 \
  torchaudio==2.2.0 \
  torcheval \
  dgl -f https://data.dgl.ai/wheels/torch-2.2/repo.html \
  torchmetrics==1.2.1 \
  transformers==4.38.0 \
  scikit-learn==1.6.1 
```

## 📊 4.Running GraphSAGE and ILP-SAGE Frameworks
Clone the repository from GitHub and navigate to the code directory.
```bash
git clone https://github.com/mohammadRahimi1993/ILP_SAGE
cd ILP-SAGE-Version2
```

Run ILP-SAGE Framework And GraphSAGE-based Framework
```bash
python main_analysis.py
python run_experiments.py
python GraphSAGE.py
```

