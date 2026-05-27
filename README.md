# 🛠️ Environment Setup Guide (Improved & GitHub‑Ready)

This guide provides a clean, structured, and GitHub‑friendly setup process for building a Python 3.12 environment with Jupyter Notebook, along with essential machine learning and graph processing libraries.

---

## 📦 1. Build the Base Environment

```bash
# Pull the official Python 3.12 Docker image
docker pull python:3.12

# Switch to root user
su

# Update package list
apt update

# Install sudo
apt install sudo

# Install Python and pip
apt install python3
apt install python3-pip

# Install nano and create a test script
apt install nano
echo 'print("Mohammad-Rahimi")' > test.py

# Install Jupyter Notebook
apt install jupyter

# Install Git
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
Clone the repository from GitHub
```bash
git clone https://github.com/mohammadRahimi1993/ILP_SAGE
```

Navigate to Code Directory
Change to the code directory
```bash
cd ILP-SAGE-Version2
```

Run ILP-SAGE Framework
```bash
python main_analysis.py
python run_experiments.py
```

Run GraphSAGE-based Framework
```bash
python GraphSAGE.py
```

