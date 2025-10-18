# Project Setup Guide

This guide will help you set up and run the project in a clean Python environment.

---

## 1. Create a Virtual Environment

First, create a virtual environment to isolate project dependencies.

```bash
python -m venv racist-detect_env
```

## 2. Activate Virtual Environment

### For Windows:

```bash
racist-detect_env\Scripts\activate
```

### For MacOS/Linux:

```bash
source racist-detect_env/bin/activate
```

## 3. Install required dependencies

```bash
pip install -r requirements.txt
```

## 4. Run 'main.py' for data preparation.

```bash
python main.py
```

## 5. Run 'model.py' for model training and evaluation.

```bash
python src/model.py
```

## 6. Check results and logs.

