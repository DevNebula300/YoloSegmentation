# YOLO Segmentation FastAPI Demo

This project demonstrates a complete single-class segmentation pipeline:

1. Train YOLO segmentation model on Kaggle GPU.
2. Download `best.pt`.
3. Serve predictions with FastAPI.

## 1) Local Setup

````bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

```bash
set CONF_THRESHOLD=0.25
````

## 2) Run API

```bash
uvicorn app:app --reload
```

Open:

- `http://127.0.0.1:8000/demo`
