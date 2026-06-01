# YOLO Segmentation FastAPI Demo

This project demonstrates a complete single-class segmentation pipeline:

1. Train YOLO segmentation model on Kaggle GPU.
2. Download `best.pt`.
3. Serve predictions with FastAPI.

## Project Files

- `app.py`: FastAPI inference API.
- `requirements.txt`: Python dependencies.
- `kaggle_training_notebook.ipynb`: Kaggle training notebook template.
- `implementation_plan_yolo_segmentation_api.md`: implementation plan.
- `sample_images/`: local test images.

## 1) Train on Kaggle

1. Open `kaggle_training_notebook.ipynb` in Kaggle.
2. Enable GPU in notebook settings.
3. Attach your single-class YOLO segmentation dataset.
4. Run notebook cells to train and validate.
5. Download:
   - `/kaggle/working/runs/seg_demo/weights/best.pt`

## 2) Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Add Model Weights

Place trained model as:

```txt
best.pt
```

in project root, or set:

```bash
set MODEL_PATH=path\to\best.pt
```

Optional confidence threshold:

```bash
set CONF_THRESHOLD=0.25
```

## 4) Run API

```bash
uvicorn app:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`

## 5) Endpoint

### `POST /predict`

Upload an image file and receive:

- `class_id`
- `class_name`
- `confidence`
- `bbox` (`x1,y1,x2,y2`)
- `mask_polygon` (segmentation polygon points)

Example response:

```json
{
  "filename": "sample.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "butterfly",
      "confidence": 0.91,
      "bbox": {
        "x1": 120.5,
        "y1": 80.2,
        "x2": 420.8,
        "y2": 390.1
      },
      "mask_polygon": [
        [130.0, 85.0],
        [145.0, 90.0],
        [160.0, 100.0]
      ]
    }
  ]
}
```

## 6) Troubleshooting

- If `/predict` returns 503:
  - Ensure `best.pt` exists in project root or `MODEL_PATH` is set.
- If model is slow locally:
  - Keep using `yolov8n-seg.pt` trained weights and moderate image sizes.
