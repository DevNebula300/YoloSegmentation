import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw, UnidentifiedImageError

# Keep Ultralytics runtime files inside the project to avoid AppData permission issues.
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".ultralytics"))

from ultralytics import YOLO


APP_TITLE = "YOLO Segmentation API"
APP_VERSION = "1.0.0"
DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_CONFIDENCE = 0.25

BASE_DIR = Path(__file__).resolve().parent
DEMO_PAGE_PATH = BASE_DIR / "static" / "demo.html"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
CONFIDENCE_THRESHOLD = float(os.getenv("CONF_THRESHOLD", DEFAULT_CONFIDENCE))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

model: YOLO | None = None
model_load_error: str | None = None


def _load_model() -> None:
    global model, model_load_error

    if not os.path.exists(MODEL_PATH):
        model_load_error = (
            f"Model file not found at '{MODEL_PATH}'. "
            "Train and copy best.pt into project root or set MODEL_PATH."
        )
        return

    try:
        model = YOLO(MODEL_PATH)
        model_load_error = None
    except Exception as exc:  # pragma: no cover
        model = None
        model_load_error = f"Failed to load model: {exc}"


def _class_name(class_id: int) -> str:
    if model is None:
        return str(class_id)

    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _mask_polygon(result: Any, index: int) -> list[list[float]]:
    if result.masks is None or result.masks.xy is None:
        return []
    if index >= len(result.masks.xy):
        return []

    polygon_array = result.masks.xy[index]
    polygon: list[list[float]] = []

    for point in polygon_array:
        x, y = float(point[0]), float(point[1])
        polygon.append([round(x, 2), round(y, 2)])

    return polygon


def _serialize_result(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []

    detections: list[dict[str, Any]] = []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]

        detections.append(
            {
                "class_id": cls_id,
                "class_name": _class_name(cls_id),
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
                "mask_polygon": _mask_polygon(result, i),
            }
        )

    return detections


async def _read_uploaded_image(file: UploadFile) -> tuple[Image.Image, np.ndarray]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Upload an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    return pil_image, np.array(pil_image)


def _run_inference(image_np: np.ndarray) -> Any:
    try:
        return model.predict(source=image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


def _render_visualization(pil_image: Image.Image, result: Any) -> Image.Image:
    rendered = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_draw = ImageDraw.Draw(rendered)

    if result.boxes is None:
        return pil_image

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        polygon = _mask_polygon(result, i)
        polygon_xy = [
            (float(x), float(y))
            for x, y in polygon
            if np.isfinite(x) and np.isfinite(y)
        ]

        if len(polygon_xy) >= 3:
            try:
                overlay_draw.polygon(
                    polygon_xy,
                    fill=(0, 210, 255, 80),
                    outline=(0, 170, 240, 200),
                )
            except ValueError:
                # Keep visualization endpoint resilient even if a mask polygon is malformed.
                pass

        overlay_draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 70, 70, 255),
            width=3,
        )
        label = f"{_class_name(cls_id)} {confidence:.2f}"
        text_draw.text((x1 + 4, max(0.0, y1 - 16)), label, fill=(255, 255, 255, 255))

    composited = Image.alpha_composite(rendered, overlay)
    return composited.convert("RGB")


@app.on_event("startup")
def on_startup() -> None:
    _load_model()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
    }


@app.get("/demo", response_class=HTMLResponse)
def demo() -> HTMLResponse:
    if not DEMO_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo page not found.")
    return HTMLResponse(DEMO_PAGE_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error
            or "Model is not loaded. Check MODEL_PATH/best.pt availability.",
        )

    _, image_np = await _read_uploaded_image(file)
    results = _run_inference(image_np)

    detections = _serialize_result(results[0]) if results else []

    return {
        "filename": file.filename or "uploaded_image",
        "detections": detections,
    }


@app.post("/predict/visualize")
async def predict_visualize(file: UploadFile = File(...)) -> StreamingResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error
            or "Model is not loaded. Check MODEL_PATH/best.pt availability.",
        )

    pil_image, image_np = await _read_uploaded_image(file)
    results = _run_inference(image_np)
    result = results[0] if results else None

    output_image = _render_visualization(pil_image, result) if result is not None else pil_image
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{(file.filename or "prediction").rsplit(".", 1)[0]}_pred.png"'
        },
    )
