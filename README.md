# 🔍 Zero-Shot Object Detection using OWL-ViT

This project implements real-time zero-shot object detection using the OWL-ViT model from Google. The system detects objects that are **not part of the standard COCO dataset** using natural language prompts like "a lightbulb" or "a gaming console" — without any retraining.

---

##  Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/zero-shot-owlvit
   cd zero-shot-owlvit
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision transformers timm opencv-python pillow
   ```

---

## Model Download & Usage

No manual download required — the OWL-ViT model is loaded via HuggingFace Transformers:

```python
from transformers import OwlViTProcessor, OwlViTForObjectDetection

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
```

This downloads and caches the model automatically.

---

## How to Run the Project

```bash
python zero-shot-vision-model.py
```

- Press `e` to edit object prompts live (e.g., "a monitor, a desk lamp")
- Press `q` to quit the app
- All predictions (labels, confidence, bounding boxes) are logged to `predictions.csv`

---


## How It Works

The system uses OWL-ViT, a vision transformer trained on image-text pairs. Each frame from the webcam is processed along with a list of user-defined text prompts. The model calculates image-text similarity and detects relevant objects with bounding boxes and confidence scores. Results are displayed in real time and logged to a CSV file for review.

The user can update prompts during runtime, making this a flexible system that adapts without needing to retrain the model.

---

## Challenges & Future Improvements

One of the biggest challenges was maintaining a smooth frame rate while running inference with a large transformer model. ONNX/TorchScript export and GPU acceleration could significantly improve this. Building a Streamlit dashboard would enhance usability, and additional features like bounding box filtering or multi-object tracking could make it even more powerful.

---

## Output

- Real-time object detection with bounding boxes
- Label + confidence overlay on live feed
- CSV log of predictions: `predictions.csv`

---

## Bonus Features Included

- [x] Live prompt editing
- [x] FPS monitoring
- [x] CSV logging of predictions
- [x] Modular, clean Python script

---

## Example Prompts

```text
"a lightbulb", "a matchstick", "a computer monitor", "a lion", "a gaming console"
```

Feel free to test your own prompts — anything outside of COCO works best!

---


