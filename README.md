# 🔍 Zero-Shot Object Detection using OWL-ViT

This project uses OWL-ViT (Open-World Localization Vision Transformer) for **zero-shot object detection** on live webcam or video.  
It can detect custom objects **not seen during training** by simply using natural language prompts — no retraining needed.

---

##  Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/AlokDhanush/Zero_shot_object_detection


2. **Create and activate a virtual environment (optional but recommended)**

python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate


3. **Install required dependencies**

pip install torch torchvision transformers timm opencv-python pillow


4. **Model Download & Usage**

No manual download is needed — the OWL-ViT model is automatically downloaded via transformers.
In the script:
from transformers import OwlViTProcessor, OwlViTForObjectDetection

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

This will automatically:
*Download and cache the model weights
*Load the processor for pre/post-processing


5. **How to Run**

python zero_shot_detection.py

Features:
*Press e to enter new prompts (e.g. "a drone, a lamp")
*Press q to quit the app
*Predictions are logged in predictions.csv
*Real-time FPS is shown on the video feed


6. **Example Prompts**
[a lightbulb, a matchstick, a computer monitor, a lion, a gaming console]


7. **Output**
*Real-time object detection with labels and bounding boxes
*Detection logs saved to predictions.csv


8.**To Do (Optional Enhancements)**
*Export to ONNX for speed
*Streamlit UI for dashboard
*Add bounding box filtering by confidence


**How It Works**
This project uses the OWL-ViT (Open-World Vision Transformer) model from Google, which enables zero-shot object detection using natural language prompts. The model processes each video frame from a webcam or video file and matches visual features to custom object descriptions (e.g., "a lightbulb", "a gaming console"). The system runs inference in real time, draws bounding boxes and confidence scores for matching prompts, and logs all predictions to a CSV file. Users can dynamically update the object categories (prompts) during runtime by pressing a key, making the system adaptable without retraining or reloading the model.

**Challenges Faced & What Can Be Improved**
One of the main challenges was balancing detection accuracy with real-time performance, especially since OWL-ViT is a large model that can be slow on CPUs. Another challenge was the lack of built-in support for certain features like prompt editing or detection logging, which had to be manually implemented. Future improvements could include exporting the model to ONNX or TorchScript for faster inference, building a simple Streamlit-based UI for interactive use, and integrating region proposals or confidence-based filtering for more efficient detection. Support for GPU acceleration or using a distilled/lightweight version of OWL-ViT would also significantly boost FPS and responsiveness.



