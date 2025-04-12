import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import time
import csv

# Load OWL-ViT model
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval()
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Initial custom prompts  
custom_labels = ["a lightbulb", "a matchstick", "a computer monitor", "a lion", "a gaming console"]

# CSV Logging process
csv_file = open("predictions.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Label", "Confidence", "Box"])
frame_count = 0

# Open webcam or video 
cap = cv2.VideoCapture(0) #Replace 0 with the video path to detect objects in the video
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess with processor
    inputs = processor(text=custom_labels, images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([frame_rgb.shape[:-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.25)[0]

    # Calculate FPS 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Draw predictions
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{custom_labels[label]} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Log to CSV file
        csv_writer.writerow([frame_count, custom_labels[label], f"{score:.2f}", [round(x, 2) for x in box.tolist()]])

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Zero-Shot Detection - OWL-ViT", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        new_input = input("Enter new prompts (comma-separated): ")
        custom_labels = [x.strip() for x in new_input.split(',') if x.strip()]
        print(f"Updated prompts: {custom_labels}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
