from ultralytics import YOLO
from PIL import Image
import cv2
import torch

cap = cv2.VideoCapture(0)
model = YOLO("best.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(rgb_frame)
    
    results = model(pil_img)
    
    for r in results:
        try:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.item()
                    cls = box.cls.item()
                    
                    if conf > 0.8 and cls == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Fire {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        print("\nWarning Fire\n")
                        break
                    elif cls == 0 or cls == 1:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        color = (0, 0, 255) if cls == 0 else (255, 255, 0)
                        label = 'Fire' if cls == 0 else 'Smoke'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        if cls == 0 and (torch.tensor([1.]) in boxes.cls):
                            print("\nWarning Fire\n")
                            break
                    else:
                        pass
        except Exception as e:
            print(f"Error: {e}")
            pass
    
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
