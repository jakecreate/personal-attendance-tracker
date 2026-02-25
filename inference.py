import os

import torch
import cv2
import numpy as np

from core.model import MobileFacenet
from facenet_pytorch import MTCNN  

device = torch.device('cpu')

mtcnn = MTCNN(image_size=112, margin=20, device=device, post_process=False)
model = MobileFacenet().to(device)

checkpoint = torch.load('model/best/068.ckpt', map_location=device)
model.load_state_dict(checkpoint['net_state_dict'])
model.eval()

def align_face(img_path):
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"Error: Could not load image {img_path}")
        return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    face_tensor = mtcnn(img_rgb) # (3, 112, 112)
    
    if face_tensor is None:
        print(f"No face detected in {img_path}")
        return None
        
    face_tensor = (face_tensor - 127.5) / 128.0
    return face_tensor.unsqueeze(0).to(device) # (1, 3, 112, 112)

def get_embedding(face_tensor):
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.cpu().numpy()

def check_match(embedding1, embedding2, threshold=0.6):
    a = embedding1.flatten()
    b = embedding2.flatten()
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return similarity > threshold, similarity

def live_recognition(avg_embs, emb_identity, threshold=0.6, skip_frames=3):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 24)
    print("starting live feed - press 'q' to quit")

    known_faces = [(avg_emb, emb_identity[tuple(avg_emb[0])]) for avg_emb in avg_embs]
    
    frame_count = 0
    last_boxes = None
    last_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            last_boxes = boxes
            last_labels = []

            if boxes is not None:
                face_tensors = mtcnn(frame_rgb)
                
                if face_tensors is not None:
                    if face_tensors.ndimension() == 3:
                        # mtcnn returns (3, 112, 112) -> (1, 3, 112, 112)
                        face_tensors = face_tensors.unsqueeze(0)
                    
                    face_tensors = (face_tensors - 127.5) / 128.0
                    
                    with torch.no_grad():
                        embeddings = model(face_tensors.to(device)).cpu().numpy()

                    for i in range(len(embeddings)):
                        current_emb = embeddings[i].reshape(1, -1)
                        best_name = "Unknown"
                        max_score = -1
                        
                        for avg_emb, name in known_faces:
                            match, score = check_match(current_emb, avg_emb, threshold)
                            if score > max_score:
                                max_score = score
                                if match:
                                    best_name = name
                        
                        last_labels.append((best_name, max_score))

        if last_boxes is not None:
            for i, box in enumerate(last_boxes):
                x1, y1, x2, y2 = box.astype(int)
                
                name = last_labels[i][0] if i < len(last_labels) else "Unknown"
                score = last_labels[i][1] if i < len(last_labels) else 0.0
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"{name} {score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('PAT', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    N = 10
    path = './faces'
    names = [name for name in os.listdir(path) if not name.startswith('.')]
    emb_identity = {}
    avg_embs = []

    for name in names:
        total_emb = 0
        for i in range(N):
            face = align_face(f'./faces/{name}/{i+1}.png')
            emb = get_embedding(face)
            total_emb += emb

        avg_emb = total_emb/N
        emb_identity[tuple(avg_emb[0])] = name
        avg_embs.append(avg_emb)

    live_recognition(avg_embs, emb_identity)

