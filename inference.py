import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import cv2
import numpy as np

import joblib
from core.model import MobileFacenet
from facenet_pytorch import MTCNN  

cpu = torch.device('cpu')
mps = torch.device('mps')

mtcnn = MTCNN(image_size=112, margin=20, device=cpu, post_process=False)
mfn = MobileFacenet().to(mps)
knn = joblib.load('model/trained_knn/my_model.joblib')
le = joblib.load('model/trained_knn/label_encoder.joblib')


checkpoint = torch.load('model/best/068.ckpt', map_location=mps)
mfn.load_state_dict(checkpoint['net_state_dict'])
mfn.eval()

def live_recognition(threshold=0.65):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("starting live feed - press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('camera not detected')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            # filter closest face
            idx = 0
            if len(boxes) > 1: 
                idx = np.argmax(np.abs(boxes[:, 0] - boxes[:, 2]) * np.abs(boxes[:, 1] - boxes[:, 3]))
            box = boxes[idx]

            face_tensors = mtcnn(frame_rgb) # (3, 112, 112) 
            if face_tensors is not None:
                if face_tensors.ndimension() == 3:
                    face_tensors = face_tensors.unsqueeze(0) # (1, 3, 112, 112)

                face_tensors = (face_tensors - 127.5) / 128.0

                with torch.no_grad():
                    embeddings = mfn(face_tensors.to(mps)).cpu().numpy() # (n, 256)

                distances, idxs = knn.kneighbors(embeddings, n_neighbors=5)
                sims = 1 - distances[0]
                
                nn_labels = knn._y[idxs[0]]
                pred = knn.predict(embeddings)[0]

                mask = nn_labels == pred 
                if any(mask):
                    avg_sim = np.mean(sims[mask])
                else:
                    avg_sim = 0

                candidate_name = le.inverse_transform([pred])[0]
                name = candidate_name if avg_sim > threshold else "Unknown"

            # display
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            text = f"{name} {avg_sim*100:.2f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('PAT', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # start
    live_recognition()
    # save log & export
    

