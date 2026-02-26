import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import sqlite3
import torch
import cv2
import numpy as np

import joblib
from sklearn.neighbors import KNeighborsClassifier
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

def convert_data(rows):
    embedding_list = []
    name_list = []
    for name, binary_emb in rows:
        embedding = np.frombuffer(binary_emb, dtype=np.float32)
        embedding_list.append(embedding)
        name_list.append(name)

    return np.array(embedding_list), np.array(name_list)

if __name__ == '__main__':
    connection = sqlite3.connect('data/department.db')
    print('connected')
    cursor = connection.cursor()
    cursor.execute("SELECT name, embedding FROM course_section")
    rows = cursor.fetchall()
    connection.close()

    X, y = convert_data(rows)
    print('Training KNN.')
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X, y)
    joblib.dump(model, 'model/trained_knn/my_model.joblib')
    print('Traing complete & saved.')

