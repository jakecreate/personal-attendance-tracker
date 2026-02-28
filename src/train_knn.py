import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import sqlite3
import torch
import cv2
import numpy as np

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from core.model import MobileFacenet
from facenet_pytorch import MTCNN  

device = torch.device('cpu')

mtcnn = MTCNN(image_size=112, margin=20, device=device, post_process=False)
mfn = MobileFacenet().to(device)

checkpoint = torch.load('model/trained_mfn/068.ckpt', map_location=device)
mfn.load_state_dict(checkpoint['net_state_dict'])
mfn.eval()

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
        embedding = mfn(face_tensor)
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
    connection = sqlite3.connect('../data/department.db')
    print('connected')
    cursor = connection.cursor()
    cursor.execute("SELECT name, embedding FROM course_section")
    rows = cursor.fetchall()
    connection.close()

    X, y = convert_data(rows)
    print('Training KNN.')
    encoder = LabelEncoder()
    knn = KNeighborsClassifier(n_neighbors=7, metric='cosine')

    y_encoded = encoder.fit_transform(y)
    knn.fit(X, y_encoded)
    
    joblib.dump(knn, 'model/trained_knn/my_model.joblib')
    joblib.dump(encoder, 'model/trained_knn/label_encoder.joblib')
    print('Traing complete & saved.')

