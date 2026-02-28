import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import sqlite3
import cv2
import torch
import numpy as np
from core.model import MobileFacenet
from facenet_pytorch import MTCNN  

device = torch.device('cpu')
mtcnn = MTCNN(image_size=112, margin=20, device=device, post_process=False)
print('MTCNN loaded.')
model = MobileFacenet().to(device)
checkpoint = torch.load('model/trained_mfn/068.ckpt', map_location=device)
model.load_state_dict(checkpoint['net_state_dict'])
model.eval()
print('MobileFaceNet loaded')

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

   
if __name__ == '__main__':
    N_IMAGES = 10
    path = '../faces'
    names = [name for name in os.listdir(path) if not name.startswith('.')]
    print('student names:', names)

    connection = sqlite3.connect('../data/department.db')
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS course_section (
            sid INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    ''')

    for name in names:
        embeddings = []
        for i in range(N_IMAGES):
            face = align_face(f'{path}/{name}/{i+1}.png')
            emb = get_embedding(face).tobytes()
            embeddings.append((name, emb))
        cursor.executemany('INSERT INTO course_section (name, embedding) VALUES (?, ?)', embeddings)
        connection.commit()
    connection.close()
    print('Data has been added')


