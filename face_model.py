import torch
import numpy as np
import pickle
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

DB_PATH = "/data/face_db.pkl"

class FaceIdentifier:
    def __init__(self, threshold=0.7):
        self.device = "cpu"
        self.detector = MTCNN(image_size=160, device=self.device)
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval()
        self.threshold = threshold
        self.embeddings = []
        self.labels = []
        self.load()

    def _embed(self, img_path):
        img = Image.open(img_path).convert("RGB")
        face = self.detector(img)
        if face is None:
            return None
        with torch.no_grad():
            emb = self.encoder(face.unsqueeze(0))
        return emb.numpy()[0]

    def identify(self, img_path):
        emb = self._embed(img_path)
        if emb is None or not self.embeddings:
            return "Unknown", 0.0

        sims = cosine_similarity([emb], self.embeddings)[0]
        idx = np.argmax(sims)

        if sims[idx] >= self.threshold:
            return self.labels[idx], float(sims[idx])
        return "Unknown", float(sims[idx])

    def train(self, img_path, name):
        emb = self._embed(img_path)
        if emb is not None:
            self.embeddings.append(emb)
            self.labels.append(name)
            self.save()

    def save(self):
        os.makedirs("/data", exist_ok=True)
        with open(DB_PATH, "wb") as f:
            pickle.dump((self.embeddings, self.labels), f)

    def load(self):
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                self.embeddings, self.labels = pickle.load(f)
