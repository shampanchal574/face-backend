from fastapi import FastAPI, UploadFile, Form
import shutil
from face_model import FaceIdentifier

app = FastAPI()
model = FaceIdentifier()

@app.post("/identify")
async def identify(image: UploadFile):
    with open("frame.jpg", "wb") as f:
        shutil.copyfileobj(image.file, f)

    name, confidence = model.identify("frame.jpg")
    return {"name": name, "confidence": confidence}

@app.post("/train")
async def train(name: str = Form(...), image: UploadFile = Form(...)):
    with open("train.jpg", "wb") as f:
        shutil.copyfileobj(image.file, f)

    model.train("train.jpg", name)
    return {"status": "trained", "name": name}
