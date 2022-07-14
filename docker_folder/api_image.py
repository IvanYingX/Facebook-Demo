import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from image_process import ImageProcessor


class ImageClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        device = "cpu"
        out_features = resnet50.fc.out_features
        self.linear = nn.Linear(out_features, num_classes).to(device)
        self.main = nn.Sequential(resnet50, self.linear).to(device)
        self.decoder = decoder

    def forward(self, inp):
        x = self.main(inp)
        return x

    def predict(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return x

    def predict_proba(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return torch.softmax(x, dim=1)

    def predict_classes(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return self.decoder[int(torch.argmax(x, dim=1))]

try:
    with open('image_decoder.pkl', 'rb') as f:
        image_decoder = pickle.load(f)
    image_model = ImageClassifier(num_classes=len(image_decoder), decoder=image_decoder)
    image_model.load_state_dict(torch.load('image_model.pt', map_location='cpu'))
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    image_processor = ImageProcessor()
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    image_tensor = image_processor(pil_image)
    probs = image_model.predict_proba(image_tensor)
    probs = [str(round(x, 2)) for x in probs[0].tolist()]
    image_category = image_model.predict_classes(image_tensor)

    return JSONResponse(content={
    "Predictions": image_category, #return predictions here
    "Probabilities": probs #return probabilities here
        })
    
if __name__ == '__main__':
  uvicorn.run("api_image:app", host="0.0.0.0", port=8080)