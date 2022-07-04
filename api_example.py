from fastapi import FastAPI
from fastapi import Request
from fastapi import File
from fastapi import Form
import uvicorn
from fastapi import UploadFile
import torch
from torch import nn
from image_process import ImageProcessor
from PIL import Image
import pickle

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

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x
    
    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]

image_processor = ImageProcessor()
with open('image_decoder.pkl', 'rb') as f:
    image_decoder = pickle.load(f)

n_classes = len(image_decoder)
img_classifier = ImageClassifier(num_classes=n_classes, decoder=image_decoder)
img_classifier.load_state_dict(torch.load('image_model.pt', map_location='cpu'))

app = FastAPI()
@app.get('/example')
def dummy(x):
    print(x)
    return 'Hello'

@app.post('/example')
def dummy_post(image: UploadFile = File(...)):
    img = Image.open(image.file)
    processed_img = image_processor(img)
    prediction = img_classifier.predict(processed_img)
    probs = img_classifier.predict_proba(processed_img)
    classes = img_classifier.predict_classes(processed_img)
    print(prediction)
    print(probs)
    print(classes)
    return "Bye"

@app.post('/text')
def read_text(text: str = Form(...)):
    print(text)
    return "Text successfully read"


if __name__ == '__main__':
    uvicorn.run('api_example:app', host='0.0.0.0', port=8090)