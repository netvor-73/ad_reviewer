from flask import Flask
from flask_restful import Resource, Api 
import torch
from torchvision import models, transforms
from torch import nn
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)
api = Api(app)


class AdReviewerApi(Resource):

    model = models.vgg16(pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

    def __init__(self) -> None:
        super().__init__()
        
        state_dict = torch.load('best.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()


    def get(self, url):

        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        # img = Image.open('nars.jpg').convert('RGB')

        img = self.transform(img)
        logps = self.model.forward(torch.unsqueeze(img, 0))
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)

        return {'top probability': str(top_p.numpy()[0, 0]),
                'top class': str(top_class.numpy()[0, 0])}

api.add_resource(AdReviewerApi, '/<path:url>')

if __name__ == '__main__':
    app.run(debug=True)