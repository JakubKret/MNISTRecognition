import torch
from torchvision import transforms
from PIL import Image
import os

from mnistModel import DigitRecognizer

def main():
    checkpointPath = 'checkpoint.pt'
    imagePath = 'customDigit.png'

    if not os.path.exists(checkpointPath):
        print('checkpoint file not exists')
        return
    if not os.path.exists(imagePath):
        print('image file not exists')
        return

    aiModel = DigitRecognizer()

    checkpoint = torch.load(checkpointPath, weights_only=False)
    aiModel.load_state_dict(checkpoint['modelState'])
    aiModel.eval()

    img = Image.open(imagePath).convert('L')

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    imgTensor = transform(img)

    imgTensor = imgTensor.view(-1,784)

    with torch.no_grad():
        rawOutput = aiModel(imgTensor)

    prediction = torch.argmax(rawOutput).item()

    print('Predicted:', prediction)

if __name__ == '__main__':
    main()