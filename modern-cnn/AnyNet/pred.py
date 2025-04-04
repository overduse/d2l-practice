import torch
from model_anynet import RegNetX32, device, model_path
from dataset import test_data

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def predict():

    num_classes = 10

    model = RegNetX32(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    print(x.shape, type(y))
    with torch.no_grad():
        x = x.unsqueeze(0).to(device) # unsqueeze to (batch, channels, height, width)
        # x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == '__main__':
    predict()
