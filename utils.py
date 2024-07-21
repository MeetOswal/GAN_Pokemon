import warnings
import torchvision

batch_size = 256

transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5) # 0.5 mean and 0.5 SD
])

