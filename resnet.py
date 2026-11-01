import torch
import torchvision.models as models

resnet_pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

resnet_scratch = models.resnet50(weights=None)

num_classes = 10
in_features = resnet_pretrained.fc.in_features
resnet_pretrained.fc = torch.nn.Linear(in_features, num_classes)

dummy_input = torch.randn(1, 3, 224, 224)
output = resnet_pretrained(dummy_input)
print("PyTorch Output Shape:", output.shape)  