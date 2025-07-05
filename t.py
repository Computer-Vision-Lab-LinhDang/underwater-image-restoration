import torchvision.models as models

# Load VGG19 pretrained
vgg19 = models.vgg19(pretrained=True)

# In toàn bộ kiến trúc
print(vgg19.features)
