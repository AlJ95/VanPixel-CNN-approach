import PIL
import torchvision.transforms as transforms
import numpy as np

# 15 Objekte
MEAN = (0.62, 0.62, 0.62)
STD = (0.12, 0.12, 0.12)


def forward(x, resize, dim_input):
    if resize:
        transform = transforms.Compose([
            transforms.Resize(dim_input[0]),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    x = PIL.Image.fromarray(np.array(x * 255 / np.max(x))
                            .astype('uint8'))
    x = transform(x)

    return x


def forward2(x, resize, dim_input):
    if resize:
        transform = transforms.Compose([
            transforms.Resize(dim_input[0]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    x = PIL.Image.fromarray(np.array(x * 255 / np.max(x))
                            .astype('uint8'))
    x = transform(x)

    return x.squeeze(0)
