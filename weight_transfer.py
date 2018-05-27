from segnet import SegNet
import torch

def transfer_pretrained_weighted():
    model = SegNet()
    corresp_name ={
        'features.0.weight': 'vgg16_block1.0.weight',
        'features.0.bias': 'vgg16_block1.0.bias',
        'features.1.weight': 'vgg16_block1.1.weight',
        'features.1.bias': 'vgg16_block1.1.bias',
        'features.1.running_mean': 'vgg16_block1.1.running_mean',
        'features.1.running_var': 'vgg16_block1.1.running_var',
        'features.3.weight': 'vgg16_block1.3.weight',
        'features.3.bias': 'vgg16_block1.3.bias',
        'features.4.weight': 'vgg16_block1.4.weight',
        'features.4.bias': 'vgg16_block1.4.bias',
        'features.4.running_mean': 'vgg16_block1.4.running_mean',
        'features.4.running_var': 'vgg16_block1.4.running_var',
        'features.7.weight': 'vgg16_block2.0.weight',
        'features.7.bias': 'vgg16_block2.0.bias',
        'features.8.weight': 'vgg16_block2.1.weight',
        'features.8.bias': 'vgg16_block2.1.bias',
        'features.8.running_mean': 'vgg16_block2.1.running_mean',
        'features.8.running_var': 'vgg16_block2.1.running_var',
        'features.10.weight': 'vgg16_block2.3.weight',
        'features.10.bias': 'vgg16_block2.3.bias',
        'features.11.weight': 'vgg16_block2.4.weight',
        'features.11.bias': 'vgg16_block2.4.bias',
        'features.11.running_mean': 'vgg16_block2.4.running_mean',
        'features.11.running_var': 'vgg16_block2.4.running_var',
        'features.14.weight': 'vgg16_block3.0.weight',
        'features.14.bias': 'vgg16_block3.0.bias',
        'features.15.weight': 'vgg16_block3.1.weight',
        'features.15.bias': 'vgg16_block3.1.bias',
        'features.15.running_mean': 'vgg16_block3.1.running_mean',
        'features.15.running_var': 'vgg16_block3.1.running_var',
        'features.17.weight': 'vgg16_block3.3.weight',
        'features.17.bias': 'vgg16_block3.3.bias',
        'features.18.weight': 'vgg16_block3.4.weight',
        'features.18.bias': 'vgg16_block3.4.bias',
        'features.18.running_mean': 'vgg16_block3.4.running_mean',
        'features.18.running_var': 'vgg16_block3.4.running_var',
        'features.20.weight': 'vgg16_block3.6.weight',
        'features.20.bias': 'vgg16_block3.6.bias',
        'features.21.weight': 'vgg16_block3.7.weight',
        'features.21.bias': 'vgg16_block3.7.bias',
        'features.21.running_mean': 'vgg16_block3.7.running_mean',
        'features.21.running_var': 'vgg16_block3.7.running_var',
        'features.24.weight': 'vgg16_block4.0.weight',
        'features.24.bias': 'vgg16_block4.0.bias',
        'features.25.weight': 'vgg16_block4.1.weight',
        'features.25.bias': 'vgg16_block4.1.bias',
        'features.25.running_mean': 'vgg16_block4.1.running_mean',
        'features.25.running_var': 'vgg16_block4.1.running_var',
        'features.27.weight': 'vgg16_block4.3.weight',
        'features.27.bias': 'vgg16_block4.3.bias',
        'features.28.weight': 'vgg16_block4.4.weight',
        'features.28.bias': 'vgg16_block4.4.bias',
        'features.28.running_mean': 'vgg16_block4.4.running_mean',
        'features.28.running_var': 'vgg16_block4.4.running_var',
        'features.30.weight': 'vgg16_block4.6.weight',
        'features.30.bias': 'vgg16_block4.6.bias',
        'features.31.weight': 'vgg16_block4.7.weight',
        'features.31.bias': 'vgg16_block4.7.bias',
        'features.31.running_mean': 'vgg16_block4.7.running_mean',
        'features.31.running_var': 'vgg16_block4.7.running_var',
        'features.34.weight': 'vgg16_block5.0.weight',
        'features.34.bias': 'vgg16_block5.0.bias',
        'features.35.weight': 'vgg16_block5.1.weight',
        'features.35.bias': 'vgg16_block5.1.bias',
        'features.35.running_mean': 'vgg16_block5.1.running_mean',
        'features.35.running_var': 'vgg16_block5.1.running_var',
        'features.37.weight': 'vgg16_block5.3.weight',
        'features.37.bias': 'vgg16_block5.3.bias',
        'features.38.weight': 'vgg16_block5.4.weight',
        'features.38.bias': 'vgg16_block5.4.bias',
        'features.38.running_mean': 'vgg16_block5.4.running_mean',
        'features.38.running_var': 'vgg16_block5.4.running_var',
        'features.40.weight': 'vgg16_block5.6.weight',
        'features.40.bias': 'vgg16_block5.6.bias',
        'features.41.weight': 'vgg16_block5.7.weight',
        'features.41.bias': 'vgg16_block5.7.bias',
        'features.41.running_mean': 'vgg16_block5.7.running_mean',
        'features.41.running_var': 'vgg16_block5.7.running_var',
    }
    s_dict = model.state_dict()
    pretrained_dict = torch.load('vgg16_bn-6c64b313.pth')  # you have to download pretrained model weight pth
    for name in pretrained_dict:
        if name not in corresp_name:
            continue
        s_dict[corresp_name[name]] = pretrained_dict[name]
    model.load_state_dict(s_dict)
    torch.save(model.state_dict(), 'transfer-vgg16-for11classes.pth')


if __name__ == '__main__':
    transfer_pretrained_weighted()