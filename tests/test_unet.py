from strive_vision.models import Unet

import torch
import torch.nn as nn



def test_forward(hidden_channels, bn_channels, num_classes, test_number=1):
    print('-'*15, 'Forward Test:', test_number, '-'*15)
    model = Unet(3, hidden_channels=hidden_channels, bn_channels=bn_channels, num_classes=num_classes)
    print('Config:', model.config)
    
    x = torch.rand(2, 3, 64, 64)
    print('input shape:', x.shape)
    
    model.eval()
    
    with torch.no_grad():
        y = model(x)
    print('output shape:', y.shape)
    
    print('\n')
    
    
def test_shape(hidden_channels, bn_channels, num_classes, test_number=1):
    print('-'*15, 'Net Shape Test:', test_number, '-'*15)
    model = Unet(3, hidden_channels=hidden_channels, bn_channels=bn_channels, num_classes=num_classes)
    model.eval()
    print("Config:", model.config)
    
    params = list(model.parameters())
    layers_params_num = [p.numel() for p in params]
    net_params_num = sum(layers_params_num)
    
    print("Number of parameters:", net_params_num)
    
    print('\n')
    
    
def test_exceptions(input_tensor, test_number=1): 
    print('-'*15, 'Exceptions Test:', test_number, '-'*15)
    model = Unet(3)
    model.eval()
    print("Config:", model.config)
    print("Input tensor shape:", input_tensor.shape)
    
    with torch.no_grad():
        try:
            predict = model(input_tensor)
        except ValueError as e:
            print("ValueError:", e)
    
    print('\n')

    


if __name__ == '__main__':
    configs_dict = {
        "hidden_channels": [
            (16, 32),
            (16, 32, 64),
            (64, 100, 200),
            (64, 64, 128, 256, 512)
        ],
        "bn_channels": [64, 128, 400, 1024],
        "num_classes": [1, 1, 1, 1]
    }
    
    configs = [dict(zip(configs_dict.keys(), vals)) for vals in zip(*configs_dict.values())]
    
    input_tensors = [
        torch.rand(1, 3, 512, 512),
        torch.rand(3, 512, 512),
        torch.rand(1, 3, 8, 8),
        torch.rand(1, 3, 500, 500)
    ]
    
    n = 1
    for cfg in configs:
        #print('-'*15, 'Test:', n, '-'*15)
        #print('config:', cfg)
        test_forward(**cfg, test_number=n)
        n += 1
        
    test_shape((64, 128, 256, 512), 1024, 1)
    
    for i in range(len(input_tensors)):
        test_exceptions(input_tensors[i], i+1)
    