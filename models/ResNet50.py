import torch
from torchvision import models
from torch import nn

class ResNet50_Model(nn.Module):
    def __init__(self,model_cfg_data):
        super(ResNet50_Model, self).__init__()
        self.model = models.resnet50(weights=None)

        if hasattr(self.model,"fc"):
            num_ftrs = self.model.fc.in_features
            self.fc_exists = True
        else:
            num_ftrs = 2048
            self.fc_exists = False
        self.model.fc = nn.Linear(num_ftrs, model_cfg_data['num_class'])

        try:
            if model_cfg_data['UNFROZEN_BLOCKS'] > 0:
                self.freeze_layers(model_cfg_data)
        except:
            model_cfg_data['UNFROZEN_BLOCKS'] = 0
    
    def forward(self,x):
        x = self.model(x)
        if not self.fc_exists:
            x = self.model.fc(x)
        x = nn.functional.softmax(x,dim=1)
        return x

    def freeze_layers(self,model_cfg_data):
        a_modules = [i for i in dict(self.model.named_modules()) if "layer" in i and "." not in i]
        for i in range(len(a_modules) - model_cfg_data['UNFROZEN_BLOCKS'],len(a_modules)):
            for name,param in self.model.named_parameters():
                if a_modules[i] in name:
                    param.requires_grad = True