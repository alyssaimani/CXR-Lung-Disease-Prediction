# import torch
import torch.nn as nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = 'deployed_asset/best_model_vitS16dino_unbal_checkpoint_epoch_6.pt'
# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)

# model = model.eval()

# for param in model.parameters():
#     param.requires_grad = False

class DinoVits16_Model(nn.Module):
    def __init__(self):
        super(DinoVits16_Model,self).__init__()

        self.avgPool = nn.AdaptiveAvgPool1d(768)
        self.fc = nn.Linear(768,4)
    def forward(self, x):
        x = self.avgPool(x)
        x = self.fc(x)
        return x



# model = torch.nn.Sequential(model)
# model = torch.nn.Sequential(model, torch.nn.AdaptiveAvgPool1d(768))
# model = torch.nn.Sequential(model, torch.nn.Linear(768,4))
# model = model.to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()