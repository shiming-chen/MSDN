import torch
from MSDN import MSDN
from dataset import UNIDataloader
import argparse
import json
from utils import evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/test_SUN.json')
config = parser.parse_args()
with open(config.config, 'r') as f:
    config.__dict__ = json.load(f)

dataloader = UNIDataloader(config)

model_gzsl = MSDN(config, normalize_V=False, normalize_F=True, is_conservative=True,
                  uniform_att_1=False, uniform_att_2=True,
                  is_conv=False, is_bias=True, non_linear_act=False).to(config.device)
model_dict = model_gzsl.state_dict()
saved_dict = torch.load('saved_model/SUN_MSDN_GZSL.pth')
saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
model_dict.update(saved_dict)
model_gzsl.load_state_dict(model_dict)

model_czsl = MSDN(config, normalize_V=False, normalize_F=True, is_conservative=True,
                  uniform_att_1=False, uniform_att_2=True,
                  is_conv=False, is_bias=True, non_linear_act=False).to(config.device)
model_dict = model_czsl.state_dict()
saved_dict = torch.load('saved_model/SUN_MSDN_CZSL.pth')
saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
model_dict.update(saved_dict)
model_czsl.load_state_dict(model_dict)

evaluation(config.batch_size, config.device,
           dataloader, model_gzsl, model_czsl)
