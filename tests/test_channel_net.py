from omegaconf import OmegaConf
import hydra
import torch

config_path = r"F:\study-note\python-note\AIproject\forecast\cloud_forecasting\configs\model\channel_net.yaml"

model_config = OmegaConf.load(config_path)

model = hydra.utils.instantiate(model_config)

# print(model)

batch = dict(
    static=[torch.randn(1, 10, 128, 128)],
    dynamic=[torch.randn(1, 10, 3, 128, 128), torch.randn(1, 10, 1, 128, 128)],
    dynamic_mask=torch.randint(0, 2, size=()),
)

(pred, _) = model(batch)

print(pred.shape)
