# from omegaconf import OmegaConf
# import hydra
# import torch

# config_path = r"F:\study-note\python-note\AIproject\forecast\cloud_forecasting\configs\model\channel_net.yaml"

# model_config = OmegaConf.load(config_path)

# model = hydra.utils.instantiate(model_config)

# # print(model)

# batch = dict(
#     static=[torch.randn(1, 10, 128, 128)],
#     dynamic=[torch.randn(1, 10, 3, 128, 128), torch.randn(1, 10, 1, 128, 128)],
#     dynamic_mask=torch.randint(0, 2, size=()),
# )

# (pred, _) = model(batch)

# print(pred.shape)

# ---  data analysis------------------------
# def find_discontinuous_points(datetime_list):
#     discontinuous_points = []

#     for i in range(1, len(datetime_list)):
#         prev_datetime = datetime.strptime(datetime_list[i - 1], "%Y%m%d%H%M")
#         curr_datetime = datetime.strptime(datetime_list[i], "%Y%m%d%H%M")

#         expected_datetime = prev_datetime + timedelta(minutes=10)

#         if curr_datetime != expected_datetime:
#             discontinuous_points.append(curr_datetime)

#     return discontinuous_points
