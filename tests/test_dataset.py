import json
import os
import random
import re
from datetime import datetime, timedelta
import string
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
import cv2
from src.data.components.flow_utils import write_flo_file

random.seed(42)
# TODO:bug fixed:The issue seems to be with the Intel OpenMP runtime library (libiomp5md.dll) being loaded multiple times or conflicting with another OpenMP runtime library. This can result in performance degradation or incorrect behavior in your program.
# set KMP_DUPLICATE_LIB_OK=TRUE in Environment Variable Workaround
# TODO 学习pyflow的Cython打包流程: https://github.com/pathak22/pyflow


def save_image_with_pil(image_data, output_file_path):
    """
    Save an image using the Python Imaging Library (PIL).

    Args:
        image_data (numpy.ndarray): Numpy array containing the image data.
        output_file_path (str): Path to the output file where the image will be saved.
    """
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(image_data.astype(np.uint8))

    # Save the PIL image to the specified file path
    pil_image.save(output_file_path)

    print("Image saved to:", output_file_path)


def get_strdatetime_filename_mapping_and_list(directory_path):
    # Regular expression pattern to match YYYYMMDD and HHMM in the filename
    pattern = r"(\d{8})_(\d{4})"

    strdatetime_list = []
    # Create a dictionary to map datetime objects to filenames
    datetime_filename_mapping = {}
    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        # Use regular expression to find the date and time in the filename
        match = re.search(pattern, filename)
        if match:
            date = match.group(1)
            time = match.group(2)
            strdatetime_list.append(date + time)

        # Populate the dictionary
    for dt, filename in zip(strdatetime_list, os.listdir(directory_path)):
        datetime_filename_mapping[dt] = filename

    return strdatetime_list, datetime_filename_mapping


def generate_subsequent_intervals(
    start_index_for_datetime_list: int,
    strdatetime_list: List[str],
    datetime_filename_mapping: Dict[str, str],  # Use Dict instead of map
    Intinterval: int = 19,
):
    datetime_list = [datetime.strptime(dt, "%Y%m%d%H%M") for dt in strdatetime_list]

    subsequent_intervals = []
    if start_index_for_datetime_list + Intinterval >= len(strdatetime_list):
        return subsequent_intervals

    tinterval = timedelta(minutes=10)
    current_datetime = datetime_list[start_index_for_datetime_list]

    for _ in range(Intinterval):
        filename = datetime_filename_mapping.get(
            current_datetime.strftime("%Y%m%d%H%M")
        )
        if filename is None:
            break  # Stop if no corresponding filename
        subsequent_intervals.append(filename)
        current_datetime += tinterval

    return subsequent_intervals


def make_dataset_json(Intinterval=19, step=0):
    # Intinterval = 19
    # step = 0
    OVERLAP = Intinterval - step

    # Replace 'path_to_directory' with the actual path to your directory containing the pictures
    directory_path = r"E:\data\H8JPEG_valid"
    (
        strdatetime_list,
        datetime_filename_mapping,
    ) = get_strdatetime_filename_mapping_and_list(directory_path)

    print(f"A total of {len(strdatetime_list)} image!")
    datasets = []
    for i in tqdm(
        range(0, len(strdatetime_list), step + 1), desc="make dataset timestamps"
    ):
        subsequent_intervals = generate_subsequent_intervals(
            i, strdatetime_list, datetime_filename_mapping, Intinterval=Intinterval
        )
        if len(subsequent_intervals) == Intinterval:
            datasets.append(subsequent_intervals)

    # with open("data/dataset_no_ovelap.json", "w") as f:
    with open(f"data/dataset_step_{step}_interval_{Intinterval}.json", "w") as f:
        print(
            f"only {len(datasets)} time series is valid wihch length is {len(datasets[0])}"
        )
        f.write(json.dumps(datasets))


def get_opticalflow_cv2(image_sequence):
    # 生成示例数据，19帧[1024, 1024, 3]的随机图像数组
    # num_frames = 19
    # image_shape = (1024, 1024, 3)
    # image_sequence = np.random.randint(
    #     0, 256, size=(num_frames,) + image_shape, dtype=np.uint8
    # )
    num_frames = len(image_sequence)
    # 初始化光流 第一帧
    prev_frame = cv2.cvtColor(image_sequence[0], cv2.COLOR_RGB2GRAY)
    FLOW_ARRAY = []
    # 循环计算稠密光流
    for i in range(1, num_frames):
        curr_frame = cv2.cvtColor(image_sequence[i], cv2.COLOR_RGB2GRAY)

        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        FLOW_ARRAY.append(flow)

        # 在这里，你可以处理光流数据，例如可视化、保存结果等
        # 这里只是简单地打印出计算的光流向量
        # print("Frame:", i)
        # print("Flow shape:", flow.shape)
        # print("Sample flow vector at (500, 500):", flow[500, 500])

        # 更新上一帧
        prev_frame = curr_frame

    return FLOW_ARRAY


def generate_random_string(length):
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


class CustomImageDatasetPIL(Dataset):
    def __init__(
        self,
        data_dir,
        split_list="data\dataset_step_0_interval_19.json",
        use_transform=False,
        get_optical_flow=False,
    ):
        with open(split_list, "r") as json_file:
            data = json.load(json_file)

        self.data_dir = data_dir
        self.image_list = data
        self.use_transform = use_transform
        self.interval = 19
        self.input_frames_num = 6
        self.use_optical_flow = False
        if get_optical_flow:
            self.use_optical_flow = True
            self._get_optical_flow = get_opticalflow_cv2

        if use_transform:
            self.transform = iaa.Sequential(
                [
                    iaa.CropToFixedSize(width=1024, height=1024),  # 裁剪到1024
                    # 选择2到4种方法做变换
                    iaa.SomeOf(
                        (2, 4),
                        [
                            iaa.Fliplr(0.5),  # 对50%的图片进行水平镜像翻转
                            iaa.Flipud(0.5),  # 对50%的图片进行垂直镜像翻转
                            # iaa.OneOf(
                            #     [
                            #         iaa.Affine(rotate=(-10, 10), scale=(1.1, 1.2)),
                            #         iaa.Affine(
                            #             translate_px={
                            #                 "x": (-10, 10),
                            #                 "y": (-7, -13),
                            #             },
                            #             scale=(1.1),
                            #         ),
                            #     ]
                            # ),
                            # Blur each image with varying strength using
                            # gaussian blur,
                            # average/uniform blur,
                            # median blur .
                            # iaa.OneOf(
                            #     [
                            #         iaa.GaussianBlur((0.2, 0.7)),
                            #         iaa.AverageBlur(k=(1, 3)),
                            #         iaa.MedianBlur(k=(1, 3)),
                            #     ]
                            # ),
                            # Sharpen each image, overlay the result with the original
                            # iaa.Sharpen(alpha=1.0, lightness=(0.7, 1.3)),
                            # Same as sharpen, but for an embossing effect.
                            # iaa.Emboss(alpha=(0, 0.3), strength=(0, 1)),
                            # 添加高斯噪声
                            # iaa.AdditiveGaussianNoise(
                            #     loc=0, scale=(0.01 * 255, 0.05 * 255)
                            # ),
                            # iaa.Grayscale((0.2, 0.7)),
                            # # iaa.Invert(0.05, per_channel=True),  # invert color channels
                            # # Add a value of -10 to 10 to each pixel.
                            # iaa.Add((-10, 10), per_channel=0.5),
                            # iaa.OneOf(
                            #     [
                            #         iaa.AddElementwise((-20, 20)),
                            #         iaa.MultiplyElementwise((0.8, 1.2)),
                            #         iaa.Multiply((0.7, 1.3)),
                            #     ]
                            # ),
                            # Improve or worsen the contrast of images.
                            # iaa.ContrastNormalization((0.7, 1.3)),
                        ],
                        # do all of the above augmentations in random order
                        random_order=True,
                    ),
                ],
                random_order=True,
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx) -> torch.Tensor:  # t c h w
        image_paths = [
            os.path.join(self.data_dir, basename) for basename in self.image_list[idx]
        ]

        frames = []
        for image_path in image_paths:
            frames.append(np.array(Image.open(image_path).convert("RGB")))

        # Apply augmentation using albumentations
        # Time Series Data Augmentation for Deep Learning: A Survey
        # https://blog.csdn.net/weixin_47954807/article/details/114908771

        if self.use_transform:
            transform_deterministic = self.transform.to_deterministic()
            augmented_frames = [
                transform_deterministic.augment_image(frame) for frame in frames
            ]
            frames = augmented_frames

        filenames = [basename for basename in self.image_list[idx]]
        if self.use_optical_flow:
            frames = self._get_optical_flow(frames)
            filenames = [basename for basename in self.image_list[idx + 1]]

        # Convert frames list to a tensor
        frames = np.stack(frames)
        frames = torch.tensor(frames, dtype=torch.float32).permute(
            0, 3, 1, 2
        )  # t h w c -> t c h w

        data = dict(images=frames, filenames=filenames)
        return data

    def plot_frames(self, data: Dict, out_directory):
        tensors, filenames = data["images"], data["filenames"]

        # Create the output directory if it doesn't exist
        os.makedirs(out_directory, exist_ok=True)

        for i in range(tensors.shape[0]):
            tensor = tensors[i]
            filename = filenames[i]

            if self.use_optical_flow:
                image_np_u = tensor.numpy()[0, :, :]
                image_np_v = tensor.numpy()[1, :, :]

                # Plot the image
                plt.figure(figsize=(20, 20))
                plt.subplot(1, 2, 1)
                plt.imshow(image_np_u, cmap="hsv")
                plt.axis("off")  # Turn off axis labels and ticks
                plt.title("u")
                plt.subplot(1, 2, 2)
                plt.imshow(image_np_v, cmap="hsv")
                plt.axis("off")  # Turn off axis labels and ticks
                plt.title("u")

                # Save the plot to the output directory
                output_path = os.path.join(out_directory, f"{filename}")
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

            else:
                # Convert the tensor to a numpy array
                # chw to hwc
                image_np = (
                    tensor.permute(1, 2, 0).numpy() / 255.0
                    if tensor.max() > 1
                    else tensor.permute(1, 2, 0).numpy()
                )

                # Plot the image
                plt.figure(figsize=(20, 20))
                plt.imshow(image_np)
                plt.axis("off")  # Turn off axis labels and ticks
                plt.title(filename)

                # Save the plot to the output directory
                output_path = os.path.join(out_directory, f"{filename}")
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

    def make_dataset(
        self, data, out_directory, random_string
    ):  # Get current timestamp):
        tensors, filenames = data["images"], data["filenames"]

        # Create the output directory if it doesn't exist
        os.makedirs(out_directory, exist_ok=True)

        for i in range(tensors.shape[0]):
            tensor = tensors[i]
            filename = filenames[i]

            if self.use_optical_flow:
                flow_data = tensor.permute(1, 2, 0).numpy()
                # image_np_u = flow_data[:, :, 0]
                # image_np_v = flow_data[:, :, 1]

                base_name = os.path.basename(filename).split(".")[0]
                output_path = os.path.join(
                    out_directory, f"{random_string}_{base_name}.flo"
                )
                write_flo_file(filename=output_path, flow_data=flow_data)

            else:
                # Convert the tensor to a numpy array
                # chw to hwc
                # no need to normalize here
                assert tensor.max() > 1
                image_np = tensor.permute(1, 2, 0).numpy()
                base_name = os.path.basename(filename).split(".")[0]
                output_path = os.path.join(
                    out_directory, f"{random_string}_{base_name}.jpg"
                )
                save_image_with_pil(image_np, output_path)


#  opencv is slower than pil
class CustomImageDatasetCV2(CustomImageDatasetPIL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx) -> torch.Tensor:
        pass
            # frame = cv2.imread(image_path)  # Read the image in BGR format using OpenCV
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB


# ----------------------------------------------------------------

if __name__ == "__main__":
    step = 19
    Intinterval = 19
    split_path = f"data/dataset_step_{step}_interval_{Intinterval}.json"
    # make_dataset_json(Intinterval=Intinterval, step=step)

    dataset = CustomImageDatasetPIL(
        data_dir="E:\data\H8JPEG_valid",
        use_transform=True,
        get_optical_flow=False,
        split_list=split_path,
    )
    # dataset = CustomImageDatasetCV2(data_dir="E:\data\H8JPEG_valid", use_transform=True)

    for _ in tqdm(range(6), position=1, colour="red"):
        random_string = generate_random_string(8)
        for i in tqdm(
            range(len(dataset)), position=2, colour="blue", desc="dataset making!!!"
        ):
            data = dataset[i]

            # tensors, filenames = data["images"], data["filenames"]
            # dataset.plot_frames(data, out_directory="E:\data\H8_demo")
            def generate_random_string(length):
                return "".join(
                    random.choice(string.ascii_letters) for _ in range(length)
                )

            print(f"generate_random_string: {random_string}")
            dataset.make_dataset(
                data,
                out_directory="E:\data\H8_dataset_aug",
                random_string=random_string,
            )
