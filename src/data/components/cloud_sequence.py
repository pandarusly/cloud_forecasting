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
from src.data.components.flow_utils import get_optical_flow_pyflow, get_opticalflow_cv2, write_flo_file


def get_time(parameter='total'):
    def deco_func(f):
        def wrapper(*arg, **kwarg):
            s_time = time.time()
            res = f(*arg, **kwarg)
            e_time = time.time()
            print('{} time use {}s'.format(parameter, e_time - s_time))
            return res
        return wrapper
    return deco_func


random.seed(42)
# TODO:fixed bug fixed:The issue seems to be with the Intel OpenMP runtime library (libiomp5md.dll) being loaded multiple times or conflicting with another OpenMP runtime library. This can result in performance degradation or incorrect behavior in your program.
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

    # print("Image saved to:", output_file_path)


def get_strdatetime_filename_mapping_and_list(directory_path):
    # Regular expression pattern to match YYYYMMDD and HHMM in the filename
    pattern = r"(\d{8})_(\d{4})"

    strdatetime_list = []
    datatimes_nums = set()
    # Create a dictionary to map datetime objects to filenames
    datetime_filename_mapping = {}
    croped_region_nums = set()
    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        # Use regular expression to find the date and time in the filename
        match = re.search(pattern, filename)
        croped_preffix = os.path.basename(filename).split('_')[0]  # 8
        croped_region_nums.add(croped_preffix)
        if match:
            date = match.group(1)
            time = match.group(2)
            strdatetime_list.append(croped_preffix+"_"+date + time)
            datatimes_nums.add(date + time)
            datetime_filename_mapping[croped_preffix +
                                      "_"+date + time] = filename
    croped_region_nums = len(list(croped_region_nums))
    datatimes_nums = len(list(datatimes_nums))

    return sorted(strdatetime_list), datetime_filename_mapping, croped_region_nums, datatimes_nums


def generate_subsequent_intervals(
    start_index_for_datetime_list: int,
    strdatetime_list: List[str],
    datetime_filename_mapping: Dict[str, str],  # Use Dict instead of map
    Intinterval: int = 19,
):
    subsequent_intervals = []
    if start_index_for_datetime_list + Intinterval >= len(strdatetime_list):
        return subsequent_intervals

    tinterval = timedelta(minutes=10)

    current_strdatetime = strdatetime_list[start_index_for_datetime_list]
    current_datetime = datetime.strptime(
        current_strdatetime.split("_")[-1], "%Y%m%d%H%M")
    croped_preffix = current_strdatetime.split("_")[0]
    for _ in range(Intinterval):
        filename = datetime_filename_mapping.get(
            croped_preffix+"_"+current_datetime.strftime("%Y%m%d%H%M")
        )
        if filename is None:
            break  # Stop if no corresponding filename
            # return subsequent_intervals
        subsequent_intervals.append(filename)
        current_datetime += tinterval

    return subsequent_intervals


def make_dataset_json(Intinterval, step, directory_path=r"data/data_cloud/H8JPEG_valid", split_path=None):
    # Intinterval = 19
    # step = 0
    # OVERLAP = Intinterval - step

    # Replace 'path_to_directory' with the actual path to your directory containing the pictures
    (
        strdatetime_list,
        datetime_filename_mapping,
        croped_region_nums,
        datatimes_nums
    ) = get_strdatetime_filename_mapping_and_list(directory_path)

    print(
        f"A total of {len(strdatetime_list)} image, and has {croped_region_nums} regions; {datatimes_nums} datatimes")
    datasets = []

    for i in range(0, croped_region_nums):
        time_seris_num = 0
        for j in range(0, datatimes_nums, step):

            subsequent_intervals = generate_subsequent_intervals(
                j, strdatetime_list, datetime_filename_mapping, Intinterval=Intinterval
            )
            if len(subsequent_intervals) == Intinterval:
                datasets.append(subsequent_intervals)

            time_seris_num += 1
        # len(datatimes_nums)
        subsequent_intervals = generate_subsequent_intervals(
            j+datatimes_nums, strdatetime_list, datetime_filename_mapping, Intinterval=Intinterval
        )
        if len(subsequent_intervals) == Intinterval:
            datasets.append(subsequent_intervals)

    json_path = f"data/dataset_step_{step}_interval_{Intinterval}.json" if split_path is None else split_path

    print(
        f"only {time_seris_num} time series is valid wihch has {croped_region_nums} regions and length is {len(datasets[0])} "
    )
    with open(json_path, "w") as f:
        f.write(json.dumps(datasets))
    return json_path


def generate_random_string(length):
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


# TODO:fixed tar -czvf x.tar.gz  x --- 使用了gzip进行了有损压缩。会影响JPG的质量。
class BaseCloudRGBSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split_path=None,
        use_transform=False,
        get_optical_flow=False,
        Intinterval=19,
        step=1,
        input_frames_num=6,
        crop_size=(1024, 1024)
    ):
        if not os.path.exists(data_dir):
            raise FileExistsError(f"data_dir {data_dir} is not exist")
        self.data_dir = data_dir
        self.use_transform = use_transform
        self.interval = Intinterval
        self.input_frames_num = input_frames_num
        self.use_optical_flow = False
        self.split_path = split_path
        self.crop_size = crop_size

        if not os.path.exists(split_path):
            self._make_dataset_json(
                Intinterval=Intinterval, step=step
            )

        with open(self.split_path, "r") as json_file:
            data = json.load(json_file)

        self.image_list = sorted(data)

        if get_optical_flow == "opencv_flow":
            self.use_optical_flow = True
            self._get_optical_flow = get_opticalflow_cv2
        elif get_optical_flow == "pyflow":
            self.use_optical_flow = True
            self._get_optical_flow = get_optical_flow_pyflow

        if use_transform:
            self.transform = iaa.Sequential(
                [
                    iaa.CropToFixedSize(
                        width=self.crop_size[0], height=self.crop_size[1]),  # 裁剪到1024
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

    @get_time("__getitem__")
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

    @get_time("plot_frames")
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

    def _make_dataset_json(self, Intinterval, step):
        split_path = f"data/dataset_step_{step}_interval_{Intinterval}.json"
        directory_path = self.data_dir
        self.split_path = make_dataset_json(
            Intinterval, step, directory_path, split_path)

    @get_time("make_dataset")
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
#  TODO:
class CloudFlowSequenceDataset(BaseCloudRGBSequenceDataset):

    def __init__(self, data_dir, split_path=None, use_transform=False, Intinterval=19, step=1, input_frames_num=6, crop_size=(1024, 1024)):
        super().__init__(data_dir, split_path, use_transform, Intinterval,
                         step, input_frames_num, crop_size, get_optical_flow=False, )

    def __getitem__(self, idx) -> torch.Tensor:
        pass
        # frame = cv2.imread(image_path)  # Read the image in BGR format using OpenCV
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

#  TODO:


class ImageCloudSequenceDataset(BaseCloudRGBSequenceDataset):

    def __init__(self, data_dir, split_path=None, use_transform=False, Intinterval=19, step=1, input_frames_num=6, crop_size=(1024, 1024)):
        super().__init__(data_dir, split_path, use_transform,
                         Intinterval, step, input_frames_num, crop_size)

    def __getitem__(self, idx) -> torch.Tensor:
        pass
        # frame = cv2.imread(image_path)  # Read the image in BGR format using OpenCV
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB


def process_dataset_maker():
    step = 19
    Intinterval = 19
    crop_size = (256, 256)
    crop_times = 6
    split_path = f"data/dataset_step_{step}_interval_{Intinterval}.json"
    data_dir = "data/data_cloud/H8JPEG_valid"
    PLot = False
    flow_type = "pyflow"  # opencv_flow pyflow none
    augment_save_dir = "data/data_cloud/H8JPEG_valid_aug_Intinterval-{}_flow_type-{}_flo".format(
        Intinterval, flow_type)

    # data_dir = augment_save_dir
    # augment_save_dir = f"data/data_cloud/H8JPEG_valid_aug_{crop_size[0]}"

    dataset = BaseCloudRGBSequenceDataset(
        data_dir=data_dir,
        use_transform=True,
        get_optical_flow=flow_type,
        split_path=split_path,
        step=step,
        Intinterval=Intinterval,
        crop_size=crop_size
    )

    for _ in tqdm(range(crop_times), position=1, colour="red"):
        random_string = generate_random_string(8)
        for i in tqdm(
            range(len(dataset)), position=2, colour="blue", desc="dataset making!!!"
        ):
            data = dataset[i]

            if PLot:
                dataset.plot_frames(data, out_directory=augment_save_dir)
            else:
                dataset.make_dataset(
                    data,
                    out_directory=augment_save_dir,
                    random_string=random_string,
                )

            if i > 1:  # only test croped image at six loops.
                break


# ----------------------------------------------------------------
# ls -l data/data_cloud/H8JPEG_valid_aug_256/*.jpg | wc -l
if __name__ == "__main__":

    process_dataset_maker()
