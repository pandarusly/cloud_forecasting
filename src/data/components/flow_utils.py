import cv2
import numpy as np

import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

try:
    import pyflow
except ImportError as e:
    print(e)


def load_flow_to_numpy(path):
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.0)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


def load_flow_to_png(path):
    flow = load_flow_to_numpy(path)
    image = flow_to_image(flow)
    return image


def write_flo_file(filename, flow_data):
    with open(filename, "wb") as f:
        magic = np.array([202021.25], dtype=np.float32)
        size = np.array([flow_data.shape[1], flow_data.shape[0]], dtype=np.int32)
        flow_data = flow_data.astype(np.float32)

        # Write the magic number, size, and flow data to the file
        magic.tofile(f)
        size.tofile(f)
        flow_data.tofile(f)


def read_flo_file(filename):
    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception("Invalid .flo file")

        size = np.fromfile(f, np.int32, count=2)
        width, height = size[0], size[1]

        flow_data = np.fromfile(f, np.float32, count=2 * width * height)
        flow_data = flow_data.reshape((height, width, 2))

    return flow_data


def get_opticalflow(image_sequence,version="by_step"):
 
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
        # 更新上一帧
        if version =="by_step":
            prev_frame = curr_frame

    return FLOW_ARRAY


def get_optical_flow_pyflow(image_sequence,version="by_step"):
    """_summary_

    Args:
        image_sequence (_type_): list of image sequences which is not normalized.
    Returns:
        _type_: new uv_sequence which length is the length of the image sequence -1.
    pyr_scale: Any, 0.5
    levels: Any, 3
    winsize: Any,  15
    iterations: Any, 3,
    poly_n: Any, 5,
    poly_sigma: Any,1.2
    flags: int , 0
    """
    # optical flow option
    alpha = float(0.012)  # 0.012
    ratio = float(0.75)  # 0.75
    minWidth = int(20)  # 20
    nOuterFPIterations = int(7)  # 7
    nInnerFPIterations = int(1)  # 1
    nSORIterations = int(30)  # 30
    img_type = int(1) # gray

    num_frames = len(image_sequence)
    # 初始化光流 第一帧
    prev_frame = image_sequence[0]
    if prev_frame.shape[-1] == 3:
        prev_frame = cv2.cvtColor(image_sequence[0], cv2.COLOR_RGB2GRAY)[:, :, None]
    prev_frame = np.ascontiguousarray(prev_frame, dtype=np.float64) / 255.0
    FLOW_ARRAY = []
    # 循环计算稠密光流
    for i in range(1, num_frames):
        curr_frame = image_sequence[i]
        if curr_frame.shape[-1] == 3:
            curr_frame = cv2.cvtColor(image_sequence[i], cv2.COLOR_RGB2GRAY)[:, :, None]
        curr_frame = np.ascontiguousarray(curr_frame, dtype=np.float64) / 255.0
        u, v, _ = pyflow.coarse2fine_flow(
            prev_frame,
            curr_frame,
            alpha,
            ratio,
            minWidth,
            nOuterFPIterations,
            nInnerFPIterations,
            nSORIterations,
            img_type,
        )

        flow = np.concatenate((u[..., None], v[..., None]), axis=2) # h w 2

        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        FLOW_ARRAY.append(flow)
        # 更新上一帧
        if version=="by_step":
            prev_frame = curr_frame

    return FLOW_ARRAY




if __name__ == "__main__":
    # flo = load_flow_to_numpy("frame_0001.flo")
    # print(flo.shape)  # (436, 1024, 2)
    # image = load_flow_to_png("frame_0001.flo")
    # plt.imshow(image)
    # plt.show()
    from PIL import Image

    file_paths = [
    "data/validate_croped/10244OhbVrpoi_H8XX_AHIXX_L2_PRJ_20201208_1500_2000M_PRJ3_EVB1040.jpg",
    "data/validate_croped/10244OhbVrpoi_H8XX_AHIXX_L2_PRJ_20201208_1510_2000M_PRJ3_EVB1040.jpg"
    ]
    image_seq = list(map(lambda path: np.array(Image.open(path).convert("RGB")), file_paths))

    FLOW_ARRAY = get_optical_flow_pyflow(
        image_seq
    )
    for FLOW_ARRAY_ in FLOW_ARRAY:   
        print(FLOW_ARRAY_.shape)
