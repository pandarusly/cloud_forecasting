import numpy as np

import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    flo = load_flow_to_numpy("frame_0001.flo")
    print(flo.shape)  # (436, 1024, 2)
    image = load_flow_to_png("frame_0001.flo")
    plt.imshow(image)
    plt.show()
