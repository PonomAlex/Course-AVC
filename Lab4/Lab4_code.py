import os
import numpy as np
import requests
from PIL import Image

origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
sample_data = requests.get(f"{origin}/api/samples/{sample_id}").json()
image_paths = [f"{origin}/images/{page['filename']}" for page in sample_data["pages"][:6]]

os.makedirs("src", exist_ok=True)
os.makedirs("results", exist_ok=True)

for url in image_paths:
    filename = url.split("/")[-1]
    r = requests.get(url)
    with open(f"src/{filename}", "wb") as f:
        f.write(r.content)

THRESHOLD = 50

KERNEL_GX = np.array([
    [0, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=np.float32)

KERNEL_GY = np.array([
    [0, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=np.float32)


def rgb_to_gray(rgb):
    weights = np.array([0.299, 0.587, 0.114])
    gray = np.tensordot(rgb.astype(np.float32), weights, axes=([2], [0]))
    return np.clip(gray, 0, 255).astype(np.uint8)


def convolve_3x3(gray, kernel):
    src = gray.astype(np.float32)
    h, w = src.shape
    padded = np.pad(src, 1, mode="edge")
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(3):
        for x in range(3):
            out += kernel[y, x] * padded[y:y + h, x:x + w]
    return out


def normalize_0_255(arr):
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


src_files = [f for f in os.listdir("src") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
src_files.sort()

for idx, filename in enumerate(src_files):
    src_path = os.path.join("src", filename)
    rgb = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
    gray = rgb_to_gray(rgb)

    gx = convolve_3x3(gray, KERNEL_GX)
    gy = convolve_3x3(gray, KERNEL_GY)
    g = np.abs(gx) + np.abs(gy)

    gx_n = normalize_0_255(gx)
    gy_n = normalize_0_255(gy)
    g_n = normalize_0_255(g)
    binary = np.where(g_n > THRESHOLD, 255, 0).astype(np.uint8)

    Image.fromarray(rgb).save(f"results/original_{idx:02d}.png")
    Image.fromarray(gray).save(f"results/grayscale_{idx:02d}.png")
    Image.fromarray(gx_n).save(f"results/gx_{idx:02d}.png")
    Image.fromarray(gy_n).save(f"results/gy_{idx:02d}.png")
    Image.fromarray(g_n).save(f"results/g_{idx:02d}.png")
    Image.fromarray(binary).save(f"results/binary_{idx:02d}_t{THRESHOLD}.png")