from pathlib import Path
import numpy as np
import requests
from PIL import Image
import os

origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

os.makedirs("src", exist_ok=True)
os.makedirs("results", exist_ok=True)

sample_data = requests.get(f"{origin}/api/samples/{sample_id}").json()
image_paths = [f"{origin}/images/{page['filename']}" for page in sample_data["pages"][:6]]

for url in image_paths:
    filename = url.split("/")[-1]
    r = requests.get(url)
    with open(f"src/{filename}", "wb") as f:
        f.write(r.content)


def load_grayscale(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.uint8)


def erode_ring_3x3(img):
    h, w = img.shape
    result = np.zeros_like(img)
    padded = np.pad(img, 1, mode='edge')
    kernel = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    kernel = np.array(kernel)
    coords = np.argwhere(kernel == 1)
    for i in range(h):
        for j in range(w):
            values = []
            for dy, dx in coords:
                values.append(padded[i + dy, j + dx])
            result[i, j] = np.min(values)
    return result


def save_image(arr, path):
    Image.fromarray(arr).save(path)


image_files = sorted([f for f in os.listdir("src") if f.endswith(('.png', '.jpg', '.jpeg'))])

for img_file in image_files:
    src_path = os.path.join("src", img_file)
    name = os.path.splitext(img_file)[0]

    gray = load_grayscale(src_path)
    filtered = erode_ring_3x3(gray)
    diff = np.abs(gray.astype(np.int16) - filtered.astype(np.int16)).astype(np.uint8)
    diff_contrasted = np.clip(diff * 10, 0, 255).astype(np.uint8)

    save_image(gray, f"results/{name}_grayscale.png")
    save_image(filtered, f"results/{name}_filtered.png")
    save_image(diff, f"results/{name}_diff.png")
    save_image(diff_contrasted, f"results/{name}_diff_contrasted.png")