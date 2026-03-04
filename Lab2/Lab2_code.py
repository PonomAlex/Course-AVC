import requests
import os
from PIL import Image
import numpy as np

origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
sample_data = requests.get(f"{origin}/api/samples/{sample_id}").json()
image_paths = [f"{origin}/images/{page['filename']}" for page in sample_data["pages"][:6]]

os.makedirs("src", exist_ok=True)
for url in image_paths:
    filename = url.split("/")[-1]
    r = requests.get(url)
    with open(f"src/{filename}", "wb") as f:
        f.write(r.content)

def rgb_to_grayscale(input_path, output_path):
    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    gray = Image.new('L', (w, h))
    pixels = gray.load()
    for y in range(h):
        for x in range(w):
            r, g, b = img.getpixel((x, y))
            y_val = int(0.299 * r + 0.587 * g + 0.114 * b)
            pixels[x, y] = y_val
    gray.save(output_path)

def niblack_binarization_fast(input_path, output_path, window_size=5, k=-0.2):
    img = Image.open(input_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral_sq = np.zeros((h + 1, w + 1), dtype=np.float64)

    for i in range(1, h + 1):
        for j in range(1, w + 1):
            val = arr[i - 1, j - 1]
            integral[i, j] = val + integral[i - 1, j] + integral[i, j - 1] - integral[i - 1, j - 1]
            integral_sq[i, j] = val * val + integral_sq[i - 1, j] + integral_sq[i, j - 1] - integral_sq[i - 1, j - 1]

    result = np.zeros_like(arr, dtype=np.uint8)
    r = window_size // 2

    for y in range(h):
        for x in range(w):
            y1 = max(0, y - r)
            y2 = min(h - 1, y + r)
            x1 = max(0, x - r)
            x2 = min(w - 1, x + r)

            sum_val = (integral[y2 + 1, x2 + 1] - integral[y1, x2 + 1] -
                       integral[y2 + 1, x1] + integral[y1, x1])

            sum_sq = (integral_sq[y2 + 1, x2 + 1] - integral_sq[y1, x2 + 1] -
                      integral_sq[y2 + 1, x1] + integral_sq[y1, x1])

            n = (y2 - y1 + 1) * (x2 - x1 + 1)
            mean = sum_val / n
            var = (sum_sq / n) - (mean * mean)
            if var < 0:
                var = 0
            std = np.sqrt(var)

            T = mean + k * std
            result[y, x] = 255 if arr[y, x] > T else 0

    Image.fromarray(result).save(output_path)

os.makedirs('results', exist_ok=True)
for fname in os.listdir('src'):
    if fname.lower().endswith(('.png', '.bmp', '.jpeg', '.jpg')):
        gray_path = f'results/{fname}_gray.bmp'
        rgb_to_grayscale(f'src/{fname}', gray_path)
        bin_path = f'results/{fname}_niblack.bmp'
        niblack_binarization_fast(gray_path, bin_path, window_size=5, k=-0.2)