import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path

GEORGIAN_LETTERS = [
    'ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ',
    'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ'
]

LETTER_NAMES = {
    'ა': 'ani', 'ბ': 'bani', 'გ': 'gani', 'დ': 'doni', 'ე': 'eni', 'ვ': 'vini', 'ზ': 'zeni',
    'თ': 'tani', 'ი': 'ini', 'კ': 'kani', 'ლ': 'lasi', 'მ': 'mani', 'ნ': 'nari', 'ო': 'oni',
    'პ': 'pari', 'ჟ': 'zhani', 'რ': 'rae', 'ს': 'sani', 'ტ': 'tari', 'უ': 'uni', 'ფ': 'phari',
    'ქ': 'kani', 'ღ': 'ghani', 'ყ': 'qari', 'შ': 'shini', 'ჩ': 'chini', 'ც': 'tsani',
    'ძ': 'dzili', 'წ': 'tsili', 'ჭ': 'chari', 'ხ': 'khani', 'ჯ': 'jani', 'ჰ': 'hae'
}

FONT_SIZE = 72

script_dir = Path(__file__).parent if '__file__' in dir() else Path.cwd()

FONT_PATHS = [
    script_dir / "georgian_data" / "NotoSansGeorgian-VariableFont_wdth,wght.ttf",
    script_dir / "georgian_data" / "static" / "NotoSansGeorgian-Regular.ttf",
]

FONT_PATH = None
for path in FONT_PATHS:
    if path and os.path.exists(str(path)):
        FONT_PATH = str(path)
        break

if FONT_PATH is None:
    print("ВНИМАНИЕ: Шрифт не найден")
    print(f"Искали в: {FONT_PATHS}")
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

print(f"Используется шрифт: {FONT_PATH}")

IMAGE_SIZE = (120, 120)
BG_COLOR = (255, 255, 255, 0)
TEXT_COLOR = (0, 0, 0, 255)

output_dir = Path("georgian_chars")
output_dir.mkdir(exist_ok=True)


def create_char_image(char, font_path, font_size, image_size, bg_color, text_color):
    img = Image.new('RGBA', image_size, bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Ошибка загрузки шрифта: {e}")
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size[0] - text_width) // 2 - bbox[0]
    y = (image_size[1] - text_height) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=text_color)
    img_array = np.array(img)
    alpha = img_array[:, :, 3]
    coords = np.argwhere(alpha > 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        img = img.crop((x_min, y_min, x_max, y_max))
    return img


char_images = {}
for char in GEORGIAN_LETTERS:
    img = create_char_image(char, FONT_PATH, FONT_SIZE, IMAGE_SIZE, BG_COLOR, TEXT_COLOR)
    img_path = output_dir / f"{ord(char):04X}_{LETTER_NAMES[char]}.png"
    img.save(img_path)
    char_images[char] = img
    print(f"Создан: {char} -> {img_path}")


def binarize_image(img):
    if img.mode == 'RGBA':
        arr = np.array(img)
        binary = ((arr[:, :, 0] < 50) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50) & (arr[:, :, 3] > 0)).astype(int)
    else:
        img_gray = img.convert('L')
        arr = np.array(img_gray)
        binary = (arr < 128).astype(int)
    return binary


def calculate_quarter_weights(binary_img):
    h, w = binary_img.shape
    mid_h = h // 2
    mid_w = w // 2
    q1 = binary_img[:mid_h, :mid_w].sum()
    q2 = binary_img[:mid_h, mid_w:].sum()
    q3 = binary_img[mid_h:, :mid_w].sum()
    q4 = binary_img[mid_h:, mid_w:].sum()
    return q1, q2, q3, q4


def calculate_specific_weights(binary_img):
    h, w = binary_img.shape
    mid_h = h // 2
    mid_w = w // 2
    area_q1 = max(1, mid_h * mid_w)
    area_q2 = max(1, mid_h * (w - mid_w))
    area_q3 = max(1, (h - mid_h) * mid_w)
    area_q4 = max(1, (h - mid_h) * (w - mid_w))
    q1, q2, q3, q4 = calculate_quarter_weights(binary_img)
    return q1 / area_q1, q2 / area_q2, q3 / area_q3, q4 / area_q4


def calculate_center_of_mass(binary_img):
    h, w = binary_img.shape
    total_weight = binary_img.sum()
    if total_weight == 0:
        return 0, 0
    y_coords, x_coords = np.where(binary_img == 1)
    return x_coords.sum() / total_weight, y_coords.sum() / total_weight


def calculate_normalized_center(binary_img):
    h, w = binary_img.shape
    cx, cy = calculate_center_of_mass(binary_img)
    return cx / w if w > 0 else 0, cy / h if h > 0 else 0


def calculate_moments_of_inertia(binary_img):
    h, w = binary_img.shape
    cx, cy = calculate_center_of_mass(binary_img)
    total_weight = binary_img.sum()
    if total_weight == 0:
        return 0, 0
    y_coords, x_coords = np.where(binary_img == 1)
    return np.sum((y_coords - cy) ** 2), np.sum((x_coords - cx) ** 2)


def calculate_normalized_moments(binary_img):
    h, w = binary_img.shape
    Ix, Iy = calculate_moments_of_inertia(binary_img)
    area = h * w
    return Ix / area if area > 0 else 0, Iy / area if area > 0 else 0


def calculate_profiles(binary_img):
    return binary_img.sum(axis=1).tolist(), binary_img.sum(axis=0).tolist()


def save_combined_profile(profile_x, profile_y, name, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(len(profile_x)), profile_x, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax1.fill_between(range(len(profile_x)), profile_x, alpha=0.3)
    ax1.set_xlabel('Y coordinate', fontsize=11)
    ax1.set_ylabel('Sum of black pixels', fontsize=11)
    ax1.set_title(f'Vertical Profile (X projection) - {name}', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.plot(range(len(profile_y)), profile_y, 'r-', linewidth=1.5, marker='s', markersize=3)
    ax2.fill_between(range(len(profile_y)), profile_y, alpha=0.3)
    ax2.set_xlabel('X coordinate', fontsize=11)
    ax2.set_ylabel('Sum of black pixels', fontsize=11)
    ax2.set_title(f'Horizontal Profile (Y projection) - {name}', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


profiles_dir = output_dir / "profiles"
profiles_dir.mkdir(exist_ok=True)

data = []

for char, img in char_images.items():
    binary = binarize_image(img)
    h, w = binary.shape

    q1, q2, q3, q4 = calculate_quarter_weights(binary)
    w1_rel, w2_rel, w3_rel, w4_rel = calculate_specific_weights(binary)
    cx, cy = calculate_center_of_mass(binary)
    cx_rel, cy_rel = calculate_normalized_center(binary)
    Ix, Iy = calculate_moments_of_inertia(binary)
    Ix_norm, Iy_norm = calculate_normalized_moments(binary)
    profile_x, profile_y = calculate_profiles(binary)

    char_code = f"{ord(char):04X}"
    char_name = LETTER_NAMES[char]

    save_combined_profile(profile_x, profile_y, char_name, profiles_dir / f"{char_code}_{char_name}_profiles.png")

    data.append({
        'char': char,
        'name': char_name,
        'unicode': f'U+{char_code}',
        'width': w,
        'height': h,
        'weight_q1': q1,
        'weight_q2': q2,
        'weight_q3': q3,
        'weight_q4': q4,
        'specific_weight_q1': round(w1_rel, 6),
        'specific_weight_q2': round(w2_rel, 6),
        'specific_weight_q3': round(w3_rel, 6),
        'specific_weight_q4': round(w4_rel, 6),
        'center_x': round(cx, 4),
        'center_y': round(cy, 4),
        'center_x_rel': round(cx_rel, 6),
        'center_y_rel': round(cy_rel, 6),
        'inertia_Ix': round(Ix, 2),
        'inertia_Iy': round(Iy, 2),
        'inertia_Ix_norm': round(Ix_norm, 4),
        'inertia_Iy_norm': round(Iy_norm, 4),
    })

df = pd.DataFrame(data)
csv_path = output_dir / "features.csv"
df.to_csv(csv_path, sep=';', index=False, encoding='utf-8')

print(f"\nСохранено {len(data)} символов")
print(f"CSV: {csv_path}")
print(f"Профили: {profiles_dir}")