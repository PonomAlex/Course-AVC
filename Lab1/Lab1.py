import numpy as np
from PIL import Image
import math
import os


def load_image(image_path):
    img = Image.open(image_path)
    return np.array(img).astype(np.float32) / 255.0


def save_image(image_array, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(save_path)
    print(f"   Сохранено: {save_path}")


def separate_rgb_components(image):
    r_component = np.zeros_like(image)
    g_component = np.zeros_like(image)
    b_component = np.zeros_like(image)

    r_component[:, :, 0] = image[:, :, 0]
    g_component[:, :, 1] = image[:, :, 1]
    b_component[:, :, 2] = image[:, :, 2]

    return r_component, g_component, b_component


def rgb_to_hsi(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Яркостная компонента I
    i = (r + g + b) / 3.0

    # Насыщенность S
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1 - 3 * min_rgb / (r + g + b + 1e-10)
    s[r + g + b == 0] = 0

    # Тон H
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-10
    theta = np.arccos(numerator / denominator)

    h = np.zeros_like(r)
    h[b <= g] = theta[b <= g]
    h[b > g] = 2 * np.pi - theta[b > g]

    # Нормализуем H от 0 до 1
    h = h / (2 * np.pi)

    return h, s, i


def hsi_to_rgb(h, s, i):
    h = h * 2 * np.pi
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Сектор RG (0 <= H < 2π/3)
    mask_rg = (h >= 0) & (h < 2 * np.pi / 3)
    if np.any(mask_rg):
        b[mask_rg] = i[mask_rg] * (1 - s[mask_rg])
        r[mask_rg] = i[mask_rg] * (1 + s[mask_rg] * np.cos(h[mask_rg]) /
                                   np.cos(np.pi / 3 - h[mask_rg] + 1e-10))
        g[mask_rg] = 3 * i[mask_rg] - (r[mask_rg] + b[mask_rg])

    # Сектор GB (2π/3 <= H < 4π/3)
    mask_gb = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    if np.any(mask_gb):
        h_adj = h[mask_gb] - 2 * np.pi / 3
        r[mask_gb] = i[mask_gb] * (1 - s[mask_gb])
        g[mask_gb] = i[mask_gb] * (1 + s[mask_gb] * np.cos(h_adj) /
                                   np.cos(np.pi / 3 - h_adj + 1e-10))
        b[mask_gb] = 3 * i[mask_gb] - (r[mask_gb] + g[mask_gb])

    # Сектор BR (4π/3 <= H < 2π)
    mask_br = (h >= 4 * np.pi / 3) & (h < 2 * np.pi)
    if np.any(mask_br):
        h_adj = h[mask_br] - 4 * np.pi / 3
        g[mask_br] = i[mask_br] * (1 - s[mask_br])
        b[mask_br] = i[mask_br] * (1 + s[mask_br] * np.cos(h_adj) /
                                   np.cos(np.pi / 3 - h_adj + 1e-10))
        r[mask_br] = 3 * i[mask_br] - (g[mask_br] + b[mask_br])

    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    return r, g, b


def invert_lightness(image):
    h, s, i = rgb_to_hsi(image)
    i_inv = 1.0 - i
    r_inv, g_inv, b_inv = hsi_to_rgb(h, s, i_inv)
    return np.stack([r_inv, g_inv, b_inv], axis=2)


def bilinear_interpolation(image, x, y):
    h, w, c = image.shape

    x_int = math.floor(x)
    y_int = math.floor(y)

    x_frac = x - x_int
    y_frac = y - y_int

    x1 = max(0, min(x_int, w - 1))
    x2 = max(0, min(x_int + 1, w - 1))
    y1 = max(0, min(y_int, h - 1))
    y2 = max(0, min(y_int + 1, h - 1))

    result = np.zeros(c)
    for channel in range(c):
        q11 = image[y1, x1, channel]
        q12 = image[y2, x1, channel]
        q21 = image[y1, x2, channel]
        q22 = image[y2, x2, channel]

        if x2 == x1 and y2 == y1:
            result[channel] = q11
        elif x2 == x1:
            result[channel] = q11 * (1 - y_frac) + q12 * y_frac
        elif y2 == y1:
            result[channel] = q11 * (1 - x_frac) + q21 * x_frac
        else:
            result[channel] = (q11 * (1 - x_frac) * (1 - y_frac) +
                               q21 * x_frac * (1 - y_frac) +
                               q12 * (1 - x_frac) * y_frac +
                               q22 * x_frac * y_frac)

    return result


def zoom_image(image, factor):
    h, w, c = image.shape

    new_h = int(round(h * factor))
    new_w = int(round(w * factor))

    new_h = max(1, new_h)
    new_w = max(1, new_w)

    result = np.zeros((new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            src_x = j / factor
            src_y = i / factor

            src_x = max(0, min(src_x, w - 1e-10))
            src_y = max(0, min(src_y, h - 1e-10))

            result[i, j] = bilinear_interpolation(image, src_x, src_y)

    return result


def main():
    # Пути к файлам
    input_image_path = 'Photo/photo-lab1.png'
    output_dir = 'Lab1'

    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    print("Загрузка изображения...")
    original = load_image(input_image_path)
    print(f"Оригинал: {original.shape[1]}x{original.shape[0]}")

    # Сохраняем оригинал для демонстрации "до"
    save_image(original, os.path.join(output_dir, '00_original.png'))

    print("\n1. ЦВЕТОВЫЕ МОДЕЛИ")

    # 1.1 Выделить компоненты R, G, B
    print("\n1.1 Выделение RGB компонент:")
    r, g, b = separate_rgb_components(original)
    save_image(r, os.path.join(output_dir, '1-1_r.png'))
    save_image(g, os.path.join(output_dir, '1-1_g.png'))
    save_image(b, os.path.join(output_dir, '1-1_b.png'))

    # 1.2 Привести к HSI, сохранить яркостную компоненту
    print("\n1.2 Преобразование в HSI, сохранение яркостной компоненты:")
    h, s, i = rgb_to_hsi(original)
    i_component = np.stack([i, i, i], axis=2)
    save_image(i_component, os.path.join(output_dir, '1-2_i_component.png'))

    # 1.3 Инвертировать яркостную компоненту
    print("\n1.3 Инвертирование яркостной компоненты:")
    inverted = invert_lightness(original)
    save_image(inverted, os.path.join(output_dir, '1-3_inverted.png'))

    print("\n2. ПЕРЕДИСКРЕТИЗАЦИЯ")

    M = 2  # Растяжение
    N = 3  # Сжатие
    K = M / N  # Итоговый коэффициент

    print(f"\nПараметры: M={M}, N={N}, K={K:.3f}")

    # 2.1 Растяжение в M раз
    print(f"\n2.1 Растяжение в {M} раза:")
    stretched = zoom_image(original, M)
    save_image(stretched, os.path.join(output_dir, f'2-1_stretched_{M}x.png'))

    # 2.2 Сжатие в N раз
    print(f"\n2.2 Сжатие в {N} раза:")
    compressed = zoom_image(original, 1 / N)
    save_image(compressed, os.path.join(output_dir, f'2-2_compressed_{N}x.png'))

    # 2.3 Передискретизация в K раз (два прохода)
    print(f"\n2.3 Передискретизация в {K:.3f} раза (два прохода):")
    # Сначала растяжение, потом сжатие
    temp = zoom_image(original, M)
    two_pass = zoom_image(temp, 1 / N)
    save_image(two_pass, os.path.join(output_dir, f'2-3_resampled_two_pass_{K:.2f}x.png'))

    # 2.4 Передискретизация в K раз (один проход)
    print(f"\n2.4 Передискретизация в {K:.3f} раза (один проход):")
    one_pass = zoom_image(original, K)
    save_image(one_pass, os.path.join(output_dir, f'2-4_resampled_one_pass_{K:.2f}x.png'))

    # Вывод информации о размерах
    print("\n" + "=" * 50)
    print("РАЗМЕРЫ ИЗОБРАЖЕНИЙ:")
    print("=" * 50)
    print(f"Оригинал:           {original.shape[1]}x{original.shape[0]}")
    print(f"Растяжение ({M}x):     {stretched.shape[1]}x{stretched.shape[0]}")
    print(f"Сжатие (1/{N}):       {compressed.shape[1]}x{compressed.shape[0]}")
    print(f"Два прохода ({K:.2f}x): {two_pass.shape[1]}x{two_pass.shape[0]}")
    print(f"Один проход ({K:.2f}x): {one_pass.shape[1]}x{one_pass.shape[0]}")

    print("\n" + "=" * 50)
    print("СОХРАНЕННЫЕ ФАЙЛЫ:")
    print("=" * 50)
    print("00_original.png - оригинал (для демонстрации 'до')")
    print("\n1. Цветовые модели:")
    print("  1-1_r.png - красная компонента")
    print("  1-1_g.png - зеленая компонента")
    print("  1-1_b.png - синяя компонента")
    print("  1-2_i_component.png - яркостная компонента HSI")
    print("  1-3_inverted.png - инвертированная яркость")
    print("\n2. Передискретизация:")
    print(f"  2-1_stretched_{M}x.png - растяжение в {M} раза")
    print(f"  2-2_compressed_{N}x.png - сжатие в {N} раза")
    print(f"  2-3_resampled_two_pass_{K:.2f}x.png - передискретизация (два прохода)")
    print(f"  2-4_resampled_one_pass_{K:.2f}x.png - передискретизация (один проход)")


if __name__ == "__main__":
    main()