from PIL import Image
import numpy as np
import os

# Đảm bảo thư mục lưu trữ tồn tại
output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hàm thêm nhiễu Gaussian
def add_gaussian_noise(image, mean=0, sigma=10):
    img_array = np.array(image)
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return Image.fromarray(noisy_img.astype(np.uint8))

# Hàm thêm nhiễu Salt & Pepper
def add_salt_pepper_noise(image, density):
    img_array = np.array(image)
    noisy_img = img_array.copy()
    num_pixels = int(density * img_array.size)
    coords = [np.random.randint(0, i, num_pixels) for i in img_array.shape]
    for i in range(num_pixels):
        if np.random.random() < 0.5:
            noisy_img[coords[0][i], coords[1][i]] = 0
        else:
            noisy_img[coords[0][i], coords[1][i]] = 255
    return Image.fromarray(noisy_img.astype(np.uint8))

# Thư mục chứa ảnh gốc
input_dir = "input_images"
densities = [0.2441, 0.1221, 0.061, 0.0122]  # 24.41%, 12.21%, 6.1%, 1.22%

# Duyệt qua tất cả các file trong thư mục input_images
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Chỉ xử lý các file ảnh
        # Đọc ảnh gốc
        image_path = os.path.join(input_dir, filename)
        original_image = Image.open(image_path)

        # Tạo tên file không có phần mở rộng
        base_name = os.path.splitext(filename)[0]

        # Thêm nhiễu kết hợp: Gaussian + Salt & Pepper
        for density in densities:
            # Bước 1: Thêm nhiễu Gaussian
            gaussian_noisy_image = add_gaussian_noise(original_image, mean=0, sigma=10)
            
            # Bước 2: Thêm nhiễu Salt & Pepper lên ảnh đã có nhiễu Gaussian
            combined_noisy_image = add_salt_pepper_noise(gaussian_noisy_image, density)
            
            # Lưu ảnh kết hợp
            combined_noisy_image.save(os.path.join(output_dir, f"{base_name}_gaussian_sp_noise_{density*100:.2f}.jpg"))