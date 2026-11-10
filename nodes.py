import torch
import torch.nn.functional as F
import math
import random
import numpy as np

class PanoramaRandomCrop:
    """
    Нода для случайного кадрирования 360-панорамы (equirectangular) 
    в плоское перспективное изображение (rectilinear) с рандомизацией 
    точки взгляда (yaw, pitch) и угла обзора (FOV).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "output_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "min_fov": ("FLOAT", {"default": 45.0, "min": 10.0, "max": 120.0, "step": 1.0, "tooltip": "Минимальный угол обзора (широкоугольный)"}),
                "max_fov": ("FLOAT", {"default": 90.0, "min": 10.0, "max": 120.0, "step": 1.0, "tooltip": "Максимальный угол обзора (телеобъектив)"}),
                "min_pitch": ("FLOAT", {"default": -30.0, "min": -90.0, "max": 90.0, "step": 1.0, "tooltip": "Минимальный наклон камеры вверх/вниз"}),
                "max_pitch": ("FLOAT", {"default": 30.0, "min": -90.0, "max": 90.0, "step": 1.0, "tooltip": "Максимальный наклон камеры вверх/вниз"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Сид для воспроизводимости рандома"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_panorama"
    CATEGORY = "Zveroboy/Panorama"

    def crop_panorama(self, panorama, output_width, output_height, min_fov, max_fov, min_pitch, max_pitch, seed):
        # Приводим seed к диапазону numpy (0 - 2**32-1)
        seed_np = seed % (2**32)
        
        # Фиксация сида для воспроизводимости
        torch.manual_seed(seed)
        np.random.seed(seed_np)
        random.seed(seed)

        B, H_e, W_e, C = panorama.shape
        device = panorama.device
        dtype = panorama.dtype

        # Перевод в формат (B, C, H, W) для PyTorch
        pano_tensor = panorama.permute(0, 3, 1, 2)
        out_images = []

        for b in range(B):
            # Рандомизация параметров взгляда
            fov = random.uniform(min_fov, max_fov)
            yaw = random.uniform(-180.0, 180.0)
            pitch = random.uniform(min_pitch, max_pitch)

            # Перевод в радианы
            fov_rad = math.radians(fov)
            yaw_rad = math.radians(yaw)
            pitch_rad = math.radians(pitch)

            # Расчет фокусного расстояния для перспективной проекции
            f = (output_width / 2.0) / math.tan(fov_rad / 2.0)

            # Создание сетки пикселей выходного изображения
            y_coords = torch.arange(output_height, device=device, dtype=dtype)
            x_coords = torch.arange(output_width, device=device, dtype=dtype)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Координаты в пространстве камеры (Pinhole model)
            x_c = x_grid - (output_width / 2.0)
            y_c = (output_height / 2.0) - y_grid
            z_c = torch.full_like(x_c, f)

            # Формирование векторов лучей и нормализация
            rays = torch.stack([x_c, y_c, z_c], dim=-1)
            rays = rays / torch.norm(rays, dim=-1, keepdim=True)

            # Матрицы поворота (Yaw вокруг Y, Pitch вокруг X)
            cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
            Rx = torch.tensor([
                [1, 0, 0],
                [0, cos_p, -sin_p],
                [0, sin_p, cos_p]
            ], device=device, dtype=dtype)

            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
            Ry = torch.tensor([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ], device=device, dtype=dtype)

            # Итоговая матрица поворота
            R = torch.matmul(Ry, Rx)

            # Применение поворота к лучам
            rays_flat = rays.reshape(-1, 3)
            world_rays = torch.matmul(rays_flat, R.T)
            world_rays = world_rays.reshape(output_height, output_width, 3)

            # Перевод мировых координат в сферические
            X = world_rays[..., 0]
            Y = world_rays[..., 1]
            Z = world_rays[..., 2]

            theta = torch.atan2(X, Z) # Азимут (Yaw)
            phi = torch.asin(torch.clamp(Y, -1.0, 1.0)) # Угол места (Pitch)

            # Маппинг на UV-координаты эквидистантной панорамы [0, 1]
            u = (theta / (2 * math.pi)) + 0.5
            u = u % 1.0  # Корректный wrap-around по горизонтали
            v = 0.5 - (phi / math.pi)

            # Перевод в диапазон grid_sample [-1, 1]
            grid_u = 2 * u - 1
            grid_v = 2 * v - 1
            grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)

            # Сэмплирование с панорамы
            pano_b = pano_tensor[b:b+1]
            sampled = F.grid_sample(pano_b, grid, mode='bilinear', padding_mode='border', align_corners=True)
            out_images.append(sampled)

        # Сборка батча и возврат в формат ComfyUI (B, H, W, C)
        out_tensor = torch.cat(out_images, dim=0).permute(0, 2, 3, 1)
        
        return (out_tensor,)
