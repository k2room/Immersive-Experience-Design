from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# 격자 형태로 점묘화 생성
def create_pointillism_effect(image_path, dot_size=5, spacing=5):
    # 이미지를 로드하고 RGB로 변환
    image = Image.open(image_path).convert('RGB')
    
    # 출력 이미지 준비
    output_image = Image.new('RGB', image.size, color='white')
    draw = ImageDraw.Draw(output_image)

    # 이미지 크기에 따라 각 픽셀을 점으로 변환
    width, height = image.size
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            # 픽셀 색상값 추출
            r, g, b = image.getpixel((x, y))
            
            # 원 그리기 (픽셀의 색상으로 채움)
            draw.ellipse((x - dot_size // 2, y - dot_size // 2, x + dot_size // 2, y + dot_size // 2), fill=(r, g, b))

    return output_image

# 완전 랜덤하게 샘플링 후 점묘화 생성
def create_random_pointillism_effect_1(image_path, dot_count=1000, dot_size=5):
    # 이미지를 로드하고 RGB로 변환
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # 출력 이미지 준비
    output_image = Image.new('RGB', image.size, color='white')
    draw = ImageDraw.Draw(output_image)

    # 지정된 수의 점을 무작위 위치에 그리기
    for _ in range(dot_count):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = image.getpixel((x, y))
        
        # 원 그리기 (픽셀의 색상으로 채움)
        draw.ellipse((x - dot_size // 2, y - dot_size // 2, x + dot_size // 2, y + dot_size // 2), fill=(r, g, b))

    return output_image

def create_grid(width, height, cell_size):
    grid = {}
    for i in range(0, width, cell_size):
        for j in range(0, height, cell_size):
            grid[(i // cell_size, j // cell_size)] = []
    return grid

def get_grid_cell(x, y, cell_size):
    return (x // cell_size, y // cell_size)

def is_far_enough(new_point, grid, cell_size, min_distance):
    x_new, y_new = new_point
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0), (0,  0), (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]
    cell_x, cell_y = get_grid_cell(x_new, y_new, cell_size)
    for dx, dy in neighbors:
        nx, ny = cell_x + dx, cell_y + dy
        if (nx, ny) in grid:
            for point in grid[(nx, ny)]:
                x_old, y_old = point
                if math.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2) < min_distance:
                    return False
    return True

def create_random_pointillism_effect(image_path, dot_count=1000, dot_size=5, min_distance=10, max_attempts=100):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    output_image = Image.new('RGB', image.size, color='white')
    draw = ImageDraw.Draw(output_image)

    # 그리드 초기화
    cell_size = min_distance
    grid = create_grid(width, height, cell_size)

    points_added = 0
    for _ in range(dot_count):
        for attempt in range(max_attempts):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if is_far_enough((x, y), grid, cell_size, min_distance):
                r, g, b = image.getpixel((x, y))
                draw.ellipse((x - dot_size // 2, y - dot_size // 2, x + dot_size // 2, y + dot_size // 2), fill=(r, g, b))
                cell_x, cell_y = get_grid_cell(x, y, cell_size)
                grid[(cell_x, cell_y)].append((x, y))
                points_added += 1
                break
        else:
            # 최대 시도 횟수 내에 적절한 위치를 찾지 못한 경우
            print(f"Stopped after adding {points_added} dots.")
            break

    return output_image

if __name__ == "__main__":
    # 이미지 경로 지정 및 함수 실행
    image_path = '/home/kimyirum/Immersive-Experience-Design/images/background_0.jpg'
    output_image = create_pointillism_effect(image_path, dot_size=2, spacing=3)
    # output_image = create_random_pointillism_effect(image_path, dot_count=30000, dot_size=2, min_distance=3)


    # 결과 이미지 출력
    plt.imshow(output_image)
    plt.axis('off')  # 축 없애기
    plt.show()
