import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.models.detection as detec
import numpy as np  # NumPy를 임포트합니다

# 모델 불러오기
model = detec.maskrcnn_resnet50_fpn_v2(weights=detec.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
model.eval()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 이미지 로드 및 전처리
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    return img.to(device)

# 사람 세그먼테이션 함수
def segment_people(image_path):
    img = load_image(image_path)
    with torch.no_grad():
        prediction = model([img])  # 이미 device 설정됨

    img_np = np.array(T.ToPILImage()(img.cpu().squeeze(0)))
    
    person_masks = [prediction[0]['masks'][i, 0].cpu().numpy() for i, label in enumerate(prediction[0]['labels']) if label == 1]
    
    if person_masks:
        all_masks = np.stack(person_masks).sum(0) >= 1
        masked_image = Image.fromarray((img_np * all_masks[..., None]).astype('uint8'))
        masked_image.save('/home/kimyirum/Immersive-Experience-Design/images/human_test.png')  # 저장 경로 설정

    plt.imshow(img_np)
    if person_masks:
        plt.imshow(all_masks, alpha=0.5, cmap='gray')
    plt.axis('off')
    plt.show()

# 사람 한 명만 세그먼테이션 함수
def segment_single_person(image_path):
    img = load_image(image_path)
    with torch.no_grad():
        prediction = model([img])  # 이미 device 설정됨

    img_np = np.array(T.ToPILImage()(img.cpu().squeeze(0)))
    
    person_masks = [(prediction[0]['masks'][i, 0].cpu().numpy(), np.sum(prediction[0]['masks'][i, 0].cpu().numpy())) 
                    for i, label in enumerate(prediction[0]['labels']) if label == 1]
    
    if person_masks:
        # 가장 큰 마스크 선택
        largest_mask = max(person_masks, key=lambda x: x[1])[0]
        masked_image = Image.fromarray((img_np * largest_mask[..., None]).astype('uint8'))
        masked_image.save('/home/kimyirum/Immersive-Experience-Design/images/humansingle_test.png')  # 저장 경로 설정

    plt.imshow(img_np)
    if person_masks:
        plt.imshow(largest_mask, alpha=0.5, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 이미지 경로 설정 및 함수 실행
    name = 'human_0.png'
    image_path = '/home/kimyirum/Immersive-Experience-Design/images/'+name
    # segment_people(image_path)
    segment_single_person(image_path)
