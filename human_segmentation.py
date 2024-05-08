import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.models.detection as detec

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

    img = T.ToPILImage()(img.cpu().squeeze(0))
    plt.imshow(img)
    
    person_masks = [prediction[0]['masks'][i, 0].cpu() for i, label in enumerate(prediction[0]['labels']) if label == 1]
    
    if person_masks:
        all_masks = torch.stack(person_masks).sum(0) >= 1
        plt.imshow(all_masks.cpu().numpy(), alpha=0.5, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 이미지 경로 설정 및 함수 실행
    image_path = '/home/kimyirum/Immersive-Experience-Design/human_test1.png'
    segment_people(image_path)
