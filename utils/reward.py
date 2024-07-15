# unfolding task: 화면에 보이는 넓이 계산
# folding task

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torchvision.transforms import Compose
from segment_anything import SamPredictor, sam_model_registry
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 참고: https://github.com/LiheYoung/Depth-Anything
def get_depth_map(img):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
        
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    
    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    return depth

# 참고: https://github.com/Victorlouisdg/cloth-competition/blob/main/evaluation-service/backend/main.py
def segment(img):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)  #"cpu"/"gpu"

    predictor = SamPredictor(sam)

    img = cv2.imread("images/donut.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(img)
    input_point = np.array([[200, 150]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks

# 참고: https://github.com/Victorlouisdg/cloth-competition/blob/main/evaluation-service/backend/compute_area.py
def calculate_areas_for_image(depth_image_path, mask_image_path, fx, fy, cx, cy):
    depth_map = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    masked_depth_map = np.where(mask > 0, depth_map, 0)
    masked_values = masked_depth_map[mask > 0]
    mean = np.mean(masked_values)
    lower_bound = mean - 0.5
    upper_bound = mean + 0.5
    masked_depth_map = np.where(
        (masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0
    )
    x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))
    X1 = (x - 0.5 - cx) * masked_depth_map / fx
    Y1 = (y - 0.5 - cy) * masked_depth_map / fy
    X2 = (x + 0.5 - cx) * masked_depth_map / fx
    Y2 = (y + 0.5 - cy) * masked_depth_map / fy
    pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))
    pixel_areas_masked = np.where(mask > 0, pixel_areas, 0)
    # Normalize the pixel areas to the range 0-255
    pixel_areas_normalized = cv2.normalize(
        pixel_areas_masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return pixel_areas_normalized


def get_reward_folding(img):

    mask=segment(img)
    depth_map=get_depth_map(img)
    
    area=calculate_areas_for_image(depth_map, mask, fx, fy, cx, cy)
    
    return area