import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import init_model
from mmengine import Config

def visualize_hrnet_features():
    config_file = 'configs/deeplabv3plus/hrnet_80k.py'
    checkpoint_file = 'work_dirs/deeplabv3_hrnet_80k.pth'
    img_file = 'data/cityscapes/leftImg8bit/train/bremen/bremen_000100_000019_leftImg8bit.png'
    cam_file = 'result/plot/cam_deeplabv3plus_hrnet.jpg'
    
    # 初始化模型
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint_file, device='cuda:0')
    model.eval()
    
    # 加载和预处理图像
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 512))
    
    # 转换为 tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    
    # 提取特征
    with torch.no_grad():
        features = model.backbone(img_tensor)
        
        # HRNet 输出是列表，取最后一个 stage 的特征
        if isinstance(features, list):
            # 取最高层特征（通常是最后一个元素或最高分辨率分支）
            feature_maps = features[-1][0]  # 取最后一个stage的第一个分支（最高分辨率）
        else:
            feature_maps = features
        
        # 对特征图进行平均并可视化
        if isinstance(feature_maps, list):
            feature_map = feature_maps[0].mean(dim=1).squeeze().cpu().numpy()
        else:
            feature_map = feature_maps.mean(dim=1).squeeze().cpu().numpy()
        
        # 归一化
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        feature_map = np.uint8(255 * feature_map)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # 叠加到原图
        superimposed_img = heatmap * 0.4 + img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # 保存结果
        cv2.imwrite(cam_file, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        print(f"Feature visualization saved to {cam_file}")

if __name__ == '__main__':
    visualize_hrnet_features()
