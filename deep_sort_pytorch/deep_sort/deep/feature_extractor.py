import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net  # 確保你的模型模組的相對導入路徑正確

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        """
        初始化特徵提取器。
        :param model_path: 模型權重檔案路徑
        :param use_cuda: 是否使用 GPU
        """
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        # 加載模型權重
        try:
            state_dict = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)['net_dict']
            self.net.load_state_dict(state_dict)
        except KeyError as e:
            raise KeyError(f"模型權重文件可能有問題：{e}")
        except Exception as e:
            raise RuntimeError(f"加載模型權重失敗：{e}")

        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)  # 調整影像大小
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        預處理影像：
        1. 將影像值縮放到 [0, 1]
        2. 調整大小到 (64, 128)
        3. 轉換為 PyTorch Tensor
        4. 正規化
        :param im_crops: 影像裁剪區域
        :return: 預處理後的影像 Batch
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        """
        對輸入的影像裁剪區域進行特徵提取。
        :param im_crops: 一個或多個影像裁剪區域
        :return: 提取的特徵向量
        """
        if not isinstance(im_crops, list):
            im_crops = [im_crops]

        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    # 測試用例
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]  # 讀取影像並轉換為 RGB 格式
    crop = img[50:178, 30:94]  # 假設影像裁剪區域
    extr = Extractor("checkpoint/ckpt.t7", use_cuda=True)
    try:
        feature = extr([crop])  # 支援多影像裁剪區域輸入
        print("特徵形狀：", feature.shape)
    except Exception as e:
        print(f"特徵提取失敗：{e}")
