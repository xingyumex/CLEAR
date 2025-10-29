import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
from thop import profile

from Utils.data_loader import InferenceDataset
from Networks.LInet import LINet


class CLEAR:
    def __init__(self, input_dir="./1_Input", output_dir="./2_Output", checkpoint="./Checkpoints/LINet.pt", k=0.005):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.k = k
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.LINet = LINet().to(self.device)
        self._setup()

    def _setup(self):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
        self.LINet.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.LINet.eval()
        os.makedirs(self.output_dir, exist_ok=True)

    def border(self, img):
        box_blurred_image = cv2.blur(img, (5, 5))
        img = img + (img - box_blurred_image)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def exponential_transform(self, img):
        img = img.astype(np.float32)
        img = np.exp(self.k * img) - 1
        img = (img / np.max(img) * 255).astype(np.uint8)
        return img

    def forward(self, output_tensor):
        output_img = (output_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_img = np.clip(output_img, 0, 255).astype(np.float32)
        img_bor = self.border(output_img)
        enhanced_img = self.exponential_transform(img_bor)
        return enhanced_img

    def enhance_images(self):
        dataset = InferenceDataset(self.input_dir)
        start_time = time.time()

        for i in tqdm(range(len(dataset)), desc="Enhancing images"):
            input_image = dataset[i]
            input_tensor = input_image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.LINet(input_tensor)
                output_tensor = torch.clamp(output_tensor, 0, 1)

            final_image = self.forward(output_tensor)
            output_path = os.path.join(self.output_dir, dataset.image_files[i])
            cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

        total_time = time.time() - start_time
        print(f"Inference completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    enhancer = CLEAR()
    enhancer.enhance_images()
