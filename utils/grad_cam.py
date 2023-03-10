import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import torch


def viewGradCam(model_path: str, cat: int, img_path: str, device : int = 0):
    device = torch.device(device)
    model = torch.load(model_path, map_location=device)['model']
    model.to(device)
    model.eval()
    img = np.array(Image.open(img_path))
    img = cv2.resize(img, (512, 512))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    input_tensor = input_tensor.to(device)
    pred = model(input_tensor)
    print('Salida del modelo: ', int(torch.argmax(pred, dim=1)[0]))


    targets = [ClassifierOutputTarget(cat)]
    #target_layers = [model.attb.cab_]
    target_layers = [model.features]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img), cam , cam_image))
    Image.fromarray(images).show()
