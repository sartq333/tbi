import torch
import numpy as np
from PIL import Image
from unet import UNet
from data import transform_img 

def load_model(weights_path, device):
    model = UNet(in_channels=3, out_channels=1)  
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transform_img()
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0) 

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output) 
    return output.squeeze(0).cpu().numpy()
    
def save_output(mask, save_path):
    mask = (mask > 0.5).astype(np.uint8)*255 
    mask_image = Image.fromarray(mask[0])  
    mask_image.save(save_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "unet_model.pth"
    model = load_model(weights_path, device)
    image_tensor = preprocess_image("DUTS-TE-Image/ILSVRC2012_test_00000003.jpg")
    mask = predict(model, image_tensor, device)
    save_output(mask, "predicted_mask.jpg")
