from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from inference import load_model, preprocess_image, predict 

original_img = Image.open("DUTS-TR-Image/ILSVRC2012_test_00000645.jpg").convert("RGB")

background_with_text = original_img.copy()
draw = ImageDraw.Draw(background_with_text)
font_size = 50
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", font_size)
text = "Hello, world!"
text_position = (50, 50)
text_color = (255, 255, 255)
draw.text(text_position, text, fill=text_color, font=font)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "unet_model.pth"
model = load_model(weights_path, device)
image_tensor = preprocess_image("DUTS-TR-Image/ILSVRC2012_test_00000645.jpg")
mask = predict(model, image_tensor, device)

print(mask.shape)

mask = mask.squeeze(0)
mask_binary = (mask > 0.5).astype(np.uint8) * 255
mask_img = Image.fromarray(mask_binary, mode="L")
mask_img = mask_img.resize(original_img.size, resample=Image.NEAREST)

original_rgba = original_img.convert("RGBA")

r, g, b, _ = original_rgba.split()
subject_img = Image.merge("RGBA", (r, g, b, mask_img))

background_with_text.paste(subject_img, (0, 0), subject_img)
background_with_text.save("final_output.png")