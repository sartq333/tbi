import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from data import transform_img 
from inference import load_model, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "unet_model.pth"
model = load_model(weights_path, device)

def process_image(image, text, font_size, text_color):
    image = image.convert("RGB")
    print(f"image: {image}")
    background_with_text = image.copy()
    draw = ImageDraw.Draw(background_with_text)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", font_size)
    text_position = (50, 50)
    # text_color = (0, 0, 0)
    text_color = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))
    draw.text(text_position, text, fill=text_color, font=font)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "unet_model.pth"
    model = load_model(weights_path, device)
    transform = transform_img()
    image_tensor = transform(image).unsqueeze(0) 
    mask = predict(model, image_tensor, device)
    mask = mask.squeeze(0)
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_binary, mode="L")
    mask_img = mask_img.resize(image.size, resample=Image.NEAREST)

    original_rgba = image.convert("RGBA")

    r, g, b, _ = original_rgba.split()
    subject_img = Image.merge("RGBA", (r, g, b, mask_img))

    background_with_text.paste(subject_img, (0, 0), subject_img)
    return background_with_text

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Enter Text"),
        gr.Slider(10, 70, value=5, step=5, label="Font Size"),
        gr.ColorPicker(value="#000000", label="Text Color")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Text Behind Image Generator",
    description="Upload an image, enter text, and choose font size to generate the output image."
)

interface.launch()