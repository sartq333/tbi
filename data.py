import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def find_mask_file(image_path, mask_dir, mask_extensions=['.png', '.jpg', '.jpeg']):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for ext in mask_extensions:
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            return mask_path
    return None

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = find_mask_file(img_path, self.mask_dir)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def transform_img():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform

if __name__ == "__main__":
    print("Dataset class")