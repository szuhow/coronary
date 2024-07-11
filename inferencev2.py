import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import resize

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            self.double_conv(in_channels, 64),
            nn.MaxPool2d(2),
            self.double_conv(64, 128),
            nn.MaxPool2d(2)
        )
        
        self.bottleneck = self.double_conv(128, 256)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.double_conv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.double_conv(128, 64)
        )
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc_out = self.encoder(x)
        bottleneck_out = self.bottleneck(enc_out)
        dec_out = self.decoder(bottleneck_out)
        out = self.final_conv(dec_out)
        return self.sigmoid(out)

def adjust_predicted_mask_intensity(predicted_mask, num_labels=27):
    """
    Adjust the intensity of the predicted mask to match the number of labels.
    The intensity scale is adjusted to spread across the available range (0-255)
    based on the number of labels.

    Parameters:
    - predicted_mask: The predicted mask output from the model.
    - num_labels: The number of labels/categories in the dataset.

    Returns:
    - adjusted_mask: The adjusted mask with intensity levels matching the number of labels.
    """
    # Normalize the predicted mask to have values between 0 and 1
    normalized_mask = predicted_mask - predicted_mask.min()
    normalized_mask = normalized_mask / normalized_mask.max()

    # Scale the normalized mask to have intensity levels that match the number of labels
    intensity_scale = 255 / num_labels
    adjusted_mask = (normalized_mask * num_labels * intensity_scale).astype(np.uint8)

    return adjusted_mask

def create_mask(image_shape, annotations):
    if 'annotations' not in annotations or 'images' not in annotations:
        print("Error: Missing 'annotations' or 'images' key in data.")
        return None
    images = {}
    if not os.path.exists('seg_train/masks'):
        os.makedirs('seg_train/masks')
    segs = dict()
    for image_annotation in annotations['images']:
        images[image_annotation['id']] = image_annotation['file_name']
    
    for annotation in annotations['annotations']:
        for image_id, image_name in images.items():
            if annotation['image_id'] == image_id:
                segs[annotation['id']] = {
                    "image_name": image_name,
                    "category_id": annotation['category_id'],
                    "segmentation": annotation['segmentation']
                }

    segs_by_image_name = defaultdict(list)
    for k, v in segs.items():
        seg_info = (v['segmentation'], v['category_id'])
        segs_by_image_name[v["image_name"]].append(seg_info)

    for image_name, segmentations in segs_by_image_name.items():
        mask = np.zeros(image_shape, dtype=np.uint8)
        unique_categories = set([seg_info[1] for seg_info in segmentations])
        category_to_brightness = {category: (i + 1) * (255 // (len(unique_categories) + 1)) for i, category in enumerate(sorted(unique_categories))}
        for seg_info in segmentations:
            points, category_id = seg_info
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            # print cooridnates of the points
            print(points)
            cv2.fillPoly(mask, [points], color=category_to_brightness[category_id])
        mask_path = f'seg_train/masks/{image_name}'
        cv2.imwrite(mask_path, mask)



import json
import os
from collections import defaultdict
from PIL import Image
import base64

def load_categories(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    categories = {category['id']: category['name'] for category in data['categories']}
    return categories

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_labels(image_shape, annotations, output_dir, image_dir, categories):
    if 'annotations' not in annotations or 'images' not in annotations:
        print("Error: Missing 'annotations' or 'images' key in data.")
        return None

    images = {}
    segs = dict()

    for image_annotation in annotations['images']:
        images[image_annotation['id']] = image_annotation['file_name']
    
    for annotation in annotations['annotations']:
        for image_id, image_name in images.items():
            if annotation['image_id'] == image_id:
                segs[annotation['id']] = {
                    "image_name": image_name,
                    "category_id": annotation['category_id'],
                    "segmentation": annotation['segmentation']
                }

    segs_by_image_name = defaultdict(list)
    for k, v in segs.items():
        seg_info = (v['segmentation'], v['category_id'])
        segs_by_image_name[v["image_name"]].append(seg_info)

    for image_name, segments in segs_by_image_name.items():
        shapes = []
        for seg, category_id in segments:
            points = seg[0]
            paired_points = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
            shape = {
                "label": categories[category_id],  # Zamień category_id na nazwę kategorii
                "points": paired_points,  # Teraz punkty są parami (x, y)
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
        
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')  # Konwertuj obraz do RGB
        image_data = convert_image_to_base64(image_path)
        image_width, image_height = image.size

        result = {
            "version": "4.5.10",    
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": image_data,
            "imageHeight": image_height,
            "imageWidth": image_width
        }

        # Zapisz wynik do pliku JSON
        output_path = f"{output_dir}/{os.path.splitext(image_name)[0]}.json"
        with open(output_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)

# Ścieżka do pliku seg_train.json


# Przykładowe wywołanie funkcji convert_labels
# convert_labels(image_shape, annotations, output_dir, image_dir, categories)

# def convert_labels(image_shape, annotations, output_dir, image_dir):
#     if 'annotations' not in annotations or 'images' not in annotations:
#         print("Error: Missing 'annotations' or 'images' key in data.")
#         return None

#     images = {}
#     segs = dict()

#     for image_annotation in annotations['images']:
#         images[image_annotation['id']] = image_annotation['file_name']
    
#     for annotation in annotations['annotations']:
#         for image_id, image_name in images.items():
#             if annotation['image_id'] == image_id:
#                 segs[annotation['id']] = {
#                     "image_name": image_name,
#                     "category_id": annotation['category_id'],
#                     "segmentation": annotation['segmentation']
#                 }

#     segs_by_image_name = defaultdict(list)
#     for k, v in segs.items():
#         seg_info = (v['segmentation'], v['category_id'])
#         segs_by_image_name[v["image_name"]].append(seg_info)

#     for image_name, segments in segs_by_image_name.items():
#         shapes = []
#         for seg, category_id in segments:
#             points = seg[0]
#             paired_points = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
#             shape = {
#                 "label": category_id,  # Assuming category_id is the label, adjust if necessary
#                 "points": paired_points,  # Now points are pairs of (x, y)
#                 "group_id": None,
#                 "shape_type": "polygon",
#                 "flags": {}
#             }
#             shapes.append(shape)
        
#         image_path = os.path.join(image_dir, image_name)
#         image_data = convert_image_to_base64(image_path)
#         image = Image.open(image_path)
#         image_width, image_height = image.size

#         result = {
#             "version": "4.5.10",    
#             "flags": {},
#             "shapes": shapes,
#             "imagePath": image_name,
#             "imageData": image_data,
#             "imageHeight": image_height,
#             "imageWidth": image_width
#         }

#         # Zapisz wynik do pliku JSON
#         output_path = f"{output_dir}/{os.path.splitext(image_name)[0]}.json"
#         with open(output_path, 'w') as json_file:
#             json.dump(result, json_file, indent=4)






    # for image_name, segmentations in segs_by_image_name.items():
    #     mask = np.zeros(image_shape, dtype=np.uint8)
    #     unique_categories = set([seg_info[1] for seg_info in segmentations])
    #     category_to_brightness = {category: (i + 1) * (255 // (len(unique_categories) + 1)) for i, category in enumerate(sorted(unique_categories))}
    #     for seg_info in segmentations:
    #         points, category_id = seg_info
    #         points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    #         cv2.fillPoly(mask, [points], color=category_to_brightness[category_id])
    #     mask_path = f'seg_train/masks/{image_name}'
    #     cv2.imwrite(mask_path, mask)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx]).float() / 255.0
        mask = read_image(self.mask_paths[idx]).float() / 255.0
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def train_model(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}")
    print("Training complete")

def parse_image(image_path, mask_path, image_shape=(512, 512)):
    image = read_image(image_path).float() / 255.0
    image = resize(image, image_shape)
    mask = read_image(mask_path).float() / 255.0
    mask = resize(mask, image_shape, interpolation=0)  # Nearest neighbor interpolation for masks
    return image, mask

def load_dataset(image_dir, mask_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def iou(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training settings.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'voc'], default='train', help='Mode: train or predict.')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps per epoch.')
    
    args = parser.parse_args()
    image_shape = (512, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.mode == 'train':
        image_dir = 'seg_train/images'
        mask_dir = 'seg_train/masks/'
        os.makedirs(mask_dir, exist_ok=True)
        
        # Load JSON data
        with open('seg_train/annotations/seg_train.json', 'r') as file:
            annotations = json.load(file)
        
        # Create masks
        create_mask(image_shape, annotations)
        
        batch_size = 8
        dataloader = load_dataset(image_dir, mask_dir, batch_size)
        
        model = UNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_model(model, dataloader, criterion, optimizer, args.epochs, device)
        
        torch.save(model.state_dict(), f'models/unet_model_{args.epochs}_epochs_{args.steps}_steps.pth')
        print("Model saved successfully.")
    
    elif args.mode == 'predict':
        model = UNet().to(device)
        model.load_state_dict(torch.load(f'models/unet_model_{args.epochs}_epochs_{args.steps}_steps.pth'))
        model.eval()
        
        test_image_path = 'predict/5.png'
        test_image = read_image(test_image_path).float() / 255.0
        test_image = resize(test_image, image_shape).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_mask = model(test_image).cpu().numpy()[0, 0]
        
        adjusted_mask = adjust_predicted_mask_intensity(predicted_mask, num_labels=26)
        
        plt.figure(figsize=(8, 4))
        plt.imshow(adjusted_mask, cmap='jet')
        plt.colorbar()
        plt.title('Predicted Mask Probabilities')
        plt.axis('off')
        plt.show()
    elif args.mode == 'voc':
        json_path = 'seg_train/annotations/seg_train.json'
        categories = load_categories(json_path)
        with open('seg_train/annotations/seg_train.json', 'r') as file:
            annotations = json.load(file)
        convert_labels(image_shape, annotations, 'seg_train/labels', 'seg_train/images',categories)
        print("Labels converted successfully.")

