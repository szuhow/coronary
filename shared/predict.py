import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from unet import UNet

def find_model_file(model_dir, epoch):
    model_files = list(Path(model_dir).glob(f'best_model_epoch_{epoch}_*.pth'))
    if not model_files:
        raise FileNotFoundError(f"No model file found for epoch {epoch} in directory {model_dir}")
    return model_files[0]

def load_model(model_path):
    model = UNet(n_channels=1, n_classes=1, bilinear=False)  # Zastąp UNet odpowiednią klasą modelu
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def inference(model, image):
    # Step 2: Preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Konwersja obrazu RGB na obraz w skali szarości
        # transforms.Resize((256, 256)),  # Resize to the model's input size
        transforms.ToTensor(),         # Convert to a PyTorch tensor
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension



    # Wykonaj inferencję
    with torch.no_grad():
        output = model(input_tensor)

    # Przetwórz wyniki
    output = torch.sigmoid(output)  # Apply sigmoid if binary segmentation
    output_image = output.squeeze().cpu().numpy()  # Convert back to numpy for visualization
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
    output_image = output_image.astype('uint8')
    print("Output min:", output_image.min())
    print("Output max:", output_image.max())
    print("Output mean:", output_image.mean())

    return output_image

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the trained models')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number of the model to load')
    args = parser.parse_args()

    # Znajdź odpowiedni plik modelu na podstawie numeru epoki
    model_path = find_model_file(args.model_dir, args.epoch)
    print(f"Loading model from {model_path}")

    # Załaduj model
    model = load_model(model_path)


    # Wczytaj obraz
    image = Image.open(args.image_path).convert('RGB')

    # Wykonaj inferencję
    output_image = inference(model, image)
    
    # image = Image.open(args.image_path).convert('RGB')

    # Wizualizacja wyników (opcjonalnie)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Model Output")
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()