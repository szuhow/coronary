import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Load JSON data
with open('seg_train/annotations/seg_train.json', 'r') as file:
    annotations = json.load(file)
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
    print(intensity_scale)
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

    # for image_name, segmentations in segs_by_image_name.items():
    #     mask = np.zeros(image_shape, dtype=np.uint8)
    #     brightness_step = 255 // len(segmentations)  # Ensure different brightness levels
    #     # current_brightness = brightness_step
    #     for seg_info in segmentations:
    #         points, category_id = seg_info  # Ignore category_id for brightness levels
    #         points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    #         cv2.fillPoly(mask, [points], color=category_id)
    #         # current_brightness += brightness_step
    #     mask_path = f'seg_train/masks/{image_name}'
    #     cv2.imwrite(mask_path, mask)
    for image_name, segmentations in segs_by_image_name.items():
        mask = np.zeros(image_shape, dtype=np.uint8)
        unique_categories = set([seg_info[1] for seg_info in segmentations])
        category_to_brightness = {category: (i + 1) * (255 // (len(unique_categories) + 1)) for i, category in enumerate(sorted(unique_categories))}
        for seg_info in segmentations:
            points, category_id = seg_info
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], color=category_to_brightness[category_id])
        mask_path = f'seg_train/masks/{image_name}'
        cv2.imwrite(mask_path, mask)

def unet_model(input_size=(512, 512, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    conv6 = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=[inputs], outputs=[conv6])
    
    return model

def parse_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, (512, 512))
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

def load_dataset(image_dir, mask_dir, batch_size):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + 1e-9) / (union + 1e-9)
    return tf.reduce_mean(f(y_true, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training settings.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Mode: train or predict.')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps per epoch.')
    args = parser.parse_args()
    image_shape = (512, 512)
    if args.mode == 'train':
        image_dir = 'seg_train/images'
        mask_dir = 'seg_train/masks/'
        os.makedirs(mask_dir, exist_ok=True)
        
        # Create masks
        create_mask(image_shape, annotations)
        
        batch_size = 8
        dataset = load_dataset(image_dir, mask_dir, batch_size)
        
        model = unet_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', iou])

        history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=args.steps // batch_size)
        
        model.save(f'models/unet_model_{args.epochs}_epochs_{args.steps}_steps.h5')
        print("Model saved successfully.")
    elif args.mode == 'predict':
        loaded_model = tf.keras.models.load_model(f'models/unet_model_{args.epochs}_epochs_{args.steps}_steps.h5')

        test_image_path = 'predict/5.png'
        test_image = load_img(test_image_path, color_mode='grayscale', target_size=image_shape)
        test_image_array = img_to_array(test_image) / 255.0

        test_image_input = np.expand_dims(test_image_array, axis=0)
        plt.figure(figsize=(8, 4))
        plt.imshow(test_image_array[:, :, 0], cmap='gray')
        plt.title('Test Image (Normalized)')
        plt.axis('off')
        plt.show()

        predicted_mask = loaded_model.predict(test_image_input) 
        adjusted_mask = adjust_predicted_mask_intensity(predicted_mask[0, :, :, 0], num_labels=26)

        #  multiply by 255 to get the brightness levels
        
        plt.figure(figsize=(8, 4))
        plt.imshow(adjusted_mask, cmap='jet')
        plt.colorbar()
        plt.title('Predicted Mask Probabilities')
        plt.axis('off')
        plt.show()
        # predicted_mask_image = np.squeeze(predicted_mask) * 255
        # plt.figure(figsize=(8, 4))
        # plt.imshow(predicted_mask, cmap='gray')
        # plt.colorbar()
        # plt.title('Predicted Mask with Brightness Levels')
        # plt.axis('off')
        # plt.show()