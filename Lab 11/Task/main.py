# import cv2
# import os
# import shutil
# from ultralytics import YOLO

# def diagnose_setup(images_folder="imagesX"):
#     """
#     Diagnose the setup and check for common issues
#     """
#     print("\n" + "="*60)
#     print("DIAGNOSTIC CHECK")
#     print("="*60)
    
#     # 1. Check current working directory
#     current_dir = os.getcwd()
#     print(f"\n1. Current working directory:")
#     print(f"   {current_dir}")
    
#     # 2. Check if imagesX folder exists
#     print(f"\n2. Looking for folder: '{images_folder}'")
#     if os.path.exists(images_folder):
#         print(f"   ✓ Folder EXISTS at: {os.path.abspath(images_folder)}")
#     else:
#         print(f"   ✗ Folder NOT FOUND!")
#         print(f"   Looking for: {os.path.abspath(images_folder)}")
#         print("\n   Available folders in current directory:")
#         folders = [f for f in os.listdir('.') if os.path.isdir(f)]
#         for folder in folders:
#             print(f"     - {folder}")
#         return False
    
#     # 3. Check if it's a directory
#     if not os.path.isdir(images_folder):
#         print(f"   ✗ '{images_folder}' exists but is not a directory!")
#         return False
    
#     # 4. List all contents
#     print(f"\n3. Contents of '{images_folder}':")
#     all_items = os.listdir(images_folder)
#     print(f"   Total items: {len(all_items)}")
    
#     # 5. Check for image files
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.gif', '.GIF')
    
#     print(f"\n   Checking each item:")
#     image_files = []
#     for f in all_items:
#         full_path = os.path.join(images_folder, f)
#         is_file = os.path.isfile(full_path)
#         if is_file:
#             print(f"     FILE: '{f}' -> ends with image ext? {f.lower().endswith(image_extensions)}")
#             if any(f.lower().endswith(ext) for ext in image_extensions):
#                 image_files.append(f)
#         else:
#             print(f"     DIR:  '{f}'")
    
#     print(f"\n4. Image files found: {len(image_files)}")
#     if image_files:
#         print("   Images:")
#         for img in image_files:
#             full_path = os.path.join(images_folder, img)
#             size = os.path.getsize(full_path)
#             print(f"     ✓ {img} ({size} bytes)")
#     else:
#         print("   ✗ No image files found!")
#         print("\n   All items in folder:")
#         for item in all_items:
#             item_path = os.path.join(images_folder, item)
#             item_type = "DIR" if os.path.isdir(item_path) else "FILE"
#             print(f"     - {item} ({item_type})")
    
#     # 6. Check subfolders
#     print(f"\n5. Subfolders:")
#     subfolders = ['non-cheating', 'cheating', 'differentX']
#     for subfolder in subfolders:
#         subfolder_path = os.path.join(images_folder, subfolder)
#         if os.path.exists(subfolder_path):
#             count = len(os.listdir(subfolder_path))
#             print(f"   ✓ {subfolder}/ exists ({count} items)")
#         else:
#             print(f"   - {subfolder}/ does not exist (will be created)")
    
#     print("\n" + "="*60)
    
#     if len(image_files) == 0:
#         print("\n⚠ ISSUE: No images found!")
#         print("\nPossible solutions:")
#         print("1. Check if images are directly in 'imagesX' folder (not in subfolders)")
#         print("2. Verify image file extensions (.jpg, .png, etc.)")
#         print("3. Make sure script is in the correct directory")
#         print("4. Try providing full path: classify_images('C:/full/path/to/imagesX')")
#         return False
    
#     return True

# def classify_images(images_folder="imagesX"):
#     """
#     Classify images using YOLO object detection:
#     - cheating: CLEAR face + object detected
#     - non-cheating: only CLEAR face detected
#     - differentX: object detected but no clear face (includes hands holding objects)
#     """
    
#     # Load YOLO model (YOLOv8)
#     print("Loading YOLO model...")
#     model = YOLO('yolov8n.pt')  # Using nano model for speed, use 'yolov8s.pt' for better accuracy
    
#     # Load face cascade - use only the most reliable one with STRICT settings
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
#     # Define folder paths
#     non_cheating_folder = os.path.join(images_folder, "non-cheating")
#     cheating_folder = os.path.join(images_folder, "cheating")
#     different_folder = os.path.join(images_folder, "differentX")
    
#     # Create folders if they don't exist
#     for folder in [non_cheating_folder, cheating_folder, different_folder]:
#         os.makedirs(folder, exist_ok=True)
    
#     # Get all image files from imagesX folder (excluding subfolders)
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
#     image_files = []
#     for f in os.listdir(images_folder):
#         full_path = os.path.join(images_folder, f)
#         if os.path.isfile(full_path) and any(f.lower().endswith(ext) for ext in image_extensions):
#             image_files.append(f)
    
#     print(f"\nFound {len(image_files)} images to process\n")
    
#     # Process each image
#     for idx, image_file in enumerate(image_files, 1):
#         image_path = os.path.join(images_folder, image_file)
#         print(f"Processing [{idx}/{len(image_files)}]: {image_file}")
        
#         # Read image for face detection
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"  ⚠ Could not read image, skipping...")
#             continue
        
#         # Create a copy for visualization
#         img_labeled = img.copy()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # STRICT face detection - only detect CLEAR, PROMINENT faces
#         faces = face_cascade.detectMultiScale(
#             gray, 
#             scaleFactor=1.1, 
#             minNeighbors=8,  # Very strict - need strong evidence of face
#             minSize=(60, 60)  # Face must be reasonably large (not tiny/blurry)
#         )
        
#         # Check if the detected face is large enough
#         has_clear_face = False
#         img_area = img.shape[0] * img.shape[1]
#         face_boxes = []
        
#         if len(faces) > 0:
#             for (x, y, w, h) in faces:
#                 face_area = w * h
#                 if face_area / img_area > 0.08:  # Face is at least 8% of image
#                     has_clear_face = True
#                     face_boxes.append((x, y, w, h))
#                     # Draw green rectangle around face
#                     cv2.rectangle(img_labeled, (x, y), (x+w, y+h), (0, 255, 0), 3)
#                     cv2.putText(img_labeled, 'FACE', (x, y-10), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         # Run YOLO detection for objects
#         results = model(image_path, verbose=False)
        
#         # Extract and draw detected objects
#         detected_objects = []
#         if len(results) > 0 and results[0].boxes is not None:
#             for box in results[0].boxes:
#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]
                
#                 # Only count non-person objects
#                 if class_name != 'person':
#                     detected_objects.append(class_name)
                    
#                     # Draw blue rectangle around object
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(img_labeled, (x1, y1), (x2, y2), (255, 0, 0), 3)
#                     cv2.putText(img_labeled, class_name.upper(), (x1, y1-10), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
#         has_object = len(detected_objects) > 0
        
#         # Debug output
#         print(f"  Clear face: {has_clear_face}, Objects: {detected_objects if has_object else 'None'}")
        
#         # Determine classification
#         if has_clear_face and has_object:
#             classification = "CHEATING"
#             destination_folder = cheating_folder
#             color = (0, 0, 255)  # Red
#             print(f"  ✓ Moved to 'cheating' (Clear face + Objects: {detected_objects})")
            
#         elif has_clear_face and not has_object:
#             classification = "NON-CHEATING"
#             destination_folder = non_cheating_folder
#             color = (0, 255, 0)  # Green
#             print(f"  ✓ Moved to 'non-cheating' (Only clear face detected)")
            
#         elif has_object:
#             classification = "DIFFERENT"
#             destination_folder = different_folder
#             color = (255, 165, 0)  # Orange
#             print(f"  ✓ Moved to 'differentX' (Objects without clear face: {detected_objects})")
            
#         else:
#             classification = "DIFFERENT"
#             destination_folder = different_folder
#             color = (255, 165, 0)  # Orange
#             print(f"  ⚠ No clear features, moving to 'differentX'")
        
#         # Add classification label at top of image
#         label_text = f"CLASS: {classification}"
#         cv2.rectangle(img_labeled, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
#         cv2.putText(img_labeled, label_text, (10, 35), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
#         # Save labeled image
#         destination = os.path.join(destination_folder, image_file)
#         cv2.imwrite(destination, img_labeled)
        
#         print()
    
#     # Print summary
#     print("\n" + "="*60)
#     print("CLASSIFICATION SUMMARY")
#     print("="*60)
#     print(f"Non-cheating (clear face only):      {len(os.listdir(non_cheating_folder))} images")
#     print(f"Cheating (clear face + objects):     {len(os.listdir(cheating_folder))} images")
#     print(f"DifferentX (objects, no clear face): {len(os.listdir(different_folder))} images")
#     print("\n📝 All images have been labeled with:")
#     print("   - GREEN boxes = Detected faces")
#     print("   - BLUE boxes = Detected objects")
#     print("   - Top label = Classification result")
#     print("="*60)

# if __name__ == "__main__":
#     # Install required packages if needed
#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         print("Installing required packages...")
#         os.system("pip install ultralytics opencv-python")
#         from ultralytics import YOLO
    
#     # First run diagnostics
#     print("\nRunning diagnostic checks...")
#     if not diagnose_setup("imagesX"):
#         print("\n❌ Diagnostic check failed. Please fix the issues above.")
#         exit(1)
    
#     # Ask user to confirm
#     print("\n" + "="*60)
#     response = input("Proceed with classification? (yes/no): ").strip().lower()
#     if response not in ['yes', 'y']:
#         print("Classification cancelled.")
#         exit(0)
    
#     # Run classification
#     classify_images("imagesX")

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PART 1: DATA AUGMENTATION
# ============================================================================

def create_augmented_dataset(source_folder="imagesX", output_folder="AugImagesX"):
    """
    Apply 5 augmentation methods to expand the dataset.
    Takes images from imagesX/non-cheating, imagesX/cheating, imagesX/differentX
    """
    
    print("\n" + "="*60)
    print("PART 1: DATA AUGMENTATION - EXPANDING DATASET")
    print("="*60)
    
    # Create output folder structure
    output_path = Path(output_folder)
    subfolders = ['non-cheating', 'cheating', 'differentX']
    
    for subfolder in subfolders:
        (output_path / subfolder).mkdir(parents=True, exist_ok=True)
    
    # Process each category from imagesX subfolders
    source_path = Path(source_folder)
    total_original = 0
    total_augmented = 0
    
    for subfolder in subfolders:
        source_subfolder = source_path / subfolder
        output_subfolder = output_path / subfolder
        
        print(f"\nChecking folder: {source_subfolder}")
        print(f"  Exists: {source_subfolder.exists()}")
        print(f"  Is directory: {source_subfolder.is_dir() if source_subfolder.exists() else 'N/A'}")
        
        if not source_subfolder.exists():
            print(f"  ⚠ Warning: {source_subfolder} folder not found, skipping...")
            continue
        
        # Get all images from this category
        all_files = list(source_subfolder.iterdir())
        print(f"  Total files in folder: {len(all_files)}")
        
        image_files = [f for f in all_files 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"  Image files found: {len(image_files)}")
        if len(image_files) > 0:
            print(f"  First few: {[f.name for f in image_files[:3]]}")
        
        if len(image_files) == 0:
            print(f"  ⚠ No image files with .jpg, .jpeg, or .png extension")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {subfolder.upper()}")
        print(f"Original images: {len(image_files)}")
        print(f"{'='*60}")
        
        category_count = 0
        
        for img_file in image_files:
            try:
                # Read original image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"  ⚠ Could not read {img_file.name}, skipping...")
                    continue
                
                # Save original image
                original_name = img_file.stem + img_file.suffix
                cv2.imwrite(str(output_subfolder / original_name), img)
                total_original += 1
                category_count += 1
                
                print(f"\n  Processing: {img_file.name}")
                
                # Method 1: Horizontal Flip
                flipped = cv2.flip(img, 1)
                aug_name = f"{img_file.stem}_flip{img_file.suffix}"
                cv2.imwrite(str(output_subfolder / aug_name), flipped)
                print(f"    ✓ Horizontal flip")
                category_count += 1
                
                # Method 2: Rotation (15 degrees)
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
                rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
                aug_name = f"{img_file.stem}_rot15{img_file.suffix}"
                cv2.imwrite(str(output_subfolder / aug_name), rotated)
                print(f"    ✓ Rotation (15°)")
                category_count += 1
                
                # Method 3: Brightness Adjustment (increase brightness)
                bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
                aug_name = f"{img_file.stem}_bright{img_file.suffix}"
                cv2.imwrite(str(output_subfolder / aug_name), bright)
                print(f"    ✓ Brightness increase")
                category_count += 1
                
                # Method 4: Gaussian Blur
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                aug_name = f"{img_file.stem}_blur{img_file.suffix}"
                cv2.imwrite(str(output_subfolder / aug_name), blurred)
                print(f"    ✓ Gaussian blur")
                category_count += 1
                
                # Method 5: Zoom (crop and resize)
                h, w = img.shape[:2]
                crop_percent = 0.15
                crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
                cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
                zoomed = cv2.resize(cropped, (w, h))
                aug_name = f"{img_file.stem}_zoom{img_file.suffix}"
                cv2.imwrite(str(output_subfolder / aug_name), zoomed)
                print(f"    ✓ Zoom/crop")
                category_count += 1
                
            except Exception as e:
                print(f"  ⚠ Error processing {img_file.name}: {e}")
                continue
        
        augmented_count = category_count - len(image_files)
        total_augmented += augmented_count
        print(f"\n  Summary for {subfolder}:")
        print(f"    Original: {len(image_files)}")
        print(f"    Total after augmentation: {category_count}")
        print(f"    New images created: {augmented_count}")
    
    # Final summary
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE!")
    print("="*60)
    print(f"Total original images copied: {total_original}")
    print(f"Total new augmented images: {total_augmented}")
    print(f"Total dataset size: {total_original + total_augmented}")
    print(f"\nAugmentation methods applied:")
    print("  1. Horizontal flip")
    print("  2. Rotation (15 degrees)")
    print("  3. Brightness adjustment")
    print("  4. Gaussian blur")
    print("  5. Zoom/crop")
    print(f"\nOutput folder: {output_folder}/")
    print("="*60)
    
    return total_original + total_augmented > 0

# ============================================================================
# PART 2: VISION TRANSFORMER MODEL
# ============================================================================

class CheatingDataset(Dataset):
    """Custom dataset for cheating detection"""
    
    def __init__(self, root_dir, processor, split='train', train_ratio=0.8):
        self.processor = processor
        self.images = []
        self.labels = []
        self.class_names = ['non-cheating', 'cheating', 'differentX']
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load images from each category
        root_path = Path(root_dir)
        all_data = []
        
        for class_name in self.class_names:
            class_path = root_path / class_name
            if not class_path.exists():
                print(f"Warning: {class_name} folder not found")
                continue
            
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
            
            for img_path in image_files:
                all_data.append((str(img_path), self.label_map[class_name]))
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(all_data)
        
        split_idx = int(len(all_data) * train_ratio)
        
        if split == 'train':
            data = all_data[:split_idx]
        else:
            data = all_data[split_idx:]
        
        self.images = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        
        print(f"{split.upper()} set: {len(self.images)} images")
        for i, class_name in enumerate(self.class_names):
            count = self.labels.count(i)
            print(f"  - {class_name}: {count}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), label

def train_vit_model(data_dir='AugImagesX', epochs=10, batch_size=8, learning_rate=2e-5):
    """
    Train Vision Transformer model for cheating detection
    """
    
    print("\n" + "="*60)
    print("PART 2: VISION TRANSFORMER (ViT) TRAINING")
    print("="*60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load pre-trained ViT model and processor
    print("\nLoading pre-trained ViT model...")
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CheatingDataset(data_dir, processor, split='train')
    val_dataset = CheatingDataset(data_dir, processor, split='val')
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n❌ Error: Not enough data to train. Check your AugImagesX folder.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-"*60)
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained("vit_cheating_detector")
    processor.save_pretrained("vit_cheating_detector")
    print("Model saved to: vit_cheating_detector/")
    
    # Evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    class_names = ['non-cheating', 'cheating', 'differentX']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as: confusion_matrix.png")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as: training_history.png")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Model saved to: vit_cheating_detector/")
    print("="*60)

def predict_image(image_path, model_dir='vit_cheating_detector'):
    """
    Predict a single image using trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ViTForImageClassification.from_pretrained(model_dir)
    processor = ViTImageProcessor.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_names = ['non-cheating', 'cheating', 'differentX']
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"\nPrediction for: {image_path}")
    print(f"Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probabilities[0][i].item()*100:.2f}%")
    
    return class_names[predicted_class], confidence

# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPLETE PIPELINE: AUGMENTATION + ViT TRAINING")
    print("="*60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import transformers
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers torch torchvision pillow scikit-learn matplotlib seaborn")
    
    # Step 1: Data Augmentation
    print("\n" + "="*60)
    print("STEP 1: DATA AUGMENTATION")
    print("="*60)
    success = create_augmented_dataset("imagesX", "AugImagesX")
    
    if not success:
        print("\n❌ Error: No images found to augment. Check your imagesX folder structure.")
        print("Expected structure:")
        print("  imagesX/")
        print("    ├── non-cheating/")
        print("    ├── cheating/")
        print("    └── differentX/")
        exit(1)
    
    # Step 2: Train ViT Model
    print("\n" + "="*60)
    print("STEP 2: VISION TRANSFORMER TRAINING")
    print("="*60)
    
    response = input("\nProceed with ViT training? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        train_vit_model(
            data_dir='AugImagesX',
            epochs=10,
            batch_size=8,
            learning_rate=2e-5
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\nTo predict a new image, run:")
        print("  python pipeline.py")
        print("Then use: predict_image('path/to/image.jpg')")
        print("="*60)
    else:
        print("\nTraining cancelled. Augmented data saved in AugImagesX/")