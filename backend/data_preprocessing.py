import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import pickle
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

class FER2013DataProcessor:
    """
    Data preprocessing class for FER-2013 dataset
    Handles loading, preprocessing, and data augmentation
    """
    
    def __init__(self, data_path: str = "data/FER2013/"):
        self.data_path = data_path
        self.img_size = (48, 48)
        self.num_classes = 7
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        
        # Create directories if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs("model/saved_models", exist_ok=True)
        
    def load_fer2013_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load FER-2013 dataset from CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (images, labels, usage_types)
        """
        print("Loading FER-2013 dataset...")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Extract pixel data and convert to numpy arrays
        images = []
        labels = []
        usage_types = df['Usage'].tolist() if 'Usage' in df.columns else []
        
        for idx, row in df.iterrows():
            # Convert pixel string to numpy array
            pixel_data = row['pixels'].split(' ')
            pixel_array = np.array([int(pixel) for pixel in pixel_data], dtype=np.uint8)
            
            # Reshape to 48x48 image
            image = pixel_array.reshape(48, 48)
            images.append(image)
            labels.append(row['emotion'])
            
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images")
        print(f"Image shape: {images[0].shape}")
        print(f"Number of classes: {len(np.unique(labels))}")
        
        return images, labels, usage_types
    
    def preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images for training
        
        Args:
            images: Raw image array
            
        Returns:
            Preprocessed images
        """
        print("Preprocessing images...")
        
        # Normalize pixel values to [0, 1]
        images = images.astype('float32') / 255.0
        
        # Add channel dimension for grayscale images
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
            
        return images
    
    def augment_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to increase dataset size
        
        Args:
            images: Original images (expected shape: N, H, W, 1)
            labels: Original labels
            
        Returns:
            Augmented images and labels
        """
        print("Applying data augmentation...")
        
        augmented_images = []
        augmented_labels = []
        
        # Ensure images have the right shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            # Remove channel dimension for CV2 operations
            img_2d = img.squeeze() if img.shape[-1] == 1 else img
            
            # Original image
            augmented_images.append(img)  # Keep original 3D shape
            augmented_labels.append(label)
            
            # Horizontal flip
            flipped = cv2.flip(img_2d, 1)
            # Add channel dimension back
            flipped = np.expand_dims(flipped, axis=-1)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
            # Slight rotation (-15 to +15 degrees)
            angle = np.random.uniform(-15, 15)
            center = (img_2d.shape[1]//2, img_2d.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_2d, rotation_matrix, (img_2d.shape[1], img_2d.shape[0]))
            # Add channel dimension back
            rotated = np.expand_dims(rotated, axis=-1)
            augmented_images.append(rotated)
            augmented_labels.append(label)
            
            # Brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            brightened = np.clip(img * brightness_factor, 0, 1)
            augmented_images.append(brightened)
            augmented_labels.append(label)
            
            if i % 1000 == 0:
                print(f"Augmented {i}/{len(images)} images")
        
        # Convert to numpy arrays - all images should now have consistent shape
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        print(f"Final augmented shape: {augmented_images.shape}")
        return augmented_images, augmented_labels
    
    def split_data(self, images: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            images: Preprocessed images
            labels: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Splitting data...")
        
        # Convert labels to categorical
        labels_categorical = to_categorical(labels, num_classes=self.num_classes)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels_categorical, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_data_distribution(self, labels: np.ndarray, save_path: str = None):
        """
        Visualize the distribution of emotions in the dataset
        
        Args:
            labels: Array of emotion labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Count distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        emotion_names = [self.emotion_labels[label] for label in unique_labels]
        
        # Create bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(emotion_names, counts, color='skyblue', alpha=0.7)
        plt.title('Emotion Distribution in Dataset')
        plt.xlabel('Emotions')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    str(count), ha='center', va='bottom')
        
        # Create pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=emotion_names, autopct='%1.1f%%', startangle=90)
        plt.title('Emotion Distribution (Percentage)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def visualize_sample_images(self, images: np.ndarray, labels: np.ndarray, 
                               num_samples: int = 21, save_path: str = None):
        """
        Visualize sample images from each emotion class
        
        Args:
            images: Image array
            labels: Label array
            num_samples: Number of samples to show
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        samples_per_class = num_samples // self.num_classes
        
        for class_idx in range(self.num_classes):
            # Find indices for this class
            class_indices = np.where(labels == class_idx)[0]
            
            # Select random samples
            selected_indices = np.random.choice(class_indices, 
                                              min(samples_per_class, len(class_indices)), 
                                              replace=False)
            
            for i, idx in enumerate(selected_indices):
                plt.subplot(self.num_classes, samples_per_class, 
                           class_idx * samples_per_class + i + 1)
                
                # Display image
                img = images[idx]
                if len(img.shape) == 3 and img.shape[-1] == 1:
                    img = img.squeeze()
                    
                plt.imshow(img, cmap='gray')
                plt.title(f'{self.emotion_labels[class_idx]}')
                plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample images saved to {save_path}")
            
        plt.show()
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           file_path: str = "data/processed_fer2013.pkl"):
        """
        Save processed data to pickle file
        
        Args:
            X_train, X_val, X_test: Image data splits
            y_train, y_val, y_test: Label data splits
            file_path: Path to save the pickle file
        """
        print("Saving processed data...")
        
        data_dict = {
            'X_train': X_train,
            'X_val': X_val, 
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'emotion_labels': self.emotion_labels,
            'img_size': self.img_size,
            'num_classes': self.num_classes
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"Processed data saved to {file_path}")
    
    def load_processed_data(self, file_path: str = "data/processed_fer2013.pkl"):
        """
        Load processed data from pickle file
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Dictionary containing processed data
        """
        print("Loading processed data...")
        
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        print("Processed data loaded successfully")
        return data_dict
    
    def process_complete_pipeline(self, csv_path: str, apply_augmentation: bool = True):
        """
        Complete data processing pipeline
        
        Args:
            csv_path: Path to FER-2013 CSV file
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Processed data splits
        """
        print("Starting complete data processing pipeline...")
        
        # Load data
        images, labels, usage_types = self.load_fer2013_csv(csv_path)
        
        # Visualize original data
        self.visualize_data_distribution(labels, "data/emotion_distribution.png")
        self.visualize_sample_images(images, labels, save_path="data/sample_images.png")
        
        # Preprocess images
        images = self.preprocess_images(images)
        
        # Apply augmentation if requested
        if apply_augmentation:
            images, labels = self.augment_data(images, labels)
            print(f"Dataset size after augmentation: {len(images)}")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(images, labels)
        
        # Save processed data
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("Data processing pipeline completed successfully!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = FER2013DataProcessor()
    
    # Process data (replace with your CSV path)
    csv_path = "data/FER2013/fer2013.csv"
    
    try:
        # Run complete pipeline
        X_train, X_val, X_test, y_train, y_val, y_test = processor.process_complete_pipeline(
            csv_path, apply_augmentation=True
        )
        
        print("\nData processing completed!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}")
        print("Please download the FER-2013 dataset and place fer2013.csv in the data/FER2013/ directory")
        print("Dataset available at: https://www.kaggle.com/datasets/msambare/fer2013")