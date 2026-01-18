import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
import os
from typing import Tuple, Dict, List, Union
import pickle

import mediapipe as mp

class EmotionPredictor:
    """
    Utility class for loading trained models and making predictions
    Designed to be used by Flask API and other applications
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_info = None
        self.emotion_labels = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        self.img_size = (48, 48)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained emotion detection model
        
        Args:
            model_path: Path to the saved model (.h5 file)
        """
        try:
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
            
            # Try to load model info
            info_path = model_path.replace('.h5', '_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                print("Model info loaded successfully")
            else:
                print("Model info file not found, using defaults")
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image according to model input shape. Supports (48,48,1) and (H,W,3) models.
        CLAHE improves contrast; standardization stabilizes logits.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        input_shape = self.model.input_shape     # e.g. (None, 48,48,1) or (None, 224,224,3)
        H, W, C = input_shape[1], input_shape[2], input_shape[3]

        # Base: CLAHE on gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        if (H, W, C) == (48, 48, 1):
            image = cv2.resize(gray, (W, H)).astype("float32") / 255.0
            image = (image - image.mean()) / (image.std() + 1e-6)
            image = np.expand_dims(image, (0, -1))  # (1,48,48,1)
            return image
        else:
            # For 3-channel backbones (e.g., EfficientNet 224x224x3)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (W, H)).astype("float32") / 255.0
            g3  = np.repeat(cv2.resize(gray, (W, H))[..., None], 3, axis=-1).astype("float32") / 255.0
            image = 0.7 * rgb + 0.3 * g3
            image = (image - image.mean()) / (image.std() + 1e-6)
            image = np.expand_dims(image, 0)  # (1,H,W,3)
            return image

    
    def predict_emotion(self, image: np.ndarray, return_probabilities: bool = False) -> Union[str, Dict]:
        """
        Predict emotion from a single image
        
        Args:
            image: Input image
            return_probabilities: Whether to return probability scores
            
        Returns:
            Predicted emotion or dictionary with emotion and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        predicted_emotion = self.emotion_labels[predicted_class]
        confidence = float(predictions[predicted_class])
        
        if return_probabilities:
            # Create probability dictionary
            probabilities = {}
            for i, prob in enumerate(predictions):
                probabilities[self.emotion_labels[i]] = float(prob)
            
            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
        else:
            return predicted_emotion
    
    def predict_batch(self, images: List[np.ndarray], return_probabilities: bool = False) -> List[Union[str, Dict]]:
        """
        Predict emotions for multiple images
        
        Args:
            images: List of input images
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        results = []
        for image in images:
            result = self.predict_emotion(image, return_probabilities)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_loaded": True,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "total_parameters": self.model.count_params(),
            "emotion_labels": self.emotion_labels
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        return info


class ModelManager:
    """
    Utility class for managing multiple trained models
    """

    def __init__(self, models_directory: str = "model/saved_models"):
        self.models_directory = models_directory
        self.available_models = self._scan_models()
        self.predictor = None  # ✅ will hold an EmotionPredictor once loaded

    def _scan_models(self) -> Dict[str, str]:
        models = {}
        if not os.path.exists(self.models_directory):
            print(f"Models directory {self.models_directory} not found")
            return models

        for file in os.listdir(self.models_directory):
            if file.endswith('.h5'):
                model_name = file.replace('.h5', '')
                model_path = os.path.join(self.models_directory, file)
                models[model_name] = model_path

        return models

    def list_models(self) -> List[str]:
        return list(self.available_models.keys())

    def get_model_path(self, model_name: str) -> str:
        if model_name in self.available_models:
            return self.available_models[model_name]
        else:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {self.list_models()}"
            )

    def load_model(self, model_name: str) -> EmotionPredictor:
        model_path = self.get_model_path(model_name)
        self.predictor = EmotionPredictor(model_path)   # ✅ store predictor
        return self.predictor

    def get_best_model(self) -> EmotionPredictor:
        best_models = [name for name in self.available_models.keys() if 'best' in name.lower()]
        if best_models:
            best_model_name = best_models[0]
            print(f"Loading best model: {best_model_name}")
            return self.load_model(best_model_name)
        elif self.available_models:
            first_model = list(self.available_models.keys())[0]
            print(f"No 'best' model found, loading: {first_model}")
            return self.load_model(first_model)
        else:
            raise ValueError("No models found in the models directory")



class ImageProcessor:
    """
    Utility class for image preprocessing and face detection
    """
    
    def __init__(self):
        # Load face cascade classifier
        self.face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """
        Use MediaPipe face detection for robust, real-time face boxes.
        """
        try:
            self._mp_fd = mp.solutions.face_detection
            # model_selection=1 => full-range; 0 => short-range
            self._fd = self._mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.6)
            print("MediaPipe face detector initialized")
        except Exception as e:
            print(f"Error initializing MediaPipe face detector: {e}")
            self._fd = None

    
    def detect_faces(self, image: np.ndarray):
        """Return list of (x0,y0,x1,y1) using MediaPipe."""
        if getattr(self, "_fd", None) is None:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self._fd.process(rgb)
        boxes = []
        if res and res.detections:
            h, w = image.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x, y, bw, bh = bb.xmin, bb.ymin, bb.width, bb.height
                # expand and clamp (margin improves emotion context)
                m = 0.15
                x0 = max(int((x - m) * w), 0)
                y0 = max(int((y - m) * h), 0)
                x1 = min(int((x + bw + m) * w), w)
                y1 = min(int((y + bh + m) * h), h)
                if x1 > x0 and y1 > y0:
                    boxes.append((x0, y0, x1, y1))
        return boxes
    
    def _align_by_eyes(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Rotate face so eye line is horizontal using MediaPipe FaceMesh (best-effort).
        Falls back to original face if landmarks not found.
        """
        try:
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
                res = mesh.process(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                if not res.multi_face_landmarks:
                    return face_bgr
                lm = res.multi_face_landmarks[0].landmark
                # rough outer eye corners: 33 (left), 263 (right)
                h, w = face_bgr.shape[:2]
                L = (lm[33].x * w, lm[33].y * h)
                R = (lm[263].x * w, lm[263].y * h)
                angle = np.degrees(np.arctan2(R[1] - L[1], R[0] - L[0]))
                M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
                return cv2.warpAffine(face_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        except Exception:
            return face_bgr
        

    
    def extract_face_roi(self, image: np.ndarray, face_coords: Tuple[int, int, int, int], 
                        padding: float = 0.2) -> np.ndarray:
        """
        Extract face region of interest from image
        
        Args:
            image: Input image
            face_coords: Face bounding box (x, y, width, height)
            padding: Additional padding around face (as fraction of face size)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_coords
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate extended coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Extract face ROI
        face_roi = image[y1:y2, x1:x2]
        
        # 🔹 Align face before resize
        face_roi = self._align_by_eyes(face_roi)
        
        return face_roi
    
    def process_image_for_emotion(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple]]:
        """
        Process image to extract faces for emotion detection
        
        Args:
            image: Input image
            
        Returns:
            List of tuples (face_image, face_coordinates)
        """
        faces = self.detect_faces(image)
        face_images = []
        
        for face_coords in faces:
            face_roi = self.extract_face_roi(image, face_coords)
            face_images.append((face_roi, face_coords))
        
        return face_images


def load_test_data(data_path: str = "data/processed_fer2013.pkl") -> Tuple:
    """
    Utility function to load test data for model validation
    
    Args:
        data_path: Path to processed data file
        
    Returns:
        Tuple of (X_test, y_test)
    """
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    return data_dict['X_test'], data_dict['y_test']


def create_sample_predictions(model_path: str, num_samples: int = 10):
    """
    Create sample predictions for testing
    
    Args:
        model_path: Path to trained model
        num_samples: Number of sample predictions to make
    """
    # Load model and test data
    predictor = EmotionPredictor(model_path)
    X_test, y_test = load_test_data()
    
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    print("Sample Predictions:")
    print("-" * 50)
    
    for i, idx in enumerate(indices):
        # Get true label
        true_label_idx = np.argmax(y_test[idx])
        true_label = predictor.emotion_labels[true_label_idx]
        
        # Make prediction
        test_image = X_test[idx]
        if len(test_image.shape) == 3:
            test_image = test_image.squeeze()
        
        prediction = predictor.predict_emotion(test_image, return_probabilities=True)
        
        print(f"Sample {i+1}:")
        print(f"  True: {true_label}")
        print(f"  Predicted: {prediction['emotion']} (confidence: {prediction['confidence']:.3f})")
        print()


if __name__ == "__main__":
    # Example usage
    print("=== Model Utils Example Usage ===")
    
    # Initialize model manager
    manager = ModelManager()
    available_models = manager.list_models()
    
    print(f"Available models: {available_models}")
    
    if available_models:
        # Load best model
        try:
            predictor = manager.get_best_model()
            print("Model loaded successfully!")
            
            # Show model info
            info = predictor.get_model_info()
            print("\nModel Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # Create sample predictions if test data exists
            try:
                create_sample_predictions(manager.get_model_path(available_models[0]))
            except FileNotFoundError:
                print("Test data not found, skipping sample predictions")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("No trained models found. Please train a model first using train_model.py")