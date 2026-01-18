import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from datetime import datetime
import json


class EmotionCNNTrainer:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.emotion_labels = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        os.makedirs("model/saved_models", exist_ok=True)
        os.makedirs("model/logs", exist_ok=True)
        os.makedirs("model/plots", exist_ok=True)

    def compile_model(self, model: Model, learning_rate: float = 0.001):
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        )
        self.model = model
        print("Model compiled successfully!")
        print(f"Total parameters: {model.count_params():,}")

    def create_basic_cnn(self) -> Model:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def create_advanced_cnn(self) -> Model:
        input_layer = Input(shape=self.input_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        output = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        return model

    def load_data(self, data_path: str = "data/processed_fer2013.pkl"):
        print("Loading processed data...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return (data_dict['X_train'], data_dict['X_val'], data_dict['X_test'],
                data_dict['y_train'], data_dict['y_val'], data_dict['y_test'])
###
###
###

# compute class weights
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train_labels), y=y_train_labels
)
class_weights = dict(enumerate(class_weights))

callbacks = [
    ModelCheckpoint(f"model/saved_models/{model_name}_best.h5", save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5),
    CSVLogger(f"model/logs/{model_name}_training.log")
]
##################
def main():
    print("=== Emotion Detection Model Training ===")
    trainer = EmotionCNNTrainer()

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
        print(f"Data loaded successfully!\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        use_advanced = input("Use advanced CNN architecture? (y/n): ").strip().lower() == 'y'
        model = trainer.create_advanced_cnn() if use_advanced else trainer.create_basic_cnn()
        model_name = "emotion_advanced_cnn" if use_advanced else "emotion_basic_cnn"

        trainer.compile_model(model)
        model.summary()

        epochs = int(input("Enter number of epochs (default 50): ") or 50)
        batch_size = int(input("Enter batch size (default 32): ") or 32)
        use_aug = input("Use data augmentation? (y/n): ").strip().lower() == 'y'

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ) if use_aug else ImageDataGenerator()

        trainer.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        results = trainer.model.evaluate(X_test, y_test)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}, Top-2 Accuracy: {results[2]:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
