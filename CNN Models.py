#!/usr/bin/env python
# coding: utf-8

# # Xception Model

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

# point to your augmented data folder
data_dir = "/content/drive/MyDrive/Coloncancer_Augmented"

# --- Data Generator ---
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(299, 299),  # Xception input size
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --- Model Definition ---
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# --- Metrics Evaluation ---
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, roc_auc_score, confusion_matrix
)

val_gen.reset()
y_true = val_gen.classes
y_pred_proba = model.predict(val_gen)
y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy    = accuracy_score(y_true, y_pred)
recall      = recall_score(y_true, y_pred)
precision   = precision_score(y_true, y_pred)
f1          = f1_score(y_true, y_pred)
specificity = tn / (tn + fp)
npv         = tn / (tn + fn)
error_rate  = 1 - accuracy
fpr         = fp / (fp + tn)
fnr         = fn / (fn + tp)
auc         = roc_auc_score(y_true, y_pred_proba)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"PPV:                {precision:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"NPV:                {npv:.4f}")
print(f"AUC:                {auc:.4f}")
print(f"Error Rate:         {error_rate:.4f}")
print(f"FPR:                {fpr:.4f}")
print(f"FNR:                {fnr:.4f}")


# # MobileNetV2 Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# point to your augmented data folder
data_dir = "/content/drive/MyDrive/Coloncancer_Augmented"

# --- Data Generator with Augmentation ---
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --- Model Definition ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# --- Metrics Evaluation ---
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, roc_auc_score, confusion_matrix
)

val_gen.reset()
y_true = val_gen.classes
y_pred_proba = model.predict(val_gen)
y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy    = accuracy_score(y_true, y_pred)
recall      = recall_score(y_true, y_pred)
precision   = precision_score(y_true, y_pred)
f1          = f1_score(y_true, y_pred)
specificity = tn / (tn + fp)
npv         = tn / (tn + fn)
error_rate  = 1 - accuracy
fpr         = fp / (fp + tn)
fnr         = fn / (fn + tp)
auc         = roc_auc_score(y_true, y_pred_proba)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"PPV:                {precision:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"NPV:                {npv:.4f}")
print(f"AUC:                {auc:.4f}")
print(f"Error Rate:         {error_rate:.4f}")
print(f"FPR:                {fpr:.4f}")
print(f"FNR:                {fnr:.4f}")


# # ResNet50V2 Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, models

# point to your augmented data folder
data_dir = "/content/drive/MyDrive/Coloncancer_Augmented"

# --- Data Generator ---
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --- Model Definition ---
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# --- Metrics Evaluation ---
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, roc_auc_score, confusion_matrix
)

val_gen.reset()
y_true = val_gen.classes
y_pred_proba = model.predict(val_gen)
y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy    = accuracy_score(y_true, y_pred)
recall      = recall_score(y_true, y_pred)
precision   = precision_score(y_true, y_pred)
f1          = f1_score(y_true, y_pred)
specificity = tn / (tn + fp)
npv         = tn / (tn + fn)
error_rate  = 1 - accuracy
fpr         = fp / (fp + tn)
fnr         = fn / (fn + tp)
auc         = roc_auc_score(y_true, y_pred_proba)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"PPV:                {precision:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"NPV:                {npv:.4f}")
print(f"AUC:                {auc:.4f}")
print(f"Error Rate:         {error_rate:.4f}")
print(f"FPR:                {fpr:.4f}")
print(f"FNR:                {fnr:.4f}")


# # VGG16 Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# point to your augmented data folder
data_dir = "/content/drive/MyDrive/Coloncancer_Augmented"

# --- Data Generator ---
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False                    # <-- important for metrics
)

# --- Model Definition ---
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False        # freeze base layers

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# --- Metrics Evaluation ---
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_score, roc_auc_score, confusion_matrix
)

val_gen.reset()
y_true = val_gen.classes
y_pred_proba = model.predict(val_gen)
y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy    = accuracy_score(y_true, y_pred)
recall      = recall_score(y_true, y_pred)         # sensitivity
precision   = precision_score(y_true, y_pred)      # PPV
f1          = f1_score(y_true, y_pred)
specificity = tn / (tn + fp)
npv         = tn / (tn + fn)
error_rate  = 1 - accuracy
fpr         = fp / (fp + tn)
fnr         = fn / (fn + tp)
auc         = roc_auc_score(y_true, y_pred_proba)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"PPV:                {precision:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"NPV:                {npv:.4f}")
print(f"AUC:                {auc:.4f}")
print(f"Error Rate:         {error_rate:.4f}")
print(f"FPR:                {fpr:.4f}")
print(f"FNR:                {fnr:.4f}")

