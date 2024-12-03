import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define the dataset path
dataset_path = r'D:/Deep_learning/image/Training'

# Get the list of categories
categories = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
data = []

# Collect data information
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for filename in os.listdir(category_path):
        data.append((category, filename))
        
# Create a DataFrame
df = pd.DataFrame(data, columns=['Category', 'Filename'])
print(df.head())
print(df['Category'].value_counts())

# Encode the labels
label_encoder = LabelEncoder()
df['Encoded_Category'] = label_encoder.fit_transform(df['Category'])

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Encoded_Category'])

print(train_df.shape, test_df.shape)

# Plot the distribution of categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Category')
plt.title('Distribution of Knee Arthritis Categories')
plt.show()

# Function to load and preprocess image
def load_and_preprocess_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize to a fixed size
    image = preprocess_input(image)  # Preprocess input for EfficientNetB0
    return image

# Data augmentation using SimCLR augmentations
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Prepare the data for training
def prepare_data(df, dataset_path):
    images = []
    labels = []
    for _, row in df.iterrows():
        filepath = os.path.join(dataset_path, row['Category'], row['Filename'])
        image = load_and_preprocess_image(filepath)
        images.append(image)
        labels.append(row['Encoded_Category'])
    return np.array(images), np.array(labels)

train_images, train_labels = prepare_data(train_df, dataset_path)
test_images, test_labels = prepare_data(test_df, dataset_path)

# Augment the training data
train_augmented_flow = datagen.flow(train_images, train_labels, batch_size=len(train_images), shuffle=False)
train_images_augmented, train_labels_augmented = next(train_augmented_flow)

# Concatenate the original and augmented data
train_images_augmented = np.concatenate([train_images, train_images_augmented])
train_labels_augmented = np.concatenate([train_labels, train_labels_augmented])

# Load the EfficientNetB0 model, excluding the top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze some layers of the base model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Contrastive Learning and GAN Integration Module
class ContrastiveGAN(Model):
    def __init__(self, feature_extractor):
        super(ContrastiveGAN, self).__init__()
        self.feature_extractor = feature_extractor
        self.flatten = Flatten()
        self.projection_head = Dense(128, activation='relu', name='projection_head')

    def call(self, inputs):
        features = self.feature_extractor(inputs)
        flattened_features = self.flatten(features)
        projection = self.projection_head(flattened_features)
        return projection  # discriminator_head 제거

# Instantiate ContrastiveGAN model and assign optimizer
contrastive_gan = ContrastiveGAN(base_model)
contrastive_gan.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Verify that the 'projection_head' layer is trainable
print("Is projection_head trainable?", contrastive_gan.projection_head.trainable)

# Generator model
generator_input = Input(shape=(100,))
x = Dense(14 * 14 * 256, activation='relu')(generator_input)
x = tf.reshape(x, (-1, 14, 14, 256))
x = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
generated_image = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', activation='tanh')(x)
generated_image = tf.image.resize(generated_image, [224, 224])
generator_model = Model(inputs=generator_input, outputs=generated_image)

# Discriminator model
input_image = Input(shape=(224, 224, 3))
x = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))(input_image)
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid')(x)
discriminator_model = Model(inputs=input_image, outputs=x)
discriminator_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN model combining generator and discriminator
discriminator_model.trainable = False
gan_input = Input(shape=(100,))
generated_image = generator_model(gan_input)
validity = discriminator_model(generated_image)
gan_model = Model(inputs=gan_input, outputs=validity)
gan_model.compile(optimizer='adam', loss='binary_crossentropy')

# Define the contrastive loss function
@tf.function
def contrastive_loss(y_true, y_pred, margin=1):
    anchor, positive, negative = tf.unstack(y_pred, axis=1)
    positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(0.0, positive_distance - negative_distance + margin)
    return tf.reduce_mean(loss) + 0.1 * (tf.reduce_mean(positive_distance) + tf.reduce_mean(negative_distance))

# Training loop with GradientTape and explicit weight updates
batch_size = 32
epochs = 100
initial_weights = contrastive_gan.projection_head.get_weights()

for epoch in range(epochs):
    # Train discriminator with real and fake data
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator_model.predict(noise)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator_model.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator_model.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator via GAN
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_y = np.ones((batch_size, 1))
    g_loss = gan_model.train_on_batch(noise, valid_y)

    # Train contrastive model with triplet samples
    with tf.GradientTape() as tape:
        anchor_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
        positive_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
        negative_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
        
        # Forward pass and get projections (single output)
        anchor_proj = contrastive_gan(anchor_images)
        positive_proj = contrastive_gan(positive_images)
        negative_proj = contrastive_gan(negative_images)
        
        stacked_projections = tf.stack([anchor_proj, positive_proj, negative_proj], axis=1)
        
        # Calculate contrastive loss
        loss = contrastive_loss(None, stacked_projections)

    # Compute gradients and apply them
    grads = tape.gradient(loss, contrastive_gan.trainable_variables)
    contrastive_gan.optimizer.apply_gradients(zip(grads, contrastive_gan.trainable_variables))

    # Check projection_head weights update
    current_weights = contrastive_gan.projection_head.get_weights()
    if np.array_equal(initial_weights, current_weights):
        print("Warning: projection_head weights have not changed.")
    else:
        print("projection_head weights updated.")
    initial_weights = current_weights

class EpochEndCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 테스트셋에 대한 평가
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        print(f"에포크 {epoch + 1} - 테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_accuracy:.4f}")

# Classification model의 훈련 및 평가
classification_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

early_stopping = EarlyStopping(
    monitor='val_loss',  # 검증 손실을 모니터링합니다.
    patience=10,         # 10번의 에포크 동안 개선되지 않으면 학습을 중지합니다.
    restore_best_weights=True,  # 가장 좋은 가중치를 복원합니다.
    verbose=1            # 중지 시 메시지를 출력합니다.
)

# Train and evaluate the classification model
history = classification_model.fit(
    train_images_augmented, 
    train_labels_augmented, 
    epochs=30, 
    validation_data=(test_images, test_labels), 
    callbacks=[lr_scheduler, EpochEndCallback()])
test_loss, test_acc = classification_model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Generate predictions and plot confusion matrix
predictions = classification_model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(test_labels, predicted_classes, target_names=label_encoder.classes_))

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm = confusion_matrix(test_labels, predicted_classes)
plot_confusion_matrix(cm, classes=label_encoder.classes_)
