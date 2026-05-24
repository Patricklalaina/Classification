# =========================================================
# LOGISTIC REGRESSION FROM SCRATCH
# IMAGE CLASSIFICATION FROM PNG IMAGES
#
# 1. Binary Classification :
#    cat vs dog
#
# 2. Multiclass Classification :
#    One-vs-All
#
# NO SCIKIT-LEARN
# NO CIFAR10 API
# LOAD PNG IMAGES DIRECTLY
# =========================================================

# =========================================================
# DATASET STRUCTURE
# =========================================================

"""
dataset/

    train/
        cat/
        dog/
        bird/
        horse/

    test/
        cat/
        dog/
        bird/
        horse/

Each folder contains PNG images.
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 64

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

# =========================================================
# CLASSES
# =========================================================

classes = [
    "cat",
    "dog",
    "bird",
    "horse"
]

# =========================================================
# IMAGE LOADING
# =========================================================

def load_images(dataset_path):

    images = []
    labels = []

    for class_id, class_name in enumerate(classes):

        class_folder = os.path.join(
            dataset_path,
            class_name
        )

        for filename in os.listdir(class_folder):

            if filename.endswith(".png"):

                image_path = os.path.join(
                    class_folder,
                    filename
                )

                # load image
                image = Image.open(image_path)

                # RGB
                image = image.convert("RGB")

                # resize
                image = image.resize(
                    (IMG_SIZE, IMG_SIZE)
                )

                # numpy
                image = np.array(image)

                # normalize
                image = image / 255.0

                images.append(image)
                labels.append(class_id)

    return np.array(images), np.array(labels)

# =========================================================
# LOAD DATASET
# =========================================================

print("Loading training images...")
X_train, y_train = load_images(TRAIN_PATH)

print("Loading test images...")
X_test, y_test = load_images(TEST_PATH)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(10,6))

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(X_train[i])

    plt.title(classes[y_train[i]])

    plt.axis("off")

plt.suptitle("Dataset Samples")
plt.show()

# =========================================================
# IMAGE EMBEDDING
# =========================================================

"""
Here:
embedding = flatten image into vector.

64x64x3 -> 12288 dimensions
"""

X_train = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test = X_test.reshape(
    X_test.shape[0],
    -1
)

print("\nEmbedded image shape:")
print(X_train.shape)

# =========================================================
# SIGMOID
# =========================================================

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

# =========================================================
# LOSS
# =========================================================

def binary_cross_entropy(y, y_hat):

    epsilon = 1e-15

    y_hat = np.clip(
        y_hat,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y * np.log(y_hat)
        +
        (1-y) * np.log(1-y_hat)
    )

    return loss

# =========================================================
# ACCURACY
# =========================================================

def accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred)

# =========================================================
# LOGISTIC REGRESSION FROM SCRATCH
# =========================================================

class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=100
    ):

        self.lr = learning_rate
        self.epochs = epochs

    # =====================================================
    # TRAIN
    # =====================================================

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):

            # linear equation
            z = np.dot(X, self.w) + self.b

            # sigmoid
            y_hat = sigmoid(z)

            # gradients
            dw = (
                1 / n_samples
            ) * np.dot(
                X.T,
                (y_hat - y)
            )

            db = (
                1 / n_samples
            ) * np.sum(
                y_hat - y
            )

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # monitoring
            if epoch % 10 == 0:

                loss = binary_cross_entropy(
                    y,
                    y_hat
                )

                print(
                    f"Epoch {epoch}"
                    f" | Loss: {loss:.4f}"
                )

    # =====================================================
    # PROBA
    # =====================================================

    def predict_proba(self, X):

        z = np.dot(X, self.w) + self.b

        return sigmoid(z)

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (
            probabilities >= 0.5
        ).astype(int)

# =========================================================
# PART 1 : BINARY
# CAT VS DOG
# =========================================================

print("\n================================")
print("BINARY CLASSIFICATION")
print("CAT VS DOG")
print("================================")

# keep cat + dog only

binary_train_mask = np.where(
    (y_train == 0) | (y_train == 1)
)[0]

binary_test_mask = np.where(
    (y_test == 0) | (y_test == 1)
)[0]

X_train_binary = X_train[binary_train_mask]
y_train_binary = y_train[binary_train_mask]

X_test_binary = X_test[binary_test_mask]
y_test_binary = y_test[binary_test_mask]

# cat = 1
# dog = 0

y_train_binary = (
    y_train_binary == 0
).astype(int)

y_test_binary = (
    y_test_binary == 0
).astype(int)

# =========================================================
# TRAIN
# =========================================================

binary_model = LogisticRegressionScratch(
    learning_rate=0.1,
    epochs=100
)

binary_model.fit(
    X_train_binary,
    y_train_binary
)

# =========================================================
# TEST
# =========================================================

y_pred_binary = binary_model.predict(
    X_test_binary
)

acc_binary = accuracy(
    y_test_binary,
    y_pred_binary
)

print(
    f"\nBinary Accuracy: "
    f"{acc_binary:.4f}"
)

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(12,8))

for i in range(12):

    plt.subplot(3,4,i+1)

    img = X_test_binary[i].reshape(
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    pred = binary_model.predict(
        X_test_binary[i].reshape(1,-1)
    )[0]

    true = y_test_binary[i]

    pred_name = (
        "cat" if pred == 1 else "dog"
    )

    true_name = (
        "cat" if true == 1 else "dog"
    )

    plt.imshow(img)

    plt.title(
        f"Pred: {pred_name}\n"
        f"True: {true_name}"
    )

    plt.axis("off")

plt.suptitle(
    "Binary Classification"
)

plt.show()

# =========================================================
# PART 2 : MULTICLASS
# ONE VS ALL
# =========================================================

print("\n================================")
print("MULTICLASS")
print("ONE VS ALL")
print("================================")

class OneVsAll:

    def __init__(
        self,
        n_classes,
        learning_rate=0.01,
        epochs=100
    ):

        self.n_classes = n_classes

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.models = []

    # =====================================================
    # TRAIN
    # =====================================================

    def fit(self, X, y):

        for class_id in range(
            self.n_classes
        ):

            print(
                f"\nTraining class "
                f"{classes[class_id]}"
            )

            # current class = 1
            # others = 0

            y_binary = (
                y == class_id
            ).astype(int)

            model = LogisticRegressionScratch(
                learning_rate=self.learning_rate,
                epochs=self.epochs
            )

            model.fit(X, y_binary)

            self.models.append(model)

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X):

        scores = []

        for model in self.models:

            probs = model.predict_proba(X)

            scores.append(probs)

        scores = np.array(scores)

        predictions = np.argmax(
            scores,
            axis=0
        )

        return predictions

# =========================================================
# TRAIN MULTICLASS
# =========================================================

multiclass_model = OneVsAll(
    n_classes=len(classes),
    learning_rate=0.1,
    epochs=100
)

multiclass_model.fit(
    X_train,
    y_train
)

# =========================================================
# TEST
# =========================================================

y_pred_multi = multiclass_model.predict(
    X_test
)

acc_multi = accuracy(
    y_test,
    y_pred_multi
)

print(
    f"\nMulticlass Accuracy: "
    f"{acc_multi:.4f}"
)

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(15,10))

for i in range(15):

    plt.subplot(3,5,i+1)

    img = X_test[i].reshape(
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    pred = multiclass_model.predict(
        X_test[i].reshape(1,-1)
    )[0]

    true = y_test[i]

    plt.imshow(img)

    plt.title(
        f"Pred: {classes[pred]}\n"
        f"True: {classes[true]}",
        fontsize=9
    )

    plt.axis("off")

plt.suptitle(
    "Multiclass Classification"
)

plt.show()

# =========================================================
# ONE VS ALL SCORES
# =========================================================

sample = X_test[0].reshape(1,-1)

print("\n================================")
print("ONE VS ALL SCORES")
print("================================")

for i, model in enumerate(
    multiclass_model.models
):

    score = model.predict_proba(
        sample
    )[0]

    print(
        f"{classes[i]} : "
        f"{score:.4f}"
    )

prediction = multiclass_model.predict(
    sample
)[0]

print(
    f"\nPredicted : "
    f"{classes[prediction]}"
)

print(
    f"True : "
    f"{classes[y_test[0]]}"
)

# =========================================================
# CONCLUSION
# =========================================================

print("\n================================")
print("CONCLUSION")
print("================================")

print("""
1. Images are loaded from PNG files.

2. Images are transformed into vectors
   (embedding by flattening).

3. Logistic Regression uses:
   sigmoid activation.

4. Binary classification:
   cat vs dog.

5. Multiclass classification:
   One-vs-All strategy.

6. One classifier is trained
   per class.

7. Logistic Regression is simple
   but limited for complex vision tasks.

8. CNNs are generally much better
   for image classification.
""")
