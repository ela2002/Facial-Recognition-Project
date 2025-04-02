import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import hog
from skimage import color
from sklearn.cluster import KMeans

# -----------------------------
# 🔹 Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# -----------------------------
# 🔹 Helper Functions
# -----------------------------
def get_person_label(label):
    """ Convert numeric labels to 'Person X' format. """
    return f"Person {label}"


def visualize_images(images, labels, title="Images"):
    """ Display images with person labels. """
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(get_person_label(labels[i]))  # Convert label to 'Person X'
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


def visualize_predicted_images(original_images, y_test, predictions):
    """ Display images with actual and predicted labels. """
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(predictions))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(original_images[i], cmap="gray")
        plt.title(f"True: {get_person_label(y_test[i])}\nPred: {get_person_label(predictions[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 🔹 Step 1: Load and Preprocess Images
# -----------------------------
def load_images_from_folders(folders, img_size=(128, 128)):
    logging.info("Loading images from folders...")
    images, labels = [], []

    for folder_path in folders:
        for img_name in os.listdir(folder_path):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)  # Standardizing size
                    label = int(img_name.split("-")[0])  # Extract person ID
                    images.append(img)
                    labels.append(label)

    logging.info(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels)


# -----------------------------
# 🔹 Step 2: Feature Extraction (SIFT + HOG)
# -----------------------------
def extract_sift_features(images, k=128):
    logging.info("Extracting SIFT features using Bag of Words...")
    sift = cv2.SIFT_create()
    descriptors_list = []
    valid_indices = []
    keypoint_images = []  # Store images with keypoints for visualization

    for idx, img in enumerate(images):
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des)
            valid_indices.append(idx)

            # Draw keypoints
            img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            keypoint_images.append(img_kp)

    if len(descriptors_list) == 0:
        raise ValueError("No valid SIFT descriptors found!")

    # KMeans for Bag of Words
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    all_descriptors = np.vstack(descriptors_list)
    kmeans.fit(all_descriptors)

    features = []
    for des in descriptors_list:
        histogram = np.zeros(k)
        labels = kmeans.predict(des)
        for label in labels:
            histogram[label] += 1
        features.append(histogram)

    # Visualize SIFT keypoints
    visualize_images(keypoint_images[:9], valid_indices[:9], title="SIFT Keypoints")

    return np.array(features), np.array(valid_indices)


def extract_hog_features(images, valid_indices):
    logging.info("Extracting HOG features...")
    hog_features = []
    hog_images = []

    for idx in valid_indices:
        img = images[idx]
        fd, hog_image = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
        hog_features.append(fd)
        hog_images.append(hog_image)

    # Visualize HOG Images
    visualize_images(hog_images[:9], valid_indices[:9], title="HOG Features")

    return np.array(hog_features)


def combine_features(sift_features, hog_features):
    logging.info(f"Combining SIFT ({sift_features.shape[0]}) and HOG ({hog_features.shape[0]}) features...")
    min_size = min(len(sift_features), len(hog_features))
    return np.hstack((sift_features[:min_size], hog_features[:min_size]))


# -----------------------------
# 🔹 Step 3: PCA for Dimensionality Reduction
# -----------------------------
def apply_pca(features, variance_threshold=0.95):
    logging.info("Applying PCA...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=variance_threshold)
    pca_features = pca.fit_transform(features_scaled)

    # PCA Variance Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid()
    plt.show()

    logging.info(f"PCA reduced features to {pca.n_components_} dimensions.")
    return pca_features, pca, scaler

def plot_pca_2d(features, labels):
    logging.info("Visualizing the first two PCA components...")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features_scaled)

    unique_labels = np.unique(labels)
    person_labels = [f"Person {l}" for l in labels]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='tab10', alpha=0.75, edgecolors='k')

    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    legend_labels = [f"Person {l}" for l in unique_labels]
    plt.legend(handles, legend_labels, title="People", loc="upper right")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - First Two Components Visualization")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot a confusion matrix for a given model's predictions."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))

    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# -----------------------------
# 🔹 Step 4: Train & Evaluate Classifiers
# -----------------------------
def train_and_evaluate_classifiers(X_train, y_train, X_test, y_test, pca, scaler, images):
    logging.info("Training classifiers...")

    # SVM Classifier
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    # Combine Predictions
    combined_pred = mode([svm_pred, knn_pred], axis=0)[0].flatten()


    # Confusion Matrix
    plot_confusion_matrix(y_test, svm_pred, "SVM")
    plot_confusion_matrix(y_test, knn_pred, "KNN")
    plot_confusion_matrix(y_test, combined_pred, "Combined (SVM + KNN)")

    # Compute and log accuracies
    svm_acc = accuracy_score(y_test, svm_pred)
    knn_acc = accuracy_score(y_test, knn_pred)
    combined_acc = accuracy_score(y_test, combined_pred)

    logging.info(f"SVM Accuracy: {svm_acc * 100:.2f}%")
    logging.info(f"KNN Accuracy: {knn_acc * 100:.2f}%")
    logging.info(f"Combined Model Accuracy: {combined_acc * 100:.2f}%")

    # Classification Report
    logging.info("Classification Report:")
    print(classification_report(y_test, combined_pred))

    # Visualize Predictions
    visualize_predicted_images(images, y_test, combined_pred)


# -----------------------------
# 🔹 Step 5: Main Pipeline
# -----------------------------
def main():
    logging.info("Starting the facial recognition process...")

    # Load dataset
    folder_paths = ["./FEI_dataset_part1","./FEI_dataset_part2","./FEI_dataset_part3","./FEI_dataset_part4",]
    images, labels = load_images_from_folders(folder_paths)

    visualize_images(images, labels, title="Loaded Images")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)

    # Extract Features
    sift_train, valid_train = extract_sift_features(X_train)
    sift_test, valid_test = extract_sift_features(X_test)

    hog_train = extract_hog_features(X_train, valid_train)
    hog_test = extract_hog_features(X_test, valid_test)

    y_train_filtered = y_train[valid_train]
    y_test_filtered = y_test[valid_test]

    # Combine Features
    train_features = combine_features(sift_train, hog_train)
    test_features = combine_features(sift_test, hog_test)

    # Apply PCA
    X_train_pca, pca, scaler = apply_pca(train_features)
    X_test_pca = pca.transform(scaler.transform(test_features))

    plot_pca_2d(X_train_pca, y_train_filtered)

    # Train and Evaluate
    train_and_evaluate_classifiers(X_train_pca, y_train_filtered, X_test_pca, y_test_filtered, pca, scaler, X_test[valid_test])


if __name__ == "__main__":
    main()
