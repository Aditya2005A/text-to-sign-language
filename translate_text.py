# translate_text.py
import os
import cv2
import matplotlib.pyplot as plt

def translate_to_sign(word, dataset_path="archive/train/asl_alphabet_train", img_size=(128, 128)):
    word = word.upper()
    images = []

    for char in word:
        char_path = os.path.join(dataset_path, char)
        if not os.path.exists(char_path):
            print(f"Character '{char}' not found in dataset.")
            continue

        # Pick the first image of the class
        img_file = os.listdir(char_path)[0]
        img_path = os.path.join(char_path, img_file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append((char, img))

    # Plot
    fig, axs = plt.subplots(1, len(images), figsize=(2 * len(images), 3))
    for i, (char, img) in enumerate(images):
        axs[i].imshow(img)
        axs[i].set_title(char)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    translate_to_sign("HI")