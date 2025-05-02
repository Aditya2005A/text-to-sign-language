# app.py
from flask import Flask, render_template, request
import os
import random

app = Flask(__name__)

DATASET_DIR = "archive/train/asl_alphabet_train"

@app.route('/', methods=['GET', 'POST'])
def index():
    images = []
    if request.method == 'POST':
        text = request.form['text'].upper()
        for char in text:
            if char.isalpha():
                char_dir = os.path.join(DATASET_DIR, char)
                if os.path.exists(char_dir):
                    # Pick a random image from the folder
                    img_file = random.choice(os.listdir(char_dir))
                    images.append(f"signs/{char}/{img_file}")
    return render_template('index.html', images=images)

if __name__ == "__main__":
    app.run(debug=True)