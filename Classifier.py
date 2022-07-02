from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, random_state=9, train_size=3500, test_size=500)

xtrainScaled = xtrain/255.00
xtestScaled = xtest/255.00

clf = LogisticRegression(
    solver="saga", multi_class="multinomial").fit(xtrainScaled, ytrain)


def get_pred(image):
    im_pil = Image.open(image)
    image_pw = im_pil.convert("L")
    image_pw_resized = image_pw.resize((28, 28), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(image_pw_resized, pixel_filter)
    image_pw_resized_inverted_scaled = np.clip(
        image_pw_resized - min_pixel, 0, 255)
    max_pixel = np.max(image_pw_resized)
    image_pw_resized_inverted_scaled = np.asarray(
        image_pw_resized_inverted_scaled)/max_pixel
    test_sample = np.asarray(image_pw_resized_inverted_scaled).reshape(1, 784)

    test_pred = clf.predict(test_sample)
    return test_pred[0]
