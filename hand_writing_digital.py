import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import datasets
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

def show_digit(data):
    pixels = np.array(data, dtype='uint8')
    pixels = pixels.reshape(28, 28)
    plt.imshow(pixels, cmap=plt.cm.gray)
    plt.show()

def show_top_100(data):
    fig, axes = plt.subplots(10, 10)
    i = 0
    for ax in axes.ravel():
        pixels = np.array(data[i], dtype='uint8')
        pixels = pixels.reshape(28, 28)
        ax.matshow(pixels, cmap=plt.cm.gray)
        i += 1
    plt.show()

def read_image_as_arr(file_name):
    img = Image.open(file_name)
    gray_img = img.convert("L")
    im_array = np.array(gray_img)
    im_array = im_array.ravel()
    return im_array

MAX_TRAIN_SIZE = 31000
digits = datasets.load_digits()

mnist = datasets.fetch_mldata('MNIST original', data_home='datahome')

clf = svm.SVC(gamma=0.0001, C=100)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

X, Y = shuffle(mnist.data, mnist.target)

x, y = X / 255., Y

mlp.fit(x[100:MAX_TRAIN_SIZE], y[100:MAX_TRAIN_SIZE])

show_top_100(X[:100])

while True:
    test_image = input()
    test_data = read_image_as_arr(test_image)
    predication = mlp.predict([test_data])
    print(predication[0])