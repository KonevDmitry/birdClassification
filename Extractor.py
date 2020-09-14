import cv2
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import os
import plotly.express as px

DATA_PATH = f'{os.getcwd()}/data'

TRAIN_PATH = f'{DATA_PATH}/train'
TEST_PATH = f'{DATA_PATH}/test'
TYPE_PATH = "{}/{}"

NUM_ELEMENTS_TO_SHOW = 2000


def get_data_and_labels():
    types = os.listdir(TRAIN_PATH)

    train, test, train_images, test_images = [], [], [], []
    for tp in types:
        train_type_path = TYPE_PATH.format(TRAIN_PATH, tp)
        for img in os.listdir(train_type_path):
            image = cv2.imread(TYPE_PATH.format(train_type_path, img), cv2.IMREAD_COLOR)
            train_images.append(image)
            train.append([tp, image.ravel()])

        test_type_path = TYPE_PATH.format(TEST_PATH, tp)
        for img in os.listdir(test_type_path):
            image = cv2.imread(TYPE_PATH.format(test_type_path, img), cv2.IMREAD_COLOR)
            test_images.append(image)
            test.append([tp, image.ravel()])

    train_labels = [i[0] for i in train]
    train_data = [i[1] for i in train]

    test_labels = [i[0] for i in test]
    test_data = [i[1] for i in test]
    return train_labels, train_data, test_labels, test_data, train_images, test_images


def embeddings(train_data):
    X = np.vstack(train_data[:NUM_ELEMENTS_TO_SHOW])
    return TSNE(n_components=3, n_jobs=1000).fit_transform(X)


def plot2d(tsne, train_labels, num):
    palette = sb.color_palette("bright", np.unique(train_labels[:num]).shape[0])
    plot = sb.scatterplot(tsne[:, 0], tsne[:, 1], hue=train_labels[:num], legend='full', palette=palette).set_title(
        "Distribution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plot3d(tsne, train_labels, num):
    fig = px.scatter_3d(x=tsne[::, 0],
                        y=tsne[::, 1],
                        z=tsne[::, 2],
                        color=[str(label) for label in train_labels[:num]],
                        opacity=0.7)
    fig.show()


def take_one_each_class():
    types = os.listdir(TRAIN_PATH)
    data, imgs = [], []
    for tp in types:
        train_type_path = TYPE_PATH.format(TRAIN_PATH, tp)
        image = cv2.imread(TYPE_PATH.format(train_type_path, os.listdir(train_type_path)[0]), cv2.IMREAD_COLOR)
        imgs.append(image)
        data.append([tp, image.ravel()])
    return [i[0] for i in data], [i[1] for i in data], imgs

def show_examples(imgs):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)  # the number of images in the grid is 5*5 (25)
        plt.imshow(imgs[i])
    plt.show()


# train_labels, train_data, test_labels, test_data, train_images, test_images = get_data_and_labels()
# print(len(test_data), len(test_labels), len(train_data), len(train_labels))
# tsne = embeddings(train_data)
# plot2d(tsne, train_labels, NUM_ELEMENTS_TO_SHOW)
# plot3d(tsne, train_labels, NUM_ELEMENTS_TO_SHOW)


show_train_labels, show_train_data, imgs = take_one_each_class()
print(len(show_train_labels), len(show_train_data))
tsne_show = embeddings(show_train_data)
plot2d(tsne_show, show_train_labels, len(show_train_labels))
plot3d(tsne_show, show_train_labels, NUM_ELEMENTS_TO_SHOW)
# show_examples(imgs)
