# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras_train import *

# Global variables
NUM_IMG = 20 # num of images to visualize

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    print(predictions_array)
    predicted_label = np.argmax(predictions_array)
    #print("predicted label is:", predicted_label)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("predicted: {} {:2.0f}%   ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == '__main__':
    # load the model we saved
    model = load_model('train-history/best_model_epoch.keras')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    #             loss=tf.keras.losses.categorical_crossentropy,
    #             metrics=['accuracy'])
    
    images, labels = images, labels = load_custom_data(data_dir)
    train_images, test_images, train_labels, test_labels = my_train_test_split(images, labels, test_size=0.2, random_state=40)
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    print(test_labels)
    # plt.figure(figsize=(10,10))
    # for i in range(2):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()
    
    # predicting images
    probability_model = tf.keras.Sequential([
                model, tf.keras.layers.Softmax()
                ])
    predictions = model.predict(test_images)
    print(predictions)

    for i in range(NUM_IMG):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)

        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(1,2,2)

        plot_value_array(i, predictions[i], test_labels)
        plt.show()