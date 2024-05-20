import tensorflow as tf
import tensorflow.keras.datasets as datasets 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import *
import numpy as np
import datetime
import PIL
from tempfile import TemporaryDirectory
dataset = datasets.cifar10
import time
import argparse 
import os 
import json

# Global variables
IMG_SIZE = 224
data_dir = './data/caltech-101/101_ObjectCategories'


# 限制程序运行的GPU编号
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# 用来限制显存的分配
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置每个GPU的显存动态增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 显存增长设置必须在程序初始阶段设置
        print(e)

# 是否启用混合精度
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 用来计时的
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)

# 所有的label名字
class_names = ['accordion', 'airplanes', 'anchor', 'ant', 'barrel']
num_classes = len(class_names)

# 调取数据
def load_custom_data(file_name):

    # 图像和标签列表
    images = []
    labels = []

    # 遍历每个类别文件夹
    class_dirs = [d for d in os.listdir(file_name) if os.path.isdir(os.path.join(file_name, d))]
    for i, class_dir in enumerate(sorted(class_dirs)):
        class_path = os.path.join(file_name, class_dir)
        for filename in os.listdir(class_path):
            if filename.endswith('.jpg'):
                img = tf.keras.preprocessing.image.load_img(os.path.join(class_path, filename), target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(i)
    
    #print(labels)
    return np.array(images), np.array(labels)


def getArgs():
    parser = argparse.ArgumentParser()

    # 指定命令参数
    parser.add_argument('--model', type=str, default="ResNet50")
    parser.add_argument('--classification_dataset_root', type=str, default=data_dir)
    parser.add_argument('--json_root', type=str, default="results.json")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    return args

# 用numpy重新实现train_test_split()
def my_train_test_split(images,labels,test_size, random_state):
    np.random.seed(random_state) # Use a constant seed to yield comparable results when running different models

    arr_indice = np.array(range(len(images)))
    np.random.shuffle(arr_indice)
    random_images = []
    random_labels = []

    for i in range(len(images)):
        random_images.append(images[arr_indice[i]])
        random_labels.append(labels[arr_indice[i]])
    
    train_images = random_images[int(test_size * len(images)):]
    train_labels = random_labels[int(test_size * len(images)):]
    test_images = random_images[:int(test_size * len(images))]
    test_labels = random_labels[:int(test_size * len(images))]

    return np.array(train_images),np.array(test_images),np.array(train_labels),np.array(test_labels)

def train(args):
    # (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    # 划分数据集为训练集和测试集
    images, labels = load_custom_data(args.classification_dataset_root)
    train_images, test_images, train_labels, test_labels = my_train_test_split(images, labels, test_size=0.2, random_state=40)
    
    print(f"train_image.shape = {train_images.shape}, || train_image.dataType = {train_images.dtype}, || train_labels.shape = {train_labels.shape}, || train_label.dataType = {train_labels.dtype}")
    print(f"test_image.shape = {test_images.shape}, || test_label.shape = {test_labels.shape}")

    # 数据预处理
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    print(f"num of train_images: {len(train_images)}")
    print(f"num of test_images: {len(test_images)}")
    # train_images = train_images / 255
    # test_images = test_images/ 255
    # to_categorical is analogous to pytorch's one-hot encoding
    train_labels = to_categorical(train_labels, num_classes) # need to match the number of classes!
    test_labels = to_categorical(test_labels, num_classes)

    # Assign the specific training model
    # If use model with pretrained imagenet weights:
    model = eval(args.model)(include_top=False, pooling='max', weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # If use model without pretrained weights:
    #model = eval(args.model)(weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    model.summary()
    model = tf.keras.models.Sequential(
        [model]
    )
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # Add final Dense prediction layer
    
    # Specify training configuration (optimizer, loss, metrics)
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    loss = tf.keras.losses.categorical_crossentropy
    metrics=['accuracy']

    # 简化一下，对于每个epoch，只训练一个batch
    # tf.profiler.experimental.start('logs')
    # 编译模型, necessary for training the model!
    model.compile(optimizer, loss, metrics)
    time_callback = TimeHistory()

    # all_model_checkpoint_callback 用来存储所有epochs中训练完的model state
    all_checkpoints_filepath = 'train-history/model_epoch{epoch:02d}.keras'
    best_checkpoint_filepath = 'train-history/best_model_epoch.keras'
    all_model_checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=all_checkpoints_filepath,
        save_freq="epoch",
        verbose=1,
        save_best_only=False)
    
    # best_model_checkpoint_callback 只存储best model state
    best_model_checkpoint_callback_all = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_filepath,
        monitor='accuracy',
        verbose=1,
        save_best_only=True)
    
    # Train the model by calling fit(), for a given # of epochs
    history = model.fit(train_images, train_labels, batch_size=args.batchsize, epochs=args.epochs, validation_data=(test_images, test_labels), callbacks=[time_callback, all_model_checkpoints_callback, best_model_checkpoint_callback_all])
    # tf.profiler.experimental.stop()

    return history, time_callback


def get_json(args, history, time_callback, json_file):
    print(history.history)
    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']
    validation_loss = history.history['val_loss']
    validation_accuracy = history.history['val_accuracy']
    for epoch, time_taken in enumerate(time_callback.times):
        print(f"Epoch {epoch + 1} took {time_taken:.2f} seconds")

        print(f"Epoch {epoch + 1}, "
            f"Training loss: {training_loss[epoch]:.4f}, "
            f"Training accuracy: {training_accuracy[epoch]:.4f}, "
            f"Validation loss: {validation_loss[epoch]:.4f}, "
            f"Validation accuracy: {validation_accuracy[epoch]:.4f}")
    
    with open(json_file,'r') as f:
        res_dict = json.load(f)
    
    res_dict["model"] = args.model
    res_dict[args.model] = {}
    total_training_loss = 0
    total_validation_loss = 0
    total_time = 0


    for epoch, time_taken in enumerate(time_callback.times):
        res_dict[args.model][f"epoch{epoch+1}"] = {
            "training_loss":training_loss[epoch], 
            "validation_loss":validation_loss[epoch], 
            "time(s)":time_taken
            }
        total_training_loss += training_loss[epoch]
        total_validation_loss += validation_loss[epoch]
        total_time += time_taken

    res_dict[args.model]["average"] = {
        "training_loss":total_training_loss/args.epochs,
        "validation_loss":total_validation_loss/args.epochs,
        "time(s)":total_time/args.epochs,
        "finished_time":str(datetime.datetime.now())
    }

    total_training_loss = 0
    total_validation_loss = 0
    total_time = 0

    with open(json_file,'w') as f:
        json.dump(res_dict,f,indent=4)
        print("已完成json文件写入，",json_file)
    


if __name__ == '__main__':

    # 初始化超参数
    args = getArgs()
    os.environ['MODEL_NAME'] = args.model

    # 训练模型
    history, time_callback = train(args)
    
    # 评估模型并记录为json文件
    get_json(args, history, time_callback, args.json_root)


