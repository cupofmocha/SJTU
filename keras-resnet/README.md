An implementation of pretrained imagenet ResNet 50 with Keras, TensorFlow. (Can be applied to other models as well)
Be Aware:
If your GPU is not powerful enough (mainly memory), then the prediction results will not be very accurate, despite the training accuracy can be very high (>90%). This is because Keras will simplify the computations (a warning message will appear in the terminal) during training, if there is insufficient gpu computation power.