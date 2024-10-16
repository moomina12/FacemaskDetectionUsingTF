Face mask detection app using TensorFlow involves multiple steps, including preparing data, building a model, training it, and deploying it for real-time detection using a webcam or uploaded images. 
I have used publicle available datasets for facemask detection, which contain images of people with and without masks.
USed pre-trained model like MobileNetV2,which is a popular model for image classification tasks; and fine-tune it for face mask detection. This transfer learning approach leverages a model that has already been trained on a large dataset (such as ImageNet) and adapts it for specific task
In data preprocessing stage, we have used Data Augmentation, which elps in improving the generalization of the model by applying transformations such as rotation, zoom, and flipping to the training images.
MobileNetV2 significantly reduces the training time by using pre-trained weights and only training the final layers.
