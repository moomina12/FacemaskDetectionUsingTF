Face mask detection app using TensorFlow involves multiple steps, including preparing data, building a model, training it, and deploying it for real-time detection using a webcam or uploaded images. 
I have used publicly available datasets for facemask detection, which contain images of people with and without masks.
USed pre-trained model like MobileNetV2,which is a popular model for image classification tasks; and fine-tune it for face mask detection.
In data preprocessing stage, we have used Data Augmentation, which helps in improving the generalization of the model by applying transformations such as rotation, zoom, and flipping to the training images.
MobileNetV2 significantly reduces the training time by using pre-trained weights and only training the final layers.
App deployment done thru streamlit
