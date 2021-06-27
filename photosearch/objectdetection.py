import numpy as np
import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.preprocessing import image


model = resnet50.ResNet50(weights='imagenet')


def get_object_names(image_path, threshold=0.1):
    """Return the names of all objects detected in the image above threshold"""

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)

    #TODO should be done in batch
    preds = model.predict(x)
    res = []
    for _, class_name, value in resnet50.decode_predictions(preds, top=10)[0]:
        if value > threshold:
            res.append(class_name)
    return res
