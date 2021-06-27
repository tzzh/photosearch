from matplotlib import pyplot
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input



vgg_face = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def detect_faces(image_path):
    """Returns list of face arrays for the given image"""
    pixels = pyplot.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    result = []
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)
        result.append(face_array)

    return result

def get_face_embeddings(faces):
    raw_samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(raw_samples, version=2)
    # create a vggface model
    # perform prediction
    yhat = vgg_face.predict(samples)
    return yhat
