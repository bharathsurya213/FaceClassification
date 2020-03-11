import matplotlib
from PIL import Image
from sklearn.cluster import KMeans
from numpy import load
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from numpy import load
from numpy import expand_dims

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainz = data['arr_0'], data['arr_1']

kmeans = KMeans(n_clusters=10)
kmeans.fit(trainX)
trainY = kmeans.labels_
d={}

for i in range(len(trainY)):
    try:
        d[trainY[i]].append(trainz[i])
    except:
        d[trainY[i]] = [trainz[i]]



# print(kmeans.cluster_centers_)
def get_face_embedder(path,required_size=(160, 160)):
    image = Image.open(path)
    image = image.convert('RGB')
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
embedding = get_embedding(model, get_face_embedder("jasonMomoa.jpg"))

ans = kmeans.predict([embedding])
print(ans)
print(d[ans[0]])

for i in set(d[ans[0]]):
    matplotlib.pyplot.imshow(matplotlib.image.imread('train/'+i))
    matplotlib.pyplot.show()


