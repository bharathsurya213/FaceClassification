# function for face detection with mtcnn
from os import listdir

from PIL import Image
from numpy import asarray, savez_compressed
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(path,filename, required_size=(160, 160)):
	z = []
	# load image from file
	image = Image.open(path)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	faces_list = []
	# extract the bounding box from the first face
	for i in range(len(results)):
		z.append(filename)
		x1, y1, width, height = results[i]['box']
		# bug fix
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		faces_list.append(face_array)
	print(len(results))
	return faces_list,z


def load_faces(directory):
	faces_list = list()
	# enumerate files
	z=[]
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		faces,z1 = extract_face(path,filename)
		# store
		faces_list.extend(faces)
		z.extend(z1)
	return faces_list,z


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
		# path
	path = directory + '/'
		# load all faces in the subdirectory
	faces,z = load_faces(path)
	X.extend(faces)
	return asarray(X), asarray(z)


# load train dataset
trainX, trainz = load_dataset('train/')
# load test dataset
# save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainz)

