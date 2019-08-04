from Model import Classifier
import numpy as np

data_dimension = 32

classifier = Classifier( number_of_classes=8 )
classifier.load_model( 'models/model.h5' )
input_X, images = classifier.prepare_images_from_dir( 'random_images/' )
input_X = input_X.reshape( ( input_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )
result_arr = classifier.predict(input_X).argmax(1)
for i in range(len(images)):
    print(images[i] + " : " + str(result_arr[i]))