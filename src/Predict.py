from Model import Classifier
from flask import Flask
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import time
import numpy as np
import os
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from flask import request

app = Flask(__name__)

data_dimension = 32

@app.route('/predict')
def predict():
    classifier = Classifier( number_of_classes=8 )
    classifier.load_model( 'models/model.h5' )
    input_X, images = classifier.prepare_images_from_dir( 'random_images/' )
    input_X = input_X.reshape( ( input_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )
    result_arr = classifier.predict(input_X).argmax(1)
    return str(result_arr[0])

@app.route('/train/<classes>')
def train(classes):
    dir_path = 'natural_images/'
    output_path = 'processed_data/'

    sub_dir_list = os.listdir( dir_path )
    images = list()
    labels = list()
    for i in range( len( sub_dir_list ) ):
        label = i
        image_names = os.listdir( dir_path + sub_dir_list[i] )
        for image_path in image_names:
            path = dir_path + sub_dir_list[i] + "/" + image_path
            image = Image.open( path ).convert( 'L' )
            resize_image = image.resize((data_dimension, data_dimension))
            array = list()
            for x in range(data_dimension):
                sub_array = list()
                for y in range(data_dimension):
                    sub_array.append(resize_image.load()[x, y])
                array.append(sub_array)
            image_data = np.array(array)
            image = np.array(np.reshape(image_data, (data_dimension, data_dimension, 1))) / 255
            images.append(image)
            labels.append( label )
        print (str(label) + " : " + sub_dir_list[i])

    x = np.array( images )
    y = np.array( keras.utils.to_categorical( np.array( labels) , num_classes=len(sub_dir_list) ) )

    train_features , test_features ,train_labels, test_labels = train_test_split( x , y , test_size=0.4 )

    np.save( '{}x.npy'.format( output_path )  , train_features )
    np.save( '{}y.npy'.format( output_path )  , train_labels )
    np.save( '{}test_x.npy'.format( output_path ) , test_features )
    np.save( '{}test_y.npy'.format( output_path ) , test_labels )
    
    X = np.load( 'processed_data/x.npy'.format( data_dimension ))
    Y = np.load( 'processed_data/y.npy'.format( data_dimension ))
    test_X = np.load( 'processed_data/test_x.npy'.format( data_dimension ))
    test_Y = np.load( 'processed_data/test_y.npy'.format( data_dimension ))

    print( X.shape )
    print( Y.shape )
    print( test_X.shape )
    print( test_Y.shape )

    X = X.reshape( ( X.shape[0] , data_dimension**2  ) ).astype( np.float32 )
    test_X = test_X.reshape( ( test_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )
    classes = int(request.view_args['classes'])
    classifier = Classifier( number_of_classes=classes )
    classifier.save_model( 'models/model.h5')

    parameters = {
        'batch_size' : 250 ,
        'epochs' : 10 ,
        'callbacks' : None ,
        'val_data' : None
    }

    classifier.fit( X , Y  , hyperparameters=parameters )
    classifier.save_model( 'models/model.h5')

    loss , accuracy = classifier.evaluate( test_X , test_Y )
    print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )

    sample_X, images = classifier.prepare_images_from_dir( 'random_images/' )
    sample_X = sample_X.reshape( ( sample_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )
    print( classifier.predict( sample_X ).argmax( 1 ) )
    return "Training is complete"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
