import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import time
import math



models_dir = f"models/OTA-FL/1/{int(time.time())}/"
logdir = f"logs/OTA-FL/1/{int(time.time())}/"


if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

def load(paths, verbose=-1):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images

    # print("injaaaa",)
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        # print(label)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


    #declear path to your mnist data folder
img_path = 'D:\\Trans_1\\Federated Learning\\trainingSet\\'

#get the path list using the path object
image_paths = list(paths.list_images(img_path))
# print(image_paths)
#apply our function
image_list, label_list = load(image_paths, verbose=10000)
# print(label_list)
#binarize the labels
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                    label_list, 
                                                    test_size=0.1, 
                                                    random_state=42)
np.random.seed(42)
def generate_rayleigh_coefficient(num_coefficients, scale=1):
    # Generate complex Gaussian random numbers with zero mean and unit variance
    gaussian_real = np.round(np.random.normal(loc=0.7, scale=scale, size=num_coefficients), 4)
    gaussian_imag = np.round(np.random.normal(loc=0.7, scale=scale, size=num_coefficients), 4)
    
    # Form complex numbers
    complex_numbers = gaussian_real + 1j * gaussian_imag
    
    # Calculate the magnitude of complex numbers
    rayleigh_coefficients = np.abs(complex_numbers)
    
    return complex_numbers[0] 



def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)




    def generate_random_numbers(n):
    # Generate random numbers
     random_numbers = np.random.rand(n)
    
    # Normalize them so that their sum is 1
     random_numbers /= random_numbers.sum()
    
     return random_numbers

# Generate 10 random numbers
    sizeper = generate_random_numbers(num_clients)
    #shard data and place at each client
    size = len(data)//num_clients
    shards= []
    previous_size =0 
    for i in range(num_clients):
     stepsize =  math.floor(len(data) * sizeper[i] ) 

     print(previous_size, stepsize)  
     shards.append(data[previous_size:previous_size + stepsize])
     previous_size += stepsize

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : [shards[i],generate_rayleigh_coefficient(1, scale=1)] for i in range(len(client_names))} 



def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard[0])
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


def Tx_Scaling(coef, rx_scale_coef):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    b= 1/(rx_scale_coef * coef)
    return b

def Rx_Scaling(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard[0])
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

my_num_client =30
I_p = 1.5
comms_round = 100


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model



clients = create_clients(X_train, y_train, num_clients=my_num_client, initial='client')
#process and batch the training data for each client
clients_batched = dict()
clients_coef = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
    clients_coef[client_name] = data[1]
    # print(clients_coef[client_name]) 
    
#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))




lr = 0.01 
comms_round = 100
loss='categorical_crossentropy'
metrics = ['accuracy']
# optimizer = SGD(learning_rate=lr, 
#                 momentum=0.9
#                )  
optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0)

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar,coef):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    # print(steps)
    beta= Tx_Scaling(coef, 1)
    for i in range(steps):
        weight_final.append(beta * coef * scalar * weight[i])
        # print(" weight[i]", len(weight[i]))
    return weight_final



def sum_scaled_weights(scaled_weight_list,noise):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    col_noise = []
    i=0
    #get the average grad accross all client gradients
    # noise = np.random.normal(0,1)
    for grad_list_tuple in zip(*scaled_weight_list):
        # print("###########################################################################################",i)
        # i = i +1
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        # o=layer_mean + noise
        # b= layer_mean - o
        # print("layer_mean",b )
        # print()
        #  nrow=784
        #  ncol=200

        noise = np.random.normal(loc=0, scale=I_p, size=layer_mean.shape) 
        avg_grad.append(layer_mean + noise / my_num_client)
        b=avg_grad
        # print(b)
        # i = i+1
    
    # print("===========",i)
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss   




#initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(784, 10)
        
#commence global training loop
for comm_round in range(comms_round):
            
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)
    
    #loop through each client and create new local model
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)

        coef= clients_coef[client]
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        local_weights=local_model.get_weights()
        gradients = [(local_w - global_w) / 0.1 for local_w, global_w in zip(local_weights, global_weights)]
        # gradients = (local_model.get_weights() - global_weights) / 0.1


        # print("-------------------------------------------------------------------------------------------------")
        # print(client, "\n")
        # print( gradients)

        #scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor, coef)
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()

    


    # global_weights_arr = np.array(global_weights)

    nrow=784
    ncol=200

    noise = np.random.normal(loc=0, scale=1, size=(nrow,ncol))     
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list,noise)
    
    #update global model 
    global_model.set_weights(average_weights)
    TIMESTEPS = comm_round
    global_model.save(f"{models_dir}/{TIMESTEPS}")





    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

