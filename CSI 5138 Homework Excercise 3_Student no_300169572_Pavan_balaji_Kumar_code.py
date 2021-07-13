# importing required packages
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,SimpleRNN,Dense,Dropout,Flatten, AveragePooling1D  
from tensorflow.keras.regularizers import l2
import os
import urllib.request as req
import tarfile
import zipfile
import random


# Downloading the required IMDB dataset
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

filename = "aclImdb_v1.tar.gz"
if not os.path.exists(filename):
    req.urlretrieve(url, filename)
    
folder = "aclImdb"
if not os.path.exists(folder):
    with tarfile.open(filename) as IMDB:
        IMDB.extractall()

# Code to download the glove dataset
# uncomment below to use 

# glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"

# save_filename = "glove.6B.zip"
# if not os.path.exists(save_filename):
#     req.urlretrieve(glove_url, save_filename)
    
# EMBEDDING_SIZE = 50

# glove_filename = "glove.6B.{}d.txt".format(EMBEDDING_SIZE)
# if not os.path.exists(glove_filename) and EMBEDDING_SIZE in [50,100,200,300]:
#     with zipfile.ZipFile(save_filename, 'r') as z:
#         z.extractall()

# Function to generate embedding matrix from glove data
def get_embedding_matrix(word_index,word_embedding):
    '''
    Parameters
    ----------
    word_index : Dict
        Dictionary consisting of word and its index given by tokenizer 
    word_embedding : Dict
        Dictionary consisting of word and its weights from glove data

    Returns
    -------
    embedding_matrix : numpy array
        Numpy array consisting of weights derived from glove data
        corresponding to the words in the word index
    '''
    num_words = len(word_index)+1
    dimension = len(word_embedding[list(word_embedding.keys())[0]])
    embedding_matrix = np.zeros((num_words ,dimension))
   
    for word,index in word_index.items():
        weights = word_embedding.get(word)
        if weights is not None:
            embedding_matrix[index-1] = weights            
        
    return embedding_matrix

# Function to get word embeddings from glove file
def get_word_embeddings(file_path):
    '''
    Parameters
    ----------
    file_path : str
        String containing the path for the pre-trained 
        word emmbeddings
        
    Returns
    -------
    embedding : Dict
        Dictionary consisting of the word as the key and 
        embeddings as the values
    '''
    embedding = dict()
    
    with open(file_path,"r",encoding="utf-8") as pretrained_embedding:
        for line in pretrained_embedding:
            values = line.split(sep=" ")
            token = values[0]
            weights = np.asarray(values[1:],dtype=np.float32)
            embedding[token] = weights 
    
    return embedding

# funtion to get data from Text files
def get_data(path,directories):
    '''
    Parameters
    ----------
    path : Str
        Path containing the required directories
    directories : List
        The directories contaning the required data

    Returns
    -------
    data : List
        Dataset obtainined fron the given directories
    label : List
        labels correseponding to each entry in the data
    '''
    data = list()
    label= list()
    
    for i in os.listdir(path=path):
        if i in directories:
            # loading data
            for j in os.listdir(path=os.path.join(path,i)):
                    with open(os.path.join(path,i,j),"r",encoding="utf-8") as review:
                        data.append(review.read())
                    # Assigning label
                    if i == directories[0]:
                        label.append([1,0])
                    else:
                        label.append([0,1])

    data = np.array(data)
    label = np.array(label)

    index = [*range(len(data))]
    random.shuffle(index)

    
    return data[index],label[index]

directories =['pos','neg'] # directories containing the required data  
X_train,Y_train = get_data("aclImdb/train",directories) # loading the train data
X_test,Y_test = get_data("aclImdb/test",directories) # loading the test data

# uncomment and run to load word embeddings from glove file
# glove_path = "glove.6B.300d.txt" # file that contains the pre-trained word embeddings
# word_embeddings = get_word_embeddings(glove_path) # loading word embeddings

# Tokenizing the input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Getting the pretrained word embedding for the words in the data from glove

# embedding_matrix = get_embedding_matrix(word_index,word_embeddings)
# np.save("glove embeddings for imbd words",embedding_matrix)

# loading the word embeddings
embedding_weights=np.load("glove embeddings for imbd words.npy")

# Function for converting text data to sequence data
def text_to_sequence(max_words):
  """  
    Parameters
    ----------
    max_words : int
        The maxium no of words from the review
    Returns
    -------
    tr_data : numpy array
        Training data
    ts_data : numpy arrayE
        Testing data
    """
  # Generates train input sequence
  tr_data = tokenizer.texts_to_sequences(X_train)
  tr_data = list(map(lambda x:list(np.array(x)-1),tr_data))
  tr_data = pad_sequences(tr_data,maxlen=max_words)   

  # Generates train input sequence
  ts_data = tokenizer.texts_to_sequences(X_test)
  ts_data = list(map(lambda x:list(np.array(x)-1),ts_data))
  ts_data = pad_sequences(ts_data,maxlen=max_words)

  return tr_data,ts_data 

# Creating testing and training data
train_data,test_data= text_to_sequence(200)

# Function to create LSTM
def create_LSTM(hidden,embedding_matrix,input_length,number_of_layers=0):
   '''
    Parameters
    ----------
    hidden : int
        State dimension
    embedding_matrix : TYPE
        Word embeddings for embedding layer
    input_length : int
        maximum words to consider
    number_of_layers : int, optional
        No of LSTM layers. The default is 0.
    Returns
    -------
    LSTM_model : neural net
        LSTM network
    '''
   name = "LSTM_with_state_dimension_"+str(hidden)
   LSTM_model = Sequential(name=name)
   LSTM_model.add(Embedding(embedding_weights.shape[0],embedding_weights.shape[1],input_length = input_length,
                          weights=[embedding_matrix],trainable = False))
   for i in range(1,number_of_layers):  
     LSTM_model.add(LSTM(hidden, dropout=0.3,kernel_regularizer=l2(.0005),return_sequences=True))
   LSTM_model.add(LSTM(hidden,dropout=0.2,kernel_regularizer=l2(.001)))
   LSTM_model.add(Flatten())
   LSTM_model.add(Dense(128, activation='relu'))
   LSTM_model.add(Dense(64, activation='relu'))
   LSTM_model.add(Dense(2, activation='sigmoid'))

   LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

   return LSTM_model

# Creating LSTM with 20 state dimensions
LSTM_model_1 = create_LSTM(20,embedding_weights,train_data.shape[1])
LSTM_model_1.summary()

# Training model_1 
LSTM_history_1 = LSTM_model_1.fit(train_data,np.array(Y_train),epochs=10,batch_size=64,validation_split=0)

# Testing model_1
LSTM_result_1=LSTM_model_1.evaluate(test_data,np.array(Y_test))

# Creating LSTM with 50 state dimensions
LSTM_model_2 = create_LSTM(50,embedding_weights,train_data.shape[1])
LSTM_model_2.summary()

# Training model_2
LSTM_history_2 = LSTM_model_2.fit(train_data,np.array(Y_train),epochs=10,batch_size=64,validation_split=0)

# Testing model_2
LSTM_result_2=LSTM_model_2.evaluate(test_data,np.array(Y_test))

# Creating LSTM with 100 state dimensions
LSTM_model_3 = create_LSTM(100,embedding_weights,train_data.shape[1])
LSTM_model_3.summary()

# Training model_3
LSTM_history_3 = LSTM_model_3.fit(train_data,np.array(Y_train),epochs=10,batch_size=64, validation_split=0)

# Testing model_3
LSTM_result_3=LSTM_model_3.evaluate(test_data,np.array(Y_test))

# Creating LSTM with 200 state dimensions
LSTM_model_4 = create_LSTM(200,embedding_weights,train_data.shape[1])
LSTM_model_4.summary()

# Training model_4
LSTM_history_4 = LSTM_model_4.fit(train_data,np.array(Y_train),epochs=10,batch_size=64,validation_split=0)

# Testing model_4
LSTM_result_4=LSTM_model_4.evaluate(test_data,np.array(Y_test))

# Creating LSTM with 500 state dimensions
LSTM_model_5 = create_LSTM(500,embedding_weights,train_data.shape[1])
LSTM_model_5.summary()

# Training model_5
LSTM_history_5 = LSTM_model_5.fit(train_data,np.array(Y_train),epochs=10,batch_size=64,validation_split=0)

# Testing model_5
LSTM_result_5=LSTM_model_5.evaluate(test_data,np.array(Y_test))

# Creating LSTM with 2 layers and 200 state dimensions
LSTM_model_6 = create_LSTM(200,embedding_weights,train_data.shape[1],2)
LSTM_model_6.summary()

# Training model_6
LSTM_history_6 = LSTM_model_6.fit(train_data,np.array(Y_train),epochs=10,batch_size=64,validation_split=0)

# Testing model_6
LSTM_result_6=LSTM_model_6.evaluate(test_data,np.array(Y_test))

# function to create RNN
def create_rnn(hidden,embedding_matrix,input_length,no_of_RNN_layers =0):
   '''
    Parameters
    ----------
    hidden : int
        State dimension
    embedding_matrix : TYPE
        Word embeddings for embedding layer
    input_length : int
        maximum words to consider
    number_of_RNN_layers : int, optional
        No of RNN layers. The default is 0.
    Returns
    -------
    RNN_model : neural net
        RNN network
    '''

   name = "RNN_with_state_dimension_"+str(hidden)
   RNN_model = Sequential(name= name)
   RNN_model.add(Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],input_length = input_length,
                          weights=[embedding_matrix],trainable = False))
   for i in range(1,no_of_RNN_layers):
     RNN_model.add(SimpleRNN(hidden, dropout=0.2,return_sequences= True))
     RNN_model.add(AveragePooling1D())
   RNN_model.add(SimpleRNN(hidden, dropout=0.2,kernel_regularizer=l2(0.001)))
   RNN_model.add(Flatten())
   RNN_model.add(Dense(128, activation='relu'))
   RNN_model.add(Dropout(0.2))
   RNN_model.add(Dense(64, activation='relu'))
   RNN_model.add(Dense(2, activation='sigmoid'))

   RNN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

   return RNN_model

# Creating RNN with 20 state dimension
RNN_model_1 = create_rnn(20,embedding_weights,train_data.shape[1])
RNN_model_1.summary()

# Training model_1 
RNN_history_1=RNN_model_1.fit(train_data,np.array(Y_train),epochs=10,batch_size=64)

# Testing model_1
RNN_result_1=RNN_model_1.evaluate(test_data,np.array(Y_test))

# Creating RNN with 50 state dimension
RNN_model_2 = create_rnn(50,embedding_weights,train_data.shape[1])
RNN_model_2.summary()

# Training model_2 
RNN_history_2=RNN_model_2.fit(train_data,np.array(Y_train),epochs=10)

# Testing model_2
RNN_result_2=RNN_model_2.evaluate(test_data,np.array(Y_test))

# Creating RNN with 100 state dimension
RNN_model_3 = create_rnn(100,embedding_weights,train_data.shape[1],2)
RNN_model_3.summary()

# Training model_3 
RNN_history_3=RNN_model_3.fit(train_data,np.array(Y_train),epochs=10,batch_size=64)

# Testing model_3
RNN_result_3=RNN_model_3.evaluate(test_data,np.array(Y_test))

# Creating RNN with 200 state dimension
RNN_model_4 = create_rnn(200,embedding_weights,train_data.shape[1],2)
RNN_model_4.summary()

# Training model_4 
RNN_history_4=RNN_model_4.fit(train_data,np.array(Y_train),epochs=10,batch_size=64)

# Testing model_4
RNN_result_4=RNN_model_4.evaluate(test_data,np.array(Y_test))

# Creating RNN with 500 state dimension
RNN_model_5 = create_rnn(500,embedding_weights,train_data.shape[1])
RNN_model_5.summary()

# Training model_5 
RNN_history_5=RNN_model_5.fit(train_data,np.array(Y_train),epochs=10)

# Testing model_5
RNN_result_5=RNN_model_5.evaluate(test_data,np.array(Y_test))

