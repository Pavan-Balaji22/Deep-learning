


# importing required modules
import idx2numpy
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,Flatten,Lambda,Input,Reshape,Layer,LeakyReLU,Dropout,BatchNormalization
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as k
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

#importing datasets for mnist
mnist_train = idx2numpy.convert_from_file("/content/drive/MyDrive/train-images.idx3-ubyte")
mnist_trainlabel = idx2numpy.convert_from_file("/content/drive/MyDrive/train-labels.idx1-ubyte")
mnist_test = idx2numpy.convert_from_file("/content/drive/MyDrive/t10k-images.idx3-ubyte")
mnist_testlabel = idx2numpy.convert_from_file("/content/drive/MyDrive/t10k-labels.idx1-ubyte")

mnist_train = mnist_train.astype("float32")/255
mnist_test = mnist_test.astype("float32")/255

# expanding the dimensions to satisfy the conv2d layer constraints
mnist_train = np.expand_dims(mnist_train,-1)
mnist_test = np.expand_dims(mnist_test,-1)

def cifar_10_load(file):
    a = list()
    b=  list()
    
    # loading all the batch files
    for i in file:
        with open(i, 'rb') as data:
            dict1 = pickle.load(data, encoding='bytes')
            keys = list(dict1.keys())
            b.append(dict1[keys[1]])
            a.append((dict1[keys[2]]).reshape(len(dict1[keys[2]]),3,32,32).transpose(0,2,3,1))
    
    # Joining all the batch files into one dataset    
    data = a[0]
    labels = b[0]

    if len(a) > 1:
      for j in range(1,len(a)):
          data = np.concatenate([data,a[j]])
          labels = np.concatenate([labels,b[j]])
        
    del a,b
    
    return data,np.array(labels)

# name of the batch files
train_lists = ["/content/drive/MyDrive/cifar-10-batches-py/data_batch_1",
               "/content/drive/MyDrive/cifar-10-batches-py/data_batch_2",
              "/content/drive/MyDrive/cifar-10-batches-py/data_batch_3",
               "/content/drive/MyDrive/cifar-10-batches-py/data_batch_4",
              "/content/drive/MyDrive/cifar-10-batches-py/data_batch_5"]

with open("/content/drive/MyDrive/cifar-10-batches-py/batches.meta",'rb') as x:
    batch_label = pickle.load(x,encoding='bytes')
    labels = batch_label[list(batch_label.keys())[1]]
    
x.close()

# Getting the cifar test and traning datasets
cifar_10_train,cifar_10_trainlabels = cifar_10_load(train_lists)
cifar_10_test,cifar_10_testlabels = cifar_10_load(["/content/drive/MyDrive/cifar-10-batches-py/test_batch"])

cifar_10_train = cifar_10_train.astype("float32")/255
cifar_10_test = cifar_10_test.astype("float32")/255
# input shapes to create neural networks
mnist_input_shape = mnist_train.shape[1:]
cifar_10_input_shape= cifar_10_train.shape[1:]

# Loss for VAE
class loss_layer(Layer):
  def loss(self,x,f,mu,sigma):
      x = k.flatten(x)
      f = k.flatten(f)

      r_loss = keras.metrics.binary_crossentropy(x,f)
      k_loss =-5e-4 * k.mean(1 + sigma - k.square(mu) - k.exp(sigma), axis=-1)

      return  k.mean(r_loss+k_loss)
  
  def call(self,input):
    x,f,mu,sigma = input
    loss = self.loss(x,f,mu,sigma)
    self.add_loss(loss,inputs = [x,f])
    return x

# CLass to create VAE
class VAE(loss_layer):
  def __init__(self,input_shape,n_conv,n_dense,latent_space):
    self.shape = input_shape
    self.n_conv = n_conv
    self.n_dense = n_dense
    self.latent_space = latent_space
    loss_layer.__init__(self)  
  
# Fucntion to perform reparameterization trick
  def latent_layer(self,aruguments):
    mu,sigma = aruguments
    eps = k.random_normal((k.shape(mu)[0],k.shape(mu)[1]))
    return mu + k.exp(sigma / 2) * eps

# Function to build encoder
  def build_encoder(self):
    self.img = Input(shape=self.shape)
    
    for i in range(self.n_conv):
      if i == 0:
        self.X = Conv2D(64, 2,strides=(2,2),activation='relu')(self.img)
      else:
        self.X = Conv2D(64,2,strides=(2,2),activation='relu')(self.X)
    
    self.conv_shape = k.int_shape(self.X)
    self.X = Flatten()(self.X)

    for i in range(self.n_dense):
       self.X = Dense(128,activation='relu')(self.X)
    
    self.mu = Dense(self.latent_space,name = 'mu')(self.X)
    self.sigma = Dense(self.latent_space,name = 'sigma')(self.X)
    self.f = Lambda(self.latent_layer,output_shape = (self.latent_space,), name ='f')([self.mu,self.sigma])

    self.encoder = Model(self.img,[self.mu,self.sigma,self.f],name = 'Encoder')
# Function to build decoder
  def build_decoder(self):
    self.d_in = Input(shape=(self.latent_space,),name = 'd_in')
    c_shape = self.conv_shape
    for i in range(self.n_dense):
      if i == (self.n_dense -1):
        self.X = Dense(np.prod(c_shape[1:]),activation="relu")(self.X)
      elif i ==0:
        self.X = Dense(128,activation="relu")(self.d_in)
      else:
        self.X = Dense(128,activation="relu")(self.X)
    
    self.X = Reshape(c_shape[1:])(self.X)
    
    for i in range(self.n_conv):                  
      self.X = Conv2DTranspose(64,2,activation="relu",strides=(2,2))(self.X)
    
    self.X = Conv2DTranspose(self.shape[-1],2,padding = 'same',activation='sigmoid',name='d_out')(self.X)

    self.decoder = Model(self.d_in,self.X,name="decoder")



# Function to build VAE
  def build(self):
    self.build_encoder()
    self.build_decoder()

    self.f_last =self.decoder(self.f)
    self.y = loss_layer()([self.img,self.f_last,self.mu,self.sigma])

    self.model = Model(self.img,self.y)
    self.model.compile(optimizer = 'adam',loss = self.loss)
    self.model.summary()
    return self.model

# Training VAE based on minist dataset for different no of hidden layers and latent space 
mnist_vae_1 = VAE(mnist_input_shape,2,2,2)  
mnist_vae_1_model =mnist_vae_1.build()
mnist_vae_1_histroy =mnist_vae_1_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_2 = VAE(mnist_input_shape,2,4,2)  
mnist_vae_2_model =mnist_vae_2.build()
mnist_vae_2_histroy =mnist_vae_2_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_3 = VAE(mnist_input_shape,2,6,2)  
mnist_vae_3_model =mnist_vae_3.build()
mnist_vae_3_histroy =mnist_vae_3_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_4 = VAE(mnist_input_shape,2,10,2)  
mnist_vae_4_model =mnist_vae_4.build()
mnist_vae_4_histroy =mnist_vae_4_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_5 = VAE(mnist_input_shape,2,12,2)  
mnist_vae_5_model =mnist_vae_5.build()
mnist_vae_5_histroy =mnist_vae_5_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_6 = VAE(mnist_input_shape,2,4,10)  
mnist_vae_6_model =mnist_vae_6.build()
mnist_vae_6_histroy =mnist_vae_6_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_7 = VAE(mnist_input_shape,2,4,20)  
mnist_vae_7_model =mnist_vae_7.build()
mnist_vae_7_histroy =mnist_vae_7_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_8 = VAE(mnist_input_shape,2,4,30)  
mnist_vae_8_model =mnist_vae_8.build()
mnist_vae_8_histroy =mnist_vae_8_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_9 = VAE(mnist_input_shape,2,4,50)  
mnist_vae_9_model =mnist_vae_9.build()
mnist_vae_9_histroy =mnist_vae_9_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)

mnist_vae_10 = VAE(mnist_input_shape,2,4,75)  
mnist_vae_10_model =mnist_vae_10.build()
mnist_vae_10_histroy =mnist_vae_10_model.fit(mnist_train,None,epochs = 10,validation_split =0.2)


# Plotting the generated images
fig,axes = plt.subplots(5,5,figsize = (15,7))
for i in range(5):
  for j in range(5):
    axes[i,j].imshow(mnist_vae_10.decoder(np.random.randn(1,75)).numpy().reshape(28,28))
fig.tight_layout()

# loss for VAE for CIFAR
class loss_layer_ci(Layer):
  def loss(self,x,f,mu,sigma):
      x = k.flatten(x)
      f = k.flatten(f)

      r_loss = keras.metrics.binary_crossentropy(x,f)
      k_loss =-0.5 * k.mean(1 + sigma - k.square(mu) - k.exp(sigma), axis=-1)

      return  k.mean(r_loss+k_loss)
  
  def call(self,input):
    x,f,mu,sigma = input
    loss = self.loss(x,f,mu,sigma)
    self.add_loss(loss,inputs = [x,f])
    return x
# CLass to create VAE for CIFAR
class VAE_ci(loss_layer_ci):
  def __init__(self,input_shape,n_dense,latent_space):
    self.shape = input_shape
    self.n_dense = n_dense
    self.latent_space = latent_space
    loss_layer_ci.__init__(self)  
  
# Function to peroform reparameterization trick
  def latent_layer(self,aruguments):
    mu,sigma = aruguments
    eps = k.random_normal((k.shape(mu)[0],k.shape(mu)[1]),mean=0,stddev=1)
    return mu + k.exp(sigma ) * eps
# Function to build encoder
  def build_encoder(self):
    self.img = Input(shape=self.shape)
    
    
    self.X = Conv2D(self.shape[-1], 2,padding='same',activation='relu')(self.img)     
    self.X = Conv2D(64,3,strides=(2,2),padding = 'same',activation='relu')(self.X)
    self.X = Conv2D(64,3,strides=(1,1),padding = 'same',activation='relu')(self.X)
    self.X = Conv2D(64,3,strides=(1,1),padding = 'valid',activation='relu')(self.X)

    self.conv_shape = k.int_shape(self.X)
    self.X = Flatten()(self.X)
    for i in range(self.n_dense):
      self.X = Dense(128,activation='relu')(self.X)

    self.mu = Dense(self.latent_space,name = 'mu')(self.X)
    self.sigma = Dense(self.latent_space,name = 'sigma')(self.X)
    self.f = Lambda(self.latent_layer,output_shape = (self.latent_space,), name ='f')([self.mu,self.sigma])

    self.encoder = Model(self.img,[self.mu,self.sigma,self.f],name = 'Encoder')
    
# Function to build decoder
  def build_decoder(self):
    self.d_in = Input(shape=(self.latent_space,),name = 'd_in')
    c_shape = self.conv_shape
    
        
    self.X = Dense(128,activation="relu")(self.d_in)
    self.X = Dense(128,activation="relu")(self.d_in)
    self.X = Dense(np.prod(c_shape[1:]),activation="relu")(self.X)
    
    self.X = Reshape(c_shape[1:])(self.X)
    
    self.X = Conv2DTranspose(64,kernel_size =3,activation="relu",padding ='same',strides=1)(self.X)
    self.X = Conv2DTranspose(64,kernel_size = 3,padding = 'valid',activation='relu',strides=1)(self.X)

    self.X = Conv2DTranspose(64,kernel_size= (3,3) ,activation="relu",padding = 'valid',strides=(2,2))(self.X)
    self.X = Conv2D(self.shape[-1],2, padding = "valid",activation='sigmoid',name='d_out')(self.X)

    self.decoder = Model(self.d_in,self.X,name="decoder")

# Function to build VAE for cifar
  def build(self):
    self.build_encoder()
    self.build_decoder()

    self.f_last =self.decoder(self.f)
    self.y = loss_layer_ci()([self.img,self.f_last,self.mu,self.sigma])

    self.model = Model(self.img,self.y)
    self.model.compile(optimizer = 'adam',loss = None)
    self.model.summary()
    return self.model

# Training VAE based on cifar_10 dataset for different nof of hidden layers and latent space 
cifar_10_1 = VAE_ci(cifar_10_input_shape,2,100)
cifar_10_1_model=cifar_10_1.build()
cifar_10_1_history = cifar_10_1_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_2 = VAE_ci(cifar_10_input_shape,4,100)
cifar_10_2_model=cifar_10_2.build()
cifar_10_2_history = cifar_10_2_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_3 = VAE_ci(cifar_10_input_shape,6,100)
cifar_10_3_model=cifar_10_3.build()
cifar_10_3_history = cifar_10_3_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_4 = VAE_ci(cifar_10_input_shape,8,100)
cifar_10_4_model=cifar_10_4.build()
cifar_10_4_history = cifar_10_4_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_5 = VAE_ci(cifar_10_input_shape,10,100)
cifar_10_5_model=cifar_10_5.build()
cifar_10_5_history = cifar_10_5_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_6 = VAE_ci(cifar_10_input_shape,4,250)
cifar_10_6_model=cifar_10_6.build()
cifar_10_6_history = cifar_10_6_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_7 = VAE_ci(cifar_10_input_shape,4,350)
cifar_10_7_model=cifar_10_7.build()
cifar_10_7_history = cifar_10_7_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_8 = VAE_ci(cifar_10_input_shape,4,500)
cifar_10_8_model=cifar_10_8.build()
cifar_10_8_history = cifar_10_8_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_9 = VAE_ci(cifar_10_input_shape,4,750)
cifar_10_9_model=cifar_10_9.build()
cifar_10_9_history = cifar_10_9_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

cifar_10_10 = VAE_ci(cifar_10_input_shape,4,1000)
cifar_10_10_model=cifar_10_10.build()
cifar_10_10_history = cifar_10_10_model.fit(cifar_10_train,None,epochs = 30 ,validation_split=0.2,batch_size=128)

# Plotting the generated images
fig,axes = plt.subplots(5,5,figsize = (15,7))
for i in range(5):
  for j in range(5):
    ind = np.random.randint(0,len(cifar_10_train),1)
    axes[i,j].imshow(cifar_10_10.decoder(cifar_10_10.encoder(cifar_10_train[ind])[0]).numpy().reshape(32,32,3))
fig.tight_layout()

# Obtaining results from training for mnist
mnist_result_1 = mnist_vae_1_histroy.history["loss"]
mnist_result_2 = mnist_vae_2_histroy.history["loss"]
mnist_result_3 = mnist_vae_3_histroy.history["loss"]
mnist_result_4 = mnist_vae_4_histroy.history["loss"]
mnist_result_5 = mnist_vae_5_histroy.history["loss"]
mnist_result_6 = mnist_vae_6_histroy.history["loss"]
mnist_result_7 = mnist_vae_7_histroy.history["loss"]
mnist_result_8 = mnist_vae_8_histroy.history["loss"]
mnist_result_9 = mnist_vae_9_histroy.history["loss"]
mnist_result_10 = mnist_vae_10_histroy.history["loss"]

# Obtaining results from training for cifar_10
cifar_10_result_1 = cifar_10_1_history.history["loss"]
cifar_10_result_2 = cifar_10_2_history.history["loss"]
cifar_10_result_3 = cifar_10_3_history.history["loss"]
cifar_10_result_4 = cifar_10_4_history.history["loss"]
cifar_10_result_5 = cifar_10_5_history.history["loss"]
cifar_10_result_6 = cifar_10_6_history.history["loss"]
cifar_10_result_7 = cifar_10_7_history.history["loss"]
cifar_10_result_8 = cifar_10_8_history.history["loss"]
cifar_10_result_9 = cifar_10_9_history.history["loss"]
cifar_10_result_10 = cifar_10_10_history.history["loss"]

# Plotting results for mnist
fig,axes =plt.subplots(nrows=1,ncols=2,figsize = (15,7))
axes[0].plot(mnist_result_1,label ="2 Hidden layers")
axes[0].plot(mnist_result_2,label ="4 Hidden layers")
axes[0].plot(mnist_result_3,label ="6 Hidden layers")
axes[0].plot(mnist_result_4,label ="10 Hidden layers")
axes[0].plot(mnist_result_5,label ="12 Hidden layers")
axes[0].set_xlabel("Epochs",size = 'x-large')
axes[0].set_ylabel("KLD loss",size = 'x-large')
axes[0].set_title('Loss for different hidden layers \n',size = 'x-large')
axes[0].legend()

axes[1].plot(mnist_result_5,label ="latent dimension =10 ")
axes[1].plot(mnist_result_6,label ="latent dimension =20")
axes[1].plot(mnist_result_7,label ="latent dimension =30")
axes[1].plot(mnist_result_8,label ="latent dimension =50")
axes[1].plot(mnist_result_10,label ="latent dimension =75")
axes[1].legend()
axes[1].set_xlabel("Epochs",size = 'x-large')
axes[1].set_ylabel("KLD loss",size = 'x-large')
axes[1].set_title('Loss for different latent dimensions \n',size = 'x-large')
fig.tight_layout()
plt.show()


# Plotting results for cifar_10
fig,axes =plt.subplots(nrows=1,ncols=2,figsize = (15,7))
axes[0].plot(cifar_10_result_1,label ="2 Hidden layers")
axes[0].plot(cifar_10_result_2,label ="4 Hidden layers")
axes[0].plot(cifar_10_result_3,label ="6 Hidden layers")
axes[0].plot(cifar_10_result_4,label ="8 Hidden layers")
axes[0].plot(cifar_10_result_5,label ="10 Hidden layers")
axes[0].set_xlabel("Epochs",size = 'x-large')
axes[0].set_ylabel("KLD loss",size = 'x-large')
axes[0].set_title('Loss for different hidden layers \n',size = 'x-large')
axes[0].legend()

axes[1].plot(cifar_10_result_5,label ="latent dimension =10 ")
axes[1].plot(cifar_10_result_6,label ="latent dimension =20")
axes[1].plot(cifar_10_result_7,label ="latent dimension =30")
axes[1].plot(cifar_10_result_8,label ="latent dimension =50")
axes[1].plot(cifar_10_result_10,label ="latent dimension =75")
axes[1].legend()
axes[1].set_xlabel("Epochs",size = 'x-large')
axes[1].set_ylabel("KLD loss",size = 'x-large')
axes[1].set_title('Loss for different latent dimensions \n',size = 'x-large')
fig.tight_layout()
plt.show()

# Merging the results for MNIST into a table
mnistpp = np.r_[mnist_vae_1_histroy.history["loss"][-1],
                mnist_vae_2_histroy.history["loss"][-1],
                mnist_vae_3_histroy.history["loss"][-1],
                mnist_vae_4_histroy.history["loss"][-1],
                mnist_vae_5_histroy.history["loss"][-1],
                mnist_vae_6_histroy.history["loss"][-1],
                mnist_vae_7_histroy.history["loss"][-1],
                mnist_vae_8_histroy.history["loss"][-1],
                mnist_vae_9_histroy.history["loss"][-1],
                mnist_vae_10_histroy.history["loss"][-1]].round(3)
a = [2,4,8,10,12,4,4,4,4,4]
b = [2,2,2,2,2,10,20,30,50,75]
columns = [" Hidden Layers","Latent Dimesnion","KLD loss"]
data = np.c_[a,b,mnistpp]
table_1 = pd.DataFrame(data,index= [*range(1,11)],columns=columns)
table_1


# Merging the results for CIFAR_10 into a table
cifar_10pp = np.r_[cifar_10_1_history.history["loss"][-1],
                   cifar_10_2_history.history["loss"][-1],
                   cifar_10_3_history.history["loss"][-1],
                   cifar_10_4_history.history["loss"][-1],
                   cifar_10_5_history.history["loss"][-1],
                   cifar_10_6_history.history["loss"][-1],
                   cifar_10_7_history.history["loss"][-1],
                   cifar_10_8_history.history["loss"][-1],
                   cifar_10_9_history.history["loss"][-1],
                   cifar_10_10_history.history["loss"][-1]].round(3)
a2 = [2,4,6,8,10,4,4,4,4,4]
b2 = [100,100,100,100,100,250,350,500,750,1000]
columns = [" Hidden Layers","Latent Dimesnion","KLD loss"]
data2 = np.c_[a,b,mnistpp]
table_2 = pd.DataFrame(data,index= [*range(1,11)],columns=columns)
table_2

# GAN

#Fuction to create generator
def  create_generator(latent_shape,node,chn):

  gen = Sequential( name='generator')
  n_nodes =  np.prod(node)
  gen.add(Dense(n_nodes,input_dim = latent_shape))
  gen.add(LeakyReLU(alpha=0.2))
  gen.add(Reshape(node))
  gen.add(Conv2DTranspose(128,(4,4),strides =(2,2),padding = 'same'))
  gen.add(LeakyReLU(alpha=0.2))  
  gen.add(Conv2DTranspose(128,(4,4),strides =(2,2),padding = 'same'))
  gen.add(LeakyReLU(alpha=0.2))
  gen.add(Conv2D(chn, (7,7), activation='sigmoid', padding='same'))
  return gen

#Fuction to create discriminator
def create_discriminator(in_shape =(28,28,1)):
  
  des = Sequential(name='discriminator')
  des.add(Conv2D(64,(3,3),strides = (2,2),padding ='same', input_shape = in_shape))
  des.add(LeakyReLU(alpha=0.2))
  des.add(Dropout(0.4))
  des.add(Conv2D(64,(3,3),strides = (2,2),padding ='same', input_shape = in_shape))
  des.add(LeakyReLU(alpha=0.2))
  des.add(Dropout(0.4))
  des.add(Flatten())
  des.add(Dense(1,activation = 'sigmoid'))
  opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
  des.compile(loss = JSD_loss ,optimizer = opt , metrics =['accuracy'])
  return des

#Fuction to load data
def load_data(dataset,no_of_samples):

  index = np.random.randint(0,dataset.shape[0],no_of_samples)
  X = dataset[index]
  y = np.ones((no_of_samples,1))

  return X,y

#Fuction to load fake data
def load_fake_data(no_of_samples,latent):
  
  X = np.random.randn(no_of_samples*latent).reshape(no_of_samples,latent)
  y= np.ones((no_of_samples,1))
  return X,y

#Fuction to load data from generator
def load_generator_data(model,no_of_samples,latent):

  input = np.random.randn(no_of_samples*latent).reshape(no_of_samples,latent)
  X = model.predict(input)
  y = np.zeros((no_of_samples,1))

  return X,y

#Fuction to create final gan model
def create_gan(gen,des):
  
  des.trainable = False

  gan = Sequential()
  gan.add(gen)
  gan.add(des)
  opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
  gan.compile(loss = JSD_loss, optimizer = opt)

  return gan

# Function to train GAN
def train(gen,des,gan,train,latent_shape,iter = 30,batch_size = 256):
  train_per_iter = int(train.shape[0]/batch_size)

  batch_half = int(batch_size/2)

  d_loss,g_loss = list(),list()
  for i in range(iter):
    for j in range(train_per_iter):

      X_true,y_true = load_data(train,batch_half)

      X_false,y_false = load_generator_data(gen,batch_half,latent_shape)

      X,y = np.vstack((X_true,X_false)),np.vstack((y_true,y_false))

      des_loss,_ = des.train_on_batch(X,y)
      d_loss.append(des_loss)
      X_gan,y_gan = load_fake_data(batch_size,latent_shape)

      gan_loss = gan.train_on_batch(X_gan,y_gan)
      g_loss.append(gan_loss)
  return d_loss,g_loss
       
 
# Function to calculate JSD
def JSD_loss(p, q):
  m = 0.5 * (p + q)
  jsd = 0.5 * keras.losses.KLD(p, m) + 0.5 * keras.losses.KLD(q, m)
  return jsd

# Defining and training models for diffrent latent space for mnist and cifar_10 dataset
latent = 100
chn = 1 
Discriminator_1 = create_discriminator(in_shape=(28,28,1))
Generator_1 =  create_generator(latent,(7,7,128),chn)
GAN_1 = create_gan(Generator_1 ,Discriminator_1)
d_loss_1,g_loss_1=train(Generator_1 , Discriminator_1 , GAN_1 , mnist_train, latent)

latent = 250
chn = 1 
Discriminator_2 = create_discriminator(in_shape=(28,28,1))
Generator_2 =  create_generator(latent,(7,7,128),chn)
GAN_2 = create_gan(Generator_2 ,Discriminator_2)
d_loss_2,g_loss_2=train(Generator_2 , Discriminator_2 , GAN_2 , mnist_train, latent)

latent = 500
chn = 1 
Discriminator_3 = create_discriminator(in_shape=(28,28,1))
Generator_3 =  create_generator(latent,(7,7,128),chn)
GAN_3 = create_gan(Generator_3 ,Discriminator_3)
d_loss_3,g_loss_3=train(Generator_3 , Discriminator_3 , GAN_3 , mnist_train, latent)

latent = 750
chn = 1 
Discriminator_4 = create_discriminator(in_shape=(28,28,1))
Generator_4 =  create_generator(latent,(7,7,128),chn)
GAN_4 = create_gan(Generator_4 ,Discriminator_4)
d_loss_4,g_loss_4 = train(Generator_4 , Discriminator_4 , GAN_4 , mnist_train, latent)

latent = 1000
chn = 1 
Discriminator_5 = create_discriminator(in_shape=(28,28,1))
Generator_5 =  create_generator(latent,(7,7,128),chn)
GAN_5 = create_gan(Generator_5 ,Discriminator_5)
d_loss_5,g_loss_5=train(Generator_5 , Discriminator_5 , GAN_5 , mnist_train, latent)

latent = 100
chn = 3
Discriminator_6 = create_discriminator(in_shape=(32,32,3))
Generator_6 =  create_generator(latent,(8,8,128),chn)
GAN_6 = create_gan(Generator_6 ,Discriminator_6)
d_loss_6,g_loss_6=train(Generator_6 , Discriminator_6 , GAN_6 , cifar_10_train, latent)

latent = 250
chn = 3
Discriminator_7 = create_discriminator(in_shape=(32,32,3))
Generator_7 =  create_generator(latent,(8,8,128),chn)
GAN_7 = create_gan(Generator_7 ,Discriminator_7)
d_loss_7,g_loss_7=train(Generator_7 , Discriminator_7 , GAN_7 , cifar_10_train, latent)

latent = 500
chn = 3
Discriminator_8 = create_discriminator(in_shape=(32,32,3))
Generator_8 =  create_generator(latent,(8,8,128),chn)
GAN_8 = create_gan(Generator_8 ,Discriminator_8)
d_loss_8,g_loss_8=train(Generator_8 , Discriminator_8 , GAN_8 , cifar_10_train, latent)

latent = 750
chn = 3
Discriminator_9 = create_discriminator(in_shape=(32,32,3))
Generator_9 =  create_generator(latent,(8,8,128),chn)
GAN_9 = create_gan(Generator_9 ,Discriminator_9)
d_loss_9,g_loss_9=train(Generator_9 , Discriminator_9 , GAN_9 , cifar_10_train, latent)

latent = 1000
chn = 3
Discriminator_10 = create_discriminator(in_shape=(32,32,3))
Generator_10 =  create_generator(latent,(8,8,128),chn)
GAN_10 = create_gan(Generator_10 ,Discriminator_10)
d_loss_10,g_loss_10=train(Generator_10 , Discriminator_10 , GAN_10 , cifar_10_train, latent)

#PLOTTING IMAGES FOR MNIST
fig,axes = plt.subplots(5,5,figsize = (15,7))
for i in range(5):
  for j in range(5):
    ind = np.random.randint(0,len(cifar_10_train),1)
    axes[i,j].imshow(Generator_3.predict(load_fake_data(1,500)[0]).reshape(28,28))
fig.tight_layout()

#PLOTTING IMAGES FOR cifar_10
fig,axes = plt.subplots(5,5,figsize = (15,7))
for i in range(5):
  for j in range(5):
    ind = np.random.randint(0,len(cifar_10_train),1)
    axes[i,j].imshow(Generator_9.predict(load_fake_data(1,750)[0]).reshape(32,32,3))
fig.tight_layout()

# Plotting loss function for discriminator and GAN for mnist
fig,axes =plt.subplots(nrows=1,ncols=2,figsize = (15,7))
axes[0].plot(g_loss_3,label ="GAN loss")
axes[0].plot(d_loss_3,label ="Discriminator loss")
axes[0].set_xlabel("Epochs",size = 'x-large')
axes[0].set_ylabel("JSD loss",size = 'x-large')
axes[0].set_title('Loss on MNIST dataset \n',size = 'x-large')
axes[0].legend()

# Plotting loss function for discriminator and GAN for CIFAR_10
axes[1].plot(g_loss_10,label ="GAN loss")
axes[1].plot(d_loss_10,label ="Discriminator loss")
axes[1].legend()
axes[1].set_xlabel("Epochs",size = 'x-large')
axes[1].set_ylabel("JSD loss",size = 'x-large')
axes[1].set_title('Loss for CIFAR dataset \n',size = 'x-large')
fig.tight_layout()
plt.show()

#WGAN

# END loss fuction
def ws_loss(true, pred):
    return k.mean(true * pred)
# function to crete critic
def create_critic(input_shape = (28,28,1)): 

  cri = Sequential()

  cri.add(Conv2D(64, 3, strides=(2,2), padding='same', input_shape=input_shape))
  cri.add(BatchNormalization(momentum=0.8))
  cri.add(LeakyReLU(alpha=0.2))
  
  cri.add(Conv2D(128,3, strides=(2,2), padding='same'))
  cri.add(BatchNormalization(momentum=0.8))
  cri.add(LeakyReLU(alpha=0.2))

  cri.add(Conv2D(256, 3, strides=(2,2), padding='same'))
  cri.add(BatchNormalization(momentum=0.8))
  cri.add(LeakyReLU(alpha=0.2))

  cri.add(Conv2D(512, 3, strides=1, padding='same'))
  cri.add(BatchNormalization(momentum=0.8))
  cri.add(LeakyReLU(alpha=0.2))
 
  cri.add(Flatten())
  cri.add(Dense(1))
  opt = keras.optimizers.RMSprop(lr = 0.00005)

  cri.compile(loss = ws_loss,optimizer = opt)
  return cri

# function to crete generator for WGAN
def create_Wgenerator(n_nodes,latent_dim,chn):
  
  init = keras.initializers.RandomNormal(stddev = 0.02)

  gen = Sequential()

  gen.add(Dense(np.prod(n_nodes), kernel_initializer=init, input_dim=latent_dim))
  gen.add(LeakyReLU(alpha=0.2))
  gen.add(Reshape(n_nodes))

  gen.add(Conv2DTranspose(128, 3, strides=(2,2), padding='same'))
  gen.add(BatchNormalization())
  gen.add(LeakyReLU(alpha=0.2))

  gen.add(Conv2DTranspose(128, 3, strides=1, padding='same'))
  gen.add(BatchNormalization())
  gen.add(LeakyReLU(alpha=0.2))

  gen.add(Conv2DTranspose(chn, 3,strides = 2, activation='tanh', padding='same'))
  return gen

# function to WGAN
def create_wgan(gen,cri,latent):
  cri.trainable = False
  input = Input(shape = (latent,))
  noise = gen(input)

  valid = cri(noise)

  wgan = Model(inputs = input,outputs = valid, name = 'wgan') 

  opt = keras.optimizers.RMSprop(lr=0.00005)
  wgan.compile(loss=ws_loss, optimizer=opt, metrics = ['accuracy'])
  return wgan

# Fuction to load data
def wload_data(dataset,no_of_samples):

  index = np.random.randint(0,dataset.shape[0],no_of_samples)
  X = dataset[index]
  y = np.ones((no_of_samples,1))

  return X,y

# Fuction to load fake data
def wload_fake_data(no_of_samples,latent):
  
  X = np.random.randn(no_of_samples*latent).reshape(no_of_samples,latent)
  y= np.ones((no_of_samples,1))
  return X,y

# Fuction to load generator data
def wload_generator_data(model,no_of_samples,latent):

  input = np.random.randn(no_of_samples*latent).reshape(no_of_samples,latent)
  X = model.predict(input)
  y = np.zeros((no_of_samples,1))

  return X,y

# Fuction to train GAN
def train_WGAN(gen, cri, gan, train, latent, epochs=10, batch=64, n_cri=5):
  
  c_loss = []
  g_loss = []
  clip_value = 0.01
  for i in range(epochs+1*(len(train// batch))):
    for k in range(n_cri):

      cri.trainable = True

      X_true,y_true = wload_data(train,batch)
      c_T = cri.train_on_batch(X_true,y_true)

      X_fake,y_fake = wload_generator_data(gen,batch,latent)
      c_F = cri.train_on_batch(X_fake,y_fake)
      
      c_lb = 0.5*(c_T+c_F)   

      cri.trainable = False
      X,y = wload_fake_data(batch,latent)
      g_lb= gan.train_on_batch(X,y)
     
    c_loss.append(c_lb)
    g_loss.append(g_lb[0])


  return c_loss,g_loss

























































































































































# Defining and traing WGAN for different latent space and datasets
latent = 100 
wgen1 = create_Wgenerator((7,7,512),latent,1)
cri1 = create_critic()
wgan1= create_wgan(wgen1,cri1,latent)
c1w,g1w=train_WGAN(wgen1,cri1,wgan1,mnist_train,latent)

latent = 200 
wgen2 = create_Wgenerator((7,7,512),latent,1)
cri2 = create_critic()
wgan2= create_wgan(wgen2,cri2,latent)
c2w,g2w=train_WGAN(wgen2,cri2,wgan2,mnist_train,latent)

latent = 350 
wgen3 = create_Wgenerator((7,7,512),latent,1)
cri3 = create_critic()
wgan3= create_wgan(wgen3,cri3,latent)
c3w,g3w=train_WGAN(wgen3,cri3,wgan3,mnist_train,latent)

latent = 500
wgen4 = create_Wgenerator((7,7,512),latent,1)
cri4 = create_critic()
wgan4 = create_wgan(wgen4,cri4,latent)
c4w,g4w=train_WGAN(wgen4,cri4,wgan4,mnist_train,latent)

latent = 1000 
wgen5 = create_Wgenerator((7,7,512),latent,1)
cri5= create_critic()
wgan5= create_wgan(wgen5,cri5,latent)
c5w,g5w=train_WGAN(wgen5,cri5,wgan5,mnist_train,latent)

latent = 100 
wgen6 = create_Wgenerator((8,8,512),latent,3)
cri6 = create_critic()
wgan6= create_wgan(wgen6,cri6,latent)
c6w,g6w=train_WGAN(wgen6,cri6,wgan6,mnist_train,latent)

latent = 200 
wgen7 = create_Wgenerator((8,8,512),latent,3)
cri7 = create_critic()
wgan7= create_wgan(wgen7,cri7,latent)
c7w,g7w=train_WGAN(wgen7,cri7,wgan7,mnist_train,latent)


latent = 500 
wgen8 = create_Wgenerator((8,8,512),latent,3)
cri8 = create_critic()
wgan8= create_wgan(wgen8,cri8,latent)
c8w,g8w=train_WGAN(wgen8,cri8,wgan8,mnist_train,latent)

latent = 350 
wgen9 = create_Wgenerator((8,8,512),latent,3)
cri9 = create_critic()
wgan9= create_wgan(wgen9,cri9,latent)
c9w,g9w=train_WGAN(wgen9,cri9,wgan9,mnist_train,latent)

latent = 1000 
wgen10 = create_Wgenerator((8,8,512),latent,3)
cri10 = create_critic()
wgan10 = create_wgan(wgen10,cri10,latent)
c10w,g10w=train_WGAN(wgen10,cri10,wgan10,mnist_train,latent)



