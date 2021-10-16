import random
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.layers import Dense, Activation,Flatten,Conv2D,MaxPooling2D

train_data=np.load('parent_100.npy',allow_pickle=True)
train_data=list(train_data)
random.shuffle(train_data)

X=[]
y=[]

for pics,labels in train_data:
    X.append(pics)
    y.append(labels)
    
X=np.array(X)
y=np.array(y)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1,random_state=42,shuffle=True)
y_test=np.array(y_test).reshape(y_test.shape[0],1)
y_train=np.array(y_train).reshape(y_train.shape[0],1)

X_train=normalize(X_train,axis=1)
X_test=normalize(X_test,axis=1)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

Img_size=100

model =Sequential()
model.add(Conv2D(64,(3,3),input_shape = (Img_size,Img_size,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3),input_shape = (Img_size,Img_size,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='sigmoid',kernel_initializer='he_uniform'))
model.add(Dense(3,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics =['accuracy'])
print(model.summary())

model.fit(X_train,y_train,batch_size = 32,epochs = 5)
model.evaluate(X_test,y_test,batch_size=32)
model.save("parent_model")