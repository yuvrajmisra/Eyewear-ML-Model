import keras
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from keras.models import Model
from scipy.spatial import distance
from sklearn.decomposition import PCA
#Read Data and Show Image
data=np.load('similar_data.npy',allow_pickle=True)
data=list(data)
X=[]
names=[]
for i,j in data:
   X.append(i) 
   names.append(j)
x=X[0]
#plt.imshow(x)
np.array(x)
#Convert to Float
X=np.float32(X)
X=list(X)
#Upload Model
model = keras.models.load_model('C:\\Users\\dgmis\\Desktop\\IIT Mandi\\India Today\\parent_model')
model.summary()
prediction=model.predict(np.array([x]))
#Feature extractor
feat_extract=Model(inputs=model.input,outputs=model.get_layer("dense").output)
feat_extract.summary()
#Feature Visualization
feat=feat_extract.predict(np.array([x]))
#plt.figure(figsize=(16,4))
#plt.plot(feat[0])
#Collecting features
features=[]
for img in X:
    feat=feat_extract.predict(np.array([img]))[0]
    features.append(feat)
features = np.array(features)
#features=np.load('features.npy',allow_pickle=True)
#PCA
pca = PCA(n_components=10)
pca.fit(features)
pca_features = pca.transform(features)

#Similarity distance
sim_i = [ distance.cosine(pca_features[200], feat) for feat in pca_features ]
close_i = sorted(range(len(sim_i)), key=lambda k: sim_i[k])[0:11]
#Visualizing closest images
I=[]
for j in close_i:
    img = np.uint8(X[j])
    I.append(img)

# concatenate the images into a single image
concat_img = np.concatenate([np.asarray(t) for t in I], axis=1)

# show the image
plt.figure(figsize = (16,12))
plt.imshow(concat_img)

pk.dump(pca, open("pca.pkl","wb"))
feat_extract.save('feat_model')
np.save("pca_features.npy",pca_features)

