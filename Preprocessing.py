import cv2
import urllib.request
import numpy as np
import pandas as pd

#Reading the Data
dt=pd.read_csv("eyewear_ml_challenge.csv")
dt=dt.iloc[:,1:]

frame_label=[]
for i in dt.index:
    if dt["frame_shape"].iloc[i]=="Rectangle":
        frame_label.append(0)
    elif dt["frame_shape"].iloc[i]=="Aviator":
        frame_label.append(1)
    elif dt["frame_shape"].iloc[i]=="Wayfarer":
        frame_label.append(2)
    else:
        frame_label.append(3)
frame_label=np.array(frame_label).reshape(dt.shape[0],1)
dt=pd.concat([dt,pd.DataFrame(frame_label)],axis=1)
dt.columns.values[5]='frame_label'
dt1=dt[dt['frame_label']==0];dt1=dt1.iloc[0:466,:]
dt2=dt[dt['frame_label']==1];dt2=dt2.iloc[0:466,:]
dt3=dt[dt['frame_label']==2];dt3=dt3.iloc[0:466,:]
dt4=dt[dt['frame_label']==3];dt4=dt4.iloc[0:466,:]
dt=pd.concat([dt1,dt2,dt3,dt4],axis=0)
count=np.unique(np.array(frame_label),return_counts=True)


frame_data=[]
IMG_SIZE1=100
IMG_SIZE2=100


for i in dt.index:
    try:
        url_response = urllib.request.urlopen(dt['Image_Front'][i])
    except urllib.error.HTTPError:
        continue
    img_array = np.asarray(bytearray(url_response.read()), dtype='uint8')
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    new_img = cv2.resize(img,(IMG_SIZE1,IMG_SIZE2))
    frame_data.append([new_img,dt['frame_label'][i]])

frame_data=np.array(frame_data)
np.save("frame_100.npy",frame_data)

parent_label=[]
for i in dt.index:
    if dt["parent_category"].iloc[i]=="eyeframe":
        parent_label.append(0)
    elif dt["parent_category"].iloc[i]=="sunglasses":
        parent_label.append(1)
    else:
        parent_label.append(2)
parent_label=np.array(parent_label).reshape(dt.shape[0],1)
dt=pd.concat([dt,pd.DataFrame(parent_label)],axis=1)
dt.columns.values[5]='parent_label'


parent_data=[]
IMG_SIZE1=100
IMG_SIZE2=100

for i in dt.index:
    try:
        url_response = urllib.request.urlopen(dt['Image_Front'][i])
    #except urllib.error.HTTPError or urllib.error.WINError:
    except urllib.error.URLError: 
        continue
    img_array = np.asarray(bytearray(url_response.read()), dtype='uint8')
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    new_img = cv2.resize(img,(IMG_SIZE1,IMG_SIZE2))
    parent_data.append([new_img,dt['parent_category'][i]])

parent_data=np.array(parent_data)
np.save("parent_100.npy",frame_data)

similar_data=[]
IMG_SIZE1=100
IMG_SIZE2=100

for i in dt.index:
    try:
        url_response = urllib.request.urlopen(dt['Image_Front'][i])
    #except urllib.error.HTTPError or urllib.error.WINError:
    except urllib.error.URLError: 
        continue
    img_array = np.asarray(bytearray(url_response.read()), dtype='uint8')
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    new_img = cv2.resize(img,(IMG_SIZE1,IMG_SIZE2))
    similar_data.append([new_img,dt['product_name'][i]])

similar_data=np.array(similar_data)
np.save("similar_data.npy",similar_data)