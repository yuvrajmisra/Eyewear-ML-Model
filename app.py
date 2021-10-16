import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import pickle as pk
import urllib.request
from scipy.spatial import distance
from flask import Flask, render_template,request
from tensorflow import keras
from tensorflow.keras.utils import normalize
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\\Users\\dgmis\\Desktop\\IIT Mandi\\India Today\\eyewear_app\\Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model1 = keras.models.load_model('C:\\Users\\dgmis\\Desktop\\IIT Mandi\\India Today\\eyewear_app\\parent_model')
model2= keras.models.load_model('C:\\Users\\dgmis\\Desktop\\IIT Mandi\\India Today\\eyewear_app\\frame_model')
feat_extract = keras.models.load_model('C:\\Users\\dgmis\\Desktop\\IIT Mandi\\India Today\\eyewear_app\\feat_model')
pca = pk.load(open("pca.pkl",'rb'))
pca_features=np.load('pca_features.npy',allow_pickle=True)
image_data=np.load('similar_data.npy',allow_pickle=True)
image_data=list(image_data)
X=[];cap=[];
for x,c in image_data:
    X.append(x)
    cap.append(c)
#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    data=[]
    x=request.form.get('URL');
    if x!='':
        url_response = urllib.request.urlopen(x,timeout=20)
        img_array = np.asarray(bytearray(url_response.read()), dtype='uint8')
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        file = request.files['fileupload']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        img= cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename),cv2.IMREAD_COLOR)
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_img = cv2.resize(img,(100,100))
    data.append(new_img)
    data=np.array(data)
    data=normalize(data,axis=1)
    prediction1 = model1.predict(data)
    prediction2 = model2.predict(data)
    output1 = np.argmax(prediction1) 
    output1 = output1.reshape(1,1)
    output2 = np.argmax(prediction2) 
    output2 = output2.reshape(1,1)
    if output1[0]==0:
        s1='eyeframe'
    elif output1[0]==1:
        s1='sunglasses'
    else:
        s1='Non-Power Reading'
    if output2[0]==0:
        s2='Rectangle'
    elif output2[0]==1:
        s2='Aviator'
    elif output2[0]==2:
        s2='Wayfarer'
    else:
        s2='Oval'
    #Uploaded Image
    img=Image.fromarray(new_img)
    data = io.BytesIO()
    img.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())  
    #Finding Similar Images
    feature=feat_extract.predict(np.array([new_img]))
    pca_feature=pca.transform(feature)
    sim_i = [ distance.cosine(pca_feature, feat) for feat in pca_features ]
    close_i = sorted(range(len(sim_i)), key=lambda k: sim_i[k])[0:11]
    #Visualizing closest images
    I=[];Caption=[];
    for j in close_i:
            img = X[j]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img)
            data = io.BytesIO()
            img.save(data, "JPEG")
            encoded_img = base64.b64encode(data.getvalue())
            I.append(encoded_img)
            Caption.append(cap[j])
    return render_template('index.html', frame_text='Frame Shape: ',parent_text='Parent Category: ',
                          similarImage_text='10 Similar Images',prediction_text1=s2,prediction_text2=s1,
                          img_data=encoded_img_data.decode('utf-8'),img_data1=I[0].decode('utf-8'),
                          img_data2=I[1].decode('utf-8'),img_data3=I[2].decode('utf-8'),
                          img_data4=I[3].decode('utf-8'),img_data5=I[4].decode('utf-8'),
                          img_data6=I[5].decode('utf-8'),img_data7=I[6].decode('utf-8'),
                          img_data8=I[7].decode('utf-8'),img_data9=I[8].decode('utf-8'),
                          img_data10=I[9].decode('utf-8'),Caption1=Caption[0],Caption2=Caption[1],
                                      Caption3=Caption[2],Caption4=Caption[3],Caption5=Caption[4],
                                      Caption6=Caption[5],Caption7=Caption[6],Caption8=Caption[7],Caption9=Caption[8],
                                      Caption10=Caption[9])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
