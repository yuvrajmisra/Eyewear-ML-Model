# Eyewear-ML-Model
The India Today Group Machine Learning Challenge

1. Run the preprocessing script and get the processed data ready for the 3 ML models. Namely we will get three types of data files of type ".npy" after running this script:

    a. frame_100.npy

    b. parent_100.npy

    c. similar_data.npy
  
2. Next, we consider the Task 2 files (A and B):

    a. Task 2 A.py contains the ML model to classify the eyewear according to the frame shape. We read from the data file "frame_100.npy" and save the model as "frame_model"          after running the script.

    b. Task 2 B.py contains the ML model to classify the eyewear according to parent category. We read from the data file "parent_100.npy" and save the model as "parent_model"        after running the script.

3. Task 1.py makes use of the model developed in script Task 2 B.py, that is "parent_model", in order to find 10 similar images. We first load this model and then read from the    data file "similar_data.npy". We finally save the following information after running this script:

    a. The image feature extractor model "feat_model"

    b. The principal component fit weights "pca.pkl"

    c. The array of features of all images obtained after performing Principal Component Analysis as "pca_features.npy"

5. Finally we run the app.py script to show the outputs on a webpage on local host. The script loads all the three models saved above and also reads all the information saved      in step 3.  

6. After running the script, we can access the webpage at localhost:5000 on our browser. The webpage accepts both image url and image file as input. If the imput image is          uploaded, it gets saved in the Uploads folder.

7. The webpage outputs the frame shape, parent category and 10 similar images.

