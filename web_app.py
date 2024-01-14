import streamlit as st
# from streamlit import caching

import hashlib
import base64
# import pickle5 as pickle
import pickle

import pandas as pd
import numpy as np
# import pickle5 as pickle
import pickle
import urllib.request

# geocode = RateLimiter(geocoder.geocode, min_delay_seconds = 1,   return_value_on_exception = None) # adding 1 second padding between calls
import requests
import urllib.parse

# img load
from PIL import Image

# models
from keras.models import load_model, Model
from keras.applications import vgg16

#load cnn model
from tensorflow import keras
import os
# model = keras.models.load_model('../Models/vgg_cnn.h5')
export_path = os.path.join(os.getcwd(), 'history_vgg.h5')
model = load_model(export_path)

# color distributions
import cv2
import imutils
import sklearn.preprocessing as preprocessing
import scipy.spatial.distance

# Function to add a background image from a local file
def add_bg_from_local():
    with open('background.jpg', "rb") as file:
        base64_image = base64.b64encode(file.read()).decode('utf-8')

    st.markdown(
        f"""
        
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Add a background image
add_bg_from_local()

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if the user exists and the password is correct
def check_user(username, password):
    try:
        users = pd.read_csv('users.csv')
        user_info = users[users['username'] == username].iloc[0]
        return hash_password(password) == user_info['password']
    except:
        return False

# Function to add a new user
def add_user(username, password):
    try:
        users = pd.read_csv('users.csv')
        if username in users['username'].values:
            return False  # User already exists
        else:
            # Add new user
            users = users.append({'username': username, 'password': hash_password(password)}, ignore_index=True)
            users.to_csv('users.csv', index=False)
            return True
    except Exception as e:
        print(e)
        return False

def login_system():
    st.sidebar.title("Login/Signup")
    menu = ["Login", "Signup"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if check_user(username, password):
                st.session_state['user'] = username
                st.sidebar.success(f"Logged in as {username}")
            else:
                st.sidebar.error("Incorrect username or password.")
                
    elif choice == "Signup":
        new_username = st.sidebar.text_input("Choose a Username", key="new_user")
        new_password = st.sidebar.text_input("Choose a Password", type="password", key="new_pass")
        
        if st.sidebar.button("Signup"):
            if add_user(new_username, new_password):
                st.sidebar.success("You have successfully signed up!")
                st.session_state['user'] = new_username
            else:
                st.sidebar.error("Username is already taken.")


# if login_system():
#     main_app()
# else:
#     st.info("please login or signup")


def main_app():
    # create vgg model with correct input size
    inputs = (150, 150, 3)
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                        input_shape=inputs)

    output = vgg.layers[-1].output
    output = keras.layers.Flatten()(output)
    vgg_model = Model(vgg.input, output)

    #dont want model weights to change durring training
    vgg_model.trainable = False
    for layer in vgg_model.layers:
        layer.trainable = False

    #st.cache
    def load_data(img_class):
        if img_class == 'parks':
            aws_pathA = 'https://streamlit3.s3-us-west-2.amazonaws.com/parksA_df.pkl'
            requests = urllib.request.urlopen(aws_pathA)
            dfA = pickle.load(requests)
            aws_pathB = 'https://streamlit3.s3-us-west-2.amazonaws.com/parksB_df.pkl'
            requests = urllib.request.urlopen(aws_pathB)
            dfB = pickle.load(requests)
            df = pd.concat([dfA, dfB])
            del dfA, dfB
            return df
        else:
            file_name = img_class.replace('/', '_')
            #arn:aws:s3:::streamlit3
            # aws_path = 'https://streamlit3.s3-us-west-2.amazonaws.com/' + file_name +'_df.pkl'
            local_path =   file_name + '_df.pkl'
            # requests = urllib.request.urlopen(aws_path)
            # df = pickle.load(requests)
            df = pickle.load(open(local_path, 'rb'))
            return df

    #st.cache
    def histogram(image, mask, bins):
        # extract a 3D color histogram from the masked region of the image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0,1,2], mask, [bins[0],bins[1],bins[2]],[0, 180, 0, 256, 0, 256])
        
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
            
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()

        return hist

    #st.cache
    def get_color_description(img_array, bins):
        color = cv2.COLOR_BGR2HSV
        img = img_array * 255
        image = cv2.cvtColor(img, color)
        
        features = []
    
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left, top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image, then update the feature vector
            hist = histogram(image, cornerMask,bins)
            features.extend(hist)

            # extract a color histogram from the elliptical region and update the feature vector
            hist = histogram(image, ellipMask, bins)
            features.extend(hist)
        return features

    @st.cache
    def load_image_streamlit(url):
        ua = UserAgent()
        headers = {'user-agent': ua.random}

        response = requests.get(url, headers = headers)
        image_io = BytesIO(response.content)
        img = Image.open(image_io)    
        return img

    def classify(img_vgg, model):
        '''
        find class using cnn model, using img vgg vector and return prediction
        '''
        cats = ['beaches/ocean', 'entertainment', 'gardens/zoo', 'landmarks', 'museums','parks']
        
        predictions = np.array(model.predict(img_vgg))
        pred = np.argmax(predictions) #find max value
        
        return cats[pred] 

    # @st.cache 
    def get_bottleneck_features(model, input_img):
        '''
        get vgg vector features of array of images
        '''
        input_imgs = np.array([input_img])
        
        features = model.predict(input_imgs, verbose=0)
        return features

    # @st.cache   
    def get_distance(img_feats, feats, max_length=100):
        '''
        get distance between vectors
        '''

        img_feats_array = np.ravel(np.array(img_feats))
        feats_array = np.ravel(np.array(feats))
        
        print("Shape of img_feats:", img_feats_array.shape)
        print("Shape of feats:", feats_array.shape)

        similarity = 1 - scipy.spatial.distance.cosine(img_feats_array, feats_array)
        return similarity
        # print("Shape of img_feats:", img_feats.shape)
        # print("Shape of feats:", feats.shape)
        # return scipy.spatial.distance.cosine(img_feats, feats)

    def get_recommendations(img_class, img_array, img_vgg):
        '''
        get df of top attractions and siplay 3 images from top attractions
        '''
        # load df with color and vgg descriptions
        df = load_data(img_class)

        #get color distribution feature vector
        bins = [8,8,8]
        img_color_des = get_color_description(img_array, bins)

        df['color_feats'] = df.apply(lambda row: get_distance(img_color_des, row['color_feats']), axis=1)
        df['vgg_feats'] = df.apply(lambda row: get_distance(img_vgg, row['vgg_feats']), axis=1)


        # df = df.astype({'name': 'category', 'location': 'category'}).dtypes

        # create color and vgg vectors and standardize 
        min_max_scaler = preprocessing.MinMaxScaler()
        color_array = df['color_feats'].values.astype(float).reshape(-1,1)
        scaled_color_array = min_max_scaler.fit_transform(color_array)
        vgg_array = df['vgg_feats'].values.astype(float).reshape(-1,1)
        scaled_vgg_array = min_max_scaler.fit_transform(vgg_array)

        # drop color and vgg columns
        df.drop(['color_feats','vgg_feats'], axis=1, inplace=True)

        # combine arrays, weighing vgg vector depending on class
        if img_class in ['beaches/ocean']:
            total_distance =  0.5*scaled_vgg_array + scaled_color_array
        elif img_class in ['gardens/zoo']:
            total_distance =  5*scaled_vgg_array + scaled_color_array
        elif img_class in ['entertainment', 'landmarks', 'museums']:
            total_distance =  20*scaled_vgg_array + scaled_color_array
        else:
            total_distance =  1* scaled_vgg_array + scaled_color_array

        # add new distance column
        df['distance'] = total_distance

        # groupb attractions and find mean distance
        grouped_df = df.groupby(['name', 'location'])['distance'].mean()
        grouped_df = pd.DataFrame(grouped_df).reset_index()

        # remove attractins with no locations
        grouped_df['length'] = grouped_df.location.str.len()
        grouped_df = grouped_df[grouped_df.length > 3]

        # sort by distance ascending
        grouped_df.sort_values(by=['distance'], ascending=True, inplace=True)

        # get top 3 attractions
        top_df = grouped_df[:3].reset_index()
        atts = [top_df.loc[0,'name'], top_df.loc[1,'name'], top_df.loc[2,'name']]

        del grouped_df

        # groupp by attraction, and get groups for top 3 attractions
        grouped = df.groupby('name')

        del df

        groups = []
        for attraction in atts:
            groups.append(grouped.get_group(attraction))
        show_recommendations(groups, atts) #show recommendations

        del grouped

        return top_df
        

    def show_recommendations(groups, atts):
        '''
        show 3 images for each recommended attraction
        '''
        for idx, group in enumerate(groups):
            df = pd.DataFrame(group).reset_index()
            st.header(atts[idx])
            imgs = [df.loc[0,'url'], df.loc[2,'url'], df.loc[5,'url']]
            st.image(imgs, width = 200)


    st.title('LETS TAKE A TRIP')
    st.header('Tourist Attraction Recommender')
    st.write('Upload an image to get some inspiration for your next vacation.')


    #font size
    st.markdown("""
    <style>
    .big-font {
        font-size:17px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # upload jpg file
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        # open img, resize, get arr, standardize, and get vgg features
        image = Image.open(uploaded_file)
        img = image.resize((150, 150)) 
        img_array = np.array(img)
        img_std = img_array/255
        img_vgg = get_bottleneck_features(vgg_model, img_std)

        #show uploaded img
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        # st.write("")

        # start_execution = st.button('Run model')
        # if start_execution:
        # gif_runner = st.image('car.gif')
        st.markdown('<p class="big-font">Calculating... this could take a few minutes</p>', unsafe_allow_html=True)

            #classify with cnn model
        label = classify(img_vgg, model)
        # if label == 'entertainment':
        #     st.markdown(f'<p class="big-font">Recommeding an {label} attraction</p>', unsafe_allow_html=True)
        # else:
        #     st.markdown(f'<p class="big-font">Recommeding a {label} attraction</p>', unsafe_allow_html=True)
        #     # st.write()

            #get recommedations and show map
        df = get_recommendations(label, img_array, img_vgg)
        # gif_runner.empty()


if 'user' not in st.session_state:
    st.session_state['user'] = None

login_system()

if st.session_state['user'] :
    # Your Streamlit app code here
    st.write(f"Welcome, {st.session_state['user']}!")
    main_app()
else:
    st.info("Please login or signup")

        