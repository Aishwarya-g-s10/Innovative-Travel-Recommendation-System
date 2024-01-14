import pandas as pd
import numpy as np

# color distributions
import cv2
import imutils

#img show 
import matplotlib.pyplot as plt
from matplotlib import image

#img load
import requests
# !pip install fake_useragent
from fake_useragent import UserAgent 
from io import BytesIO
from PIL import Image

def get_bottleneck_features(model, input_imgs):
    '''
    get vgg features for an array of images
    '''
    features = model.predict(input_imgs, verbose=0)
    return features

def histogram(image, mask, bins):
    '''
    get histogram of reg, green, and blue color distribution feature vectors for a section of an image (represented by a mask)
    return 3 histograms as a single color feature vecotr
    '''
    # extract a 3D color histogram from the masked region of the image, using the supplied number of bins per channel
    hist = cv2.calcHist([image], [0,1,2], mask, [bins[0],bins[1],bins[2]],[0, 180, 0, 256, 0, 256])
    
    # normalize the histogram if we are using OpenCV 2.4
    if imutils.is_cv2():
        hist = cv2.normalize(hist).flatten()
        
    # otherwise handle for OpenCV 3+
    else:
        hist = cv2.normalize(hist, hist).flatten()

    return hist

def get_color_description(img_array, bins):
    '''
    get color distribution feature vector for an image with specified number of bins
    split image up into 5 sections, 4 corners and a center ellipse, getting r,g, and b distribution vectors
    '''
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

def plot_color_hist(image, bins, mask, name):
    '''
    plot red, green, and blue color distribution for a section of an image
    '''

    rgb    = cv2.split(image)

    plt.figure(figsize = (10,5))
    colors = ['r','g','b']
    
    for i,col in enumerate(colors):
        arr = rgb[i] * mask
        arr = arr.flatten()
        plt.hist(list(arr), bins, [1,256], color=col) 

        plt.xlim([0,256])
        
    #save
    if name != None:
        path = '../Graphs/' + 'color_hist_bottom_right' + '.png'
        plt.savefig(path)
    
    plt.show()

def show_image(img):
    
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.grid(False)
    plt.show()
    
def load_image(url):
    '''
    load an image from url, return image array
    '''
    ua = UserAgent()
    headers = {'user-agent': ua.random}

    response = requests.get(url, headers = headers)
    image_io = BytesIO(response.content)
    img = Image.open(image_io)    
    return np.array(img)
    
def show_save_image(img, name):
    '''
    show and save image for presentation
    '''
    # show
    plt.rcParams['figure.figsize']=(4,4)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.grid(False)
    plt.show()
    
    #save
    path = '' + name + '.png'
    image.imsave(path, img)
    
def show_images(img_list, category):
    '''
    show multiple images
    '''
    plt.figure(figsize = (25,15));
    t = f"Showing from {category} category"
    
    plt.suptitle(t, fontsize=20)
    for index, img in enumerate(img_list):
        plt.subplot(10,10,index+1);
        plt.imshow(img)
        plt.axis('off')

def get_categories_top_bottom_imgs(df, categories, num):
    '''
    find centroid of df, based on color features and return closest and furthest images from centroid
    '''
    grouped = df.groupby('category')
    group_centroid = df.groupby("category")["color_feats"].apply(lambda x: np.array(x.tolist()).mean(axis=0))

    firstvals = {}
    lastvals = {}

    
    for idx,cat in enumerate(categories):
        group = grouped.get_group(cat) 

        cent = group_centroid[cat]

        sorted_dist = get_sorted_distances_list(group, cent)
        
        firstvals[cat] = [k for k in list(sorted_dist.keys())[:num]]
        lastvals[cat] = [k for k in list(sorted_dist.keys())[-num:]]
    
    return (firstvals, lastvals)

def get_sorted_distances_list(df, feat):
    '''
    get distances and return sroted dictionary with indexes as keys and distances as values
    '''
    distances = {}
    for index, row in df.iterrows():
        pt = np.array(row[4])
        dist = np.linalg.norm(feat-pt)
        distances[index] = dist

    # distances
    sorted_dist = dict(sorted(distances.items(), key=lambda item: item[1]))
    
    return sorted_dist

def remove_bottom_imgs(X, df, categories, bottom_imgs_dict, remove):
    '''
    remove furthest imgages from the centroid
    '''
    remove_indexes = remove
    for cat in categories:
        remove_indexes = remove_indexes + bottom_imgs_dict[cat]
        
    df.drop(df.index[remove_indexes], inplace = True)
    X = np.delete(X, remove_indexes, 0)
    
    return X, df