****Tourist Attraction Recommender****

**Overview**

This project aims to provide a unique approach to tourist attraction recommendations by utilizing image inputs for classification. Unlike traditional travel planning platforms, our recommender leverages neural networks and image classification techniques to suggest attractions based on the visual content of an input image.

The primary motivation behind this project is to offer a novel solution to alleviate the stress associated with vacation planning. Whether users wish to recreate past experiences from old travel photos or discover new destinations from social media images, our machine learning model aims to inspire and simplify the vacation planning process.

**Dataset Collection and Labeling**

We began by web scraping attractions and their respective images. To facilitate image classification, we used natural language processing (NLP) and topic modeling on attraction names to create attraction classes. Text data underwent preprocessing steps, such as punctuation removal, lowercasing, and stop word removal. Some attractions required manual labeling correction after the initial NLP labeling.

**Neural Network Training**

Attraction classes and image labels were used to train a neural network with transfer learning, employing the VGG-16 model. To address issues like overfitting and class imbalance, dropout layers were added after each dense layer, and L2 regularizers were incorporated. Class combining and random undersampling techniques were employed to handle class imbalances. Additionally, data augmentation was performed on the remaining images in the training dataset, including flip, random transform, and noise variations.

**Image Classification and Recommendation**

Once the neural network was trained, it classified images and recommended attractions based on similarity. This was achieved by calculating distances between images using both cosine distance and a combination of VGG-16 feature vectors and color distribution vectors. Color distribution vectors were derived by dividing an image into sections and representing color distributions of red, green, and blue in each section. These vectors were then combined into a single vector representing the image's color features.

The distances were summed up, with the VGG-16 vector multiplied by a scalar to account for varying importance. Aggregating images by attraction name, the mean distance between the input image and all attractions in the class was computed. Attractions with the shortest distances were recommended to the user.
