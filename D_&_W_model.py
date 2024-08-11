#--------------------IMPORTING A LIBRARY---------------------------------------------------
import os #Importing OS (Operating System) library, which deals with interacting with OS 
import pickle #Importing Pickle,  library which  deals with reading and writing the data
import numpy as np #Importing numpy library which deals with handling integers
from sklearn.model_selection import train_test_split #Imporing train_test_split that deals with testing, training and splitting the dataset
from sklearn.svm import SVC #Importing SVC(Support Vector Classification) which deals with classification related tasks
import cv2 #Importing cv2(Computer Vision) library which is a open CV library (Very useful in dealing with tasks such as object detection and image processing)
import matplotlib.pyplot as plt #Importing matplotlib.pyplot library which deals with inserting data in the graph
import random #Importing random library which deals with generating data randomly
#------------------------------------------------------------------------------------------



#----------------INSERTING A CLASS DIRECTORY----------------------
#input_dir = 'C:\AI Practice\D_&_W' #Creating "input_dir" variable to create a directory, where the classes are placed
#classMaterial = ['Deadpool', 'Wolverine'] #Creating "classMaterial" variable to add the classes within the "D_&_W" folder
#-----------------------------------------------------------------



#-------------------------ADDING  IMAGES WITH LABELS USING ARRAYS--------------------
#data = [] #Creating the array variable "data" which collects the images and corresponding labels

#for classMaterials in classMaterial: #Using the for loop, to loop the data in classMaterial
    #path = os.path.join(input_dir, classMaterials) #Creating a variable "path" to use the OS library to access the class (input_dir) and combine them in the classMaterials list
    #label = classMaterial.index(classMaterials) #Creating a variable "label" which finds the posistion of the data into the classMaterial list

    #for img in os.listdir(path): #Using the for loop, to iterate the images within the "path" variable
        #imgpath = os.path.join(path, img) #Adds the full path of the data in each image file
        #classMaterial_img = cv2.imread(imgpath, 0) #Creating the "classMaterial_img" variable which reads the images in grayscale model (0)
        #try: #Using "try" block to resize the images when the machine has found each images (data) into 50x50 pixels
            #classMaterial_img = cv2.resize(classMaterial_img,(50,50)) #Creating "classMaterial_img" to resize the images by 50x50 pixels
            #image = np.array(classMaterial_img).flatten() #Creating "image" variable to convert the resized images into 1D (1 Dimensional) array 
            #data.append([image, label]) #Adds the flatten images into "data" array
        #except Exception as e: #Using "except" block to pass the errors in resizing the images 
            #pass 


#print(len(data)) #Printing the amount of images within the "data" array
#--------------------------------------------------------------------




#-------------------------WRITING THE DATASET WHICH INCLUDES THE IMAGE AND LABELS-----------------------
#pickle_in = open('dataset1.pickle', 'wb') #Creating "pickle_in" variable to open the pickle file name "dataset1" and write the binary numbers, which is used to store serialised data
#pickle.dump(data,pickle_in) #Storing the images found in the "data" array variable into the "dataset1.pickle" file
#pickle_in.close() #After it has finished reading  the file, it close the file (proventing from memeory leaks) 
#-------------------------------------------------------------------------------------------------------




#------------------------READING THE DATASET---------------------------------------------------
pickle_in = open('dataset1.pickle', 'rb') #Creating "pickle_in" variable to open the pickle file name "dataset1" and read the binary (rb)
data=pickle.load(pickle_in) #Deserialise the data from file object
pickle_in.close() #After it has finished reading  the file, it close the file (proventing from memeory leaks) 
#---------------------------------------------------------------------------------------------



#------------------------------------GENERATING THE IMAGES RANDOMLY THAT IS IN THE DATA ARRAY (WHICH CONTAINTS THE IMAGES AND LABELS)----------------------------
random.shuffle(data) #Genrating and shuffling  the data in the dataset
features = [] #Declaring a array variable, which collects images of deadpool and wolverine
labels=[] #Declaring a array variable, which collects labels (classes) of deadpool and wolverine
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------




#------------------------------ADDS GENERATED DATA IN THE FEATURE AND LABEL ARRAY----------------------------
for feature,label in data: #By using the for loop, i am separating the features and labels from the data collection
    features.append(feature) #Adding featires to the feature in data (dataset)
    labels.append(label) #Adding labels to the label in data (dataset)
#----------------------------------------------------------------------------------------------------------






#-------------------------PREPARING AND TESTING THE DATA----------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.50) #Separates the x_train and y_train to train the dataset of feature and x_test into y_test to test the dataset in feature and labels. test_size attribute will have 50% of testing the data and 50% of training the data

#model = SVC(C=1, gamma='auto',kernel='poly') #Declaring "model" variable to create the model, by using SVC (Support Vector Classification). "C=1" prevents from the data on overlifting (causing error), "gamma='auto'" which automatically selects the images and "kernal='poly'" gives higher dimentional data points
#model.fit(xtrain,ytrain) #Fitting the model variable in xtrain and ytrain 
#-----------------------------------------------------------------------------------------------------------






#-------------------------WRITING THE MODEL-----------------------
#pick = open('model.h5', 'wb') #Creating "pickle_in" variable to open the h5 file name "model" and write the binary numbers, which is used to store serialised data
#pickle.dump(model,pick) #Storing the images found in the "model" array variable into the "model.h5" file
#pick.close() #After it has finished reading  the file, it close the file (proventing from memeory leaks) 
#--------------------------------------------------------------







#------------------------------------------READING THE MODEL-----------------------------------------
pick = open('model.h5', 'rb') #Declaring a variable "pick" to open the file 'model.h5' where it s converted into binary and it reads the file (rb = read binary)
model = pickle.load(pick) #Deserialise the data from file object
pick.close() #After it has finished reading  the file, it close the file (proventing from memeory leaks) 
#----------------------------------------------------------------------------------------------------







#----------------------------------------------------------ADDING A PREDICTION AND ACCURACY OF THE DATA
prediction = model.predict(xtest) #Declaring a variable "prediction" to use a model with attribute predict to predict the dataset by using the xtest
accuracy = model.score(xtest, ytest) #Desclaring a variable "accuracy" to use a model with score attribute to give accuracy of the dataset that AI has predicted by using the xtest and ytest

classMaterials = ['Deadpool', 'Wolverine'] #Declaring an array "classMaterials", which represents the classes 

print('Accuracy:', accuracy) #AI gives the accuracy score that AI has predicted
print('The AI has guessed: ', classMaterials[prediction[0]]) #AI gives the result of the material by using the material variable
classMaterial = xtest[0].reshape(50,50) #Takes the data stored in xtest (1 dimentional array) and it rearranges the grid with 50 rows and 50 columns 
plt.imshow(classMaterial, cmap='gray') #Display a image of the material in gray colour (cmap)
plt.show() #Shows the image
#-------------------------------------------------------------------------------------------------



