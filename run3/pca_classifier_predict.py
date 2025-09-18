# FINAL CLASSIFIER CODE FOR PREDICTION OF TESTING IMAGES AND PRODUCING RUN3.TXT
import joblib
import h5py
import numpy as np
import re

#Extract filename as a number to sort output in numerical order of labelling
def extract_number(name):
    return int(re.search(r'\d+', name).group())

#Load the trained model
classifier = joblib.load("svm_pca_model.joblib")

#Load the prepared test data feature vectors
with h5py.File("test_features.h5", "r") as f:
    features = f["features"][:]
    filenames = f["filenames"][:]

#Do the predictions
predictions = classifier.predict(features)

# Scene labels based on how we extracted it 
scene_labels = [
    'bedroom', 'coast', 'forest', 'highway', 'industrial', 'insidecity',
    'kitchen', 'livingroom', 'mountain', 'office', 'opencountry',
    'store', 'street', 'suburb', 'tallBuilding'
]

# each image name with its predicted label
answer = list(zip(filenames, predictions))
#sort in numerical order
answer.sort(key=lambda x: extract_number(x[0].decode('utf-8')))

# write to output file
with open("run3.txt", "w") as f:
    for fname, pred in answer:
        #pictures were labelled as 0_enhanced.jp so remove the _enhanced to get correct format
        clean_name = fname.decode('utf-8').replace('_enhanced', '')

        #get the scene label from the index
        class_label = scene_labels[pred]

        #Write in correct format
        f.write(f"{clean_name} {class_label}\n")

print("Predictions done and written to file.")
