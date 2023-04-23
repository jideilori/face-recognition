import pandas as pd
import numpy as np
import utils.augment_images
import utils.extract_features
# Build a classifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle


emb_df = pd.read_csv('res/face_db.csv')
x,y = emb_df.drop(['target'],axis=1), emb_df[['target']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,stratify=y, random_state=42)

print('Dataset: train=%d, test=%d' % (x_train.shape[0], x_test.shape[0]))

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(np.array(y_train).ravel())
trainy = out_encoder.transform(np.array(y_train).ravel())
testy = out_encoder.transform(np.array(y_test).ravel())
np.save('res/people.npy', out_encoder.classes_)

trainX = np.array(x_train)
testX = np.array(x_test)
model = SVC(kernel='rbf',probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
pickle.dump(model, open('res/facemodel.pkl','wb'))


