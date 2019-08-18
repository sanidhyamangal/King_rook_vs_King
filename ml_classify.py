from load_data import LoadData # data preprocessor class
from sklearn.model_selection import train_test_split # to split dataset

# load some model for prediction task
from sklearn.tree import DecisionTreeClassifier # decission tree classifier
from sklearn.naive_bayes import GaussianNB # gausssian naive bayes classifier
from sklearn.neighbors import KNeighborsClassifier # knn classifier
from sklearn.ensemble import RandomForestClassifier # rf classifer

# import f1 score matrix
from sklearn.metrics import f1_score

# for pickling the data
import pickle

# helper function to save the model
def save_model(learner, file_name):
    with open(file_name + '.pkl', 'wb') as fp:
        pickle.dump(learner, fp)

# helper function to train and predict
def train_predict(learner, X_train, y_train, X_test, y_test):
    results = {}

    # set learner to learner
    learner = learner

    # print training model
    print("Training model {}". format(learner.__class__.__name__))
    learner.fit(X_train, y_train)

    # print sabing the model
    print("Saving model {}". format(learner.__class__.__name__))
    save_model(learner, learner.__class__.__name__)
    
    # predict model
    print("Predicting Model")
    prediction_test = learner.predict(X_test)

    results['accu'] = f1_score(y_test, prediction_test, average='micro')

    return results

# load data
dataLoader = LoadData('./kr-vs-k.csv') # data loader for kr-vs-k

data = dataLoader.load_processed_data() # load preprocessed data

# split data
X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=42)

# initalize classifiers
clf_DT = DecisionTreeClassifier() # decission tree classifier
clf_NB = GaussianNB() # GaussianNB classifier
clf_KNN = KNeighborsClassifier(n_neighbors=4) # KNN classifier
clf_RF = RandomForestClassifier(n_estimators=1000) # Random Forest with 100 trees

# list of classifiers to iterate
classifier_list = [clf_DT, clf_KNN, clf_RF]

# a results dict for measuring results
results = dict()

for clf in classifier_list:
    clf_name = clf.__class__.__name__
    results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)


# save results into pickle file
with open('results.pkl', 'wb') as fp:
    pickle.dump(results, fp)

print("Results: \n", results)