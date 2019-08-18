# load pandas for data preprocessing 
import pandas as pd


# a class for Loading and preprocessing data
class LoadData(object):
    def __init__(self, file_loc):
        self.file_loc = file_loc
    
    def __load_seperate_data(self):
        # load data using pd dataframe
        data = pd.read_csv(self.file_loc)

        features = data[data.columns[:-1]] # seperate features
        labels = data[data.columns[-1]] # seperate labels

        return features, labels
    def load_processed_data(self):
        # seperate_data into features and labels
        features, labels = self.__load_seperate_data()

        # one hot encode features
        features_with_dummies = pd.get_dummies(features)

        # onehot encode labels
        labels_with_dummies = pd.get_dummies(labels, prefix='', prefix_sep='')

        # list the root_labels
        root_labels = labels_with_dummies.columns

        # return set of labels
        return {'features':features_with_dummies, 'labels':labels_with_dummies, 'root_labels': root_labels}
