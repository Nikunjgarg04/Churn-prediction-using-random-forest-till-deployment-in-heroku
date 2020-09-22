#Let's start with importing necessary libraries
import pickle
import numpy as np
import pandas as pd

class predObj:

    def predict_log(self, dict_pred):


        with open("random_forest_classifier_model.pkl", 'rb') as f:
            model = pickle.load(f)
        data_df = pd.DataFrame(dict_pred,index=[1,])
        predict = model.predict(data_df)
        if predict[0] ==1 :
            result = 'will churn'
        else:
            result ='will not churn'

        return result



