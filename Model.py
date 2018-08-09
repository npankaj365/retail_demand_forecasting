import math
import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from statsmodels.tsa.arima_model import ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error 

class Model():
    def __init__(self, file):
        self.data = self.load_data(file)
        self.select_data()
        self.intrapolate_data()
        self.split_data()
        self.core_model()

    def load_data(self, file):
        csv_data = pd.read_csv(file, index_col='Day', parse_dates=['Day', 'Fiscal Week'], date_parser=lambda s: dtparser.parse(s).date())
        return csv_data

    def select_data(self):#Could make this dynamic
        self.sales_data = self.data.query('Class==10').SalesD.astype('float')

    def intrapolate_data(self):
        # Fill in missing values (some dates are missing in the index)
        start_date = self.sales_data.index[0].date()
        end_date = self.sales_data.index[-1].date()
        date_range = pd.date_range(start_date, end_date)
        self.X = self.sales_data.reindex(date_range, fill_value=0).values

    def split_data(self):
        self.X_train, self.X_test = model_selection.train_test_split(self.X, test_size = 0.2)
        print("Training Set size: {} and Test Set size: {}".format(self.X_train.size, self.X_test.size))
        self.X_train = list(self.X_train)

    def core_model(self):
        pass

    def get_mse(self, test_data, predicted_data):
        return mean_squared_error(test_data, predicted_data)

    def get_mae(self, test_data, predicted_data):
        return mean_absolute_error(test_data, predicted_data)


class ARIMA_Model(Model):

    def __init__(self, file):
        super().__init__(file)

    def core_model(self):
        predicted = []
        error = []
        for i in range(len(self.X_test)):
            # Rebuild Model every iteration
            model = ARIMA(self.X_train, order = (5,1,0)).fit(disp=0)

            # Predict test data
            predicted.append(model.forecast()[0])
            
            # Update training data
            self.X_train.append(self.X_test[i])
            
            #Calculate error
            error.append(math.fabs(predicted[i][0] - self.X_test[i]))

            print('%f, predicted=%f, expected=%f, error=%f' % (i, predicted[i][0], self.X_test[i], error[i]))

        self.mse = self.get_mse(self.X_test, predicted)
        self.mae = self.get_mae(self.X_test, predicted)
        
        model.save('arima_model.pkl')
        print("Mean Squared error = {:.4f}".format(self.mse))
        print("Mean Absolute error = {:.4f}".format(self.mae))


#Starter
file = 'SalesData.csv'
m = ARIMA_Model(file)
print(m.mse)

