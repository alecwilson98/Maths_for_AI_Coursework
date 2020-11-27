import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reads the dataset and removes the serial number column
df = pd.read_csv('/home/alecwilson/PycharmProjects/Maths_for_AI_Coursework/Admission_Predict_Ver1.csv')
df = df.drop(columns = 'Serial No.')

# Normalise the values
column_max = df.max()
df_norm = df/column_max
df_norm['Chance of Admit '] = (df_norm['Chance of Admit '])*0.97 # Reverses the normalisation of the probability values

# Splits the train and test sets (400/100)
train = df_norm.iloc[:400]
test = df_norm.iloc[400:]

# Defines the main linear regression class
class LinearRegression:

    def __init__(self, iterations, learning_rate, mini_batch_size, X, Y):
        # Hyper-parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size

        # Number of training examples and features
        self.m, self.n = X.shape

        # Initialising weights
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

    def fit(self):
        for i in range(self.iterations):
            self.grad_des()

        for i in range(self.iterations):
            self.stochastic()

        for i in range(self.iterations):
            self.mini_batch()

    def grad_des(self):

    def stochastic(self):

    def mini_batch(self):

    def predict(self):

    def visualisation(self):



def main():
    lr = LinearRegression()
    lr.fit()
    lr.predict()
    lr.visualisation()

if __name__ == '__main__':
    main()





