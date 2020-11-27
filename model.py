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



def main():
    lr = LinearRegression()


if __name__ == '__main__':
    main()





