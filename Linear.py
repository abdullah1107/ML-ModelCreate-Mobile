#import library
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
#from sklearn.tree import _decision_tree_classifier
#from ._converter_internal import _convert_sklearn_model
#read the dataset y=f(X)
dataset = pd.read_csv('Salary_Data.csv')
# X = dataset.iloc[:,:-1].values  #read X without last colum
# y = dataset.iloc[:,1].values    #read y second coloum
#
#
# #Split the dataset into training and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# #Predicting the Test Result
# y_pred = regressor.predict(X_test)
#
# print(y_pred)



# from sklearn.linear_model import LinearRegression
# import pandas as pd
#
# # Load data
# data = pd.read_csv('houses.csv')

# Train a model
model = LinearRegression()
model.fit(dataset[["YearsExperience"]], dataset["Salary"])

# Convert and save the scikit-learn model
import coremltools as ct

model = ct.converters.sklearn.convert(model, ["YearsExperience"], "Salary")


# Set model metadata
model.author = 'Mamun'
model.license = 'MIT'
model.short_description = 'Predicts the Salary of an employee:'
model.version = '1'

# Set feature descriptions manually
model.input_description['YearsExperience'] = 'Years of Experience'
# model.input_description['bathrooms'] = 'Number of bathrooms'
# model.input_description['size'] = 'Size (in square feet)'

# Set the output descriptions
model.output_description['Salary'] = 'Salary of the Employee'

# Save the model
model.save('salary.mlmodel')

#Visualise the Training set results
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary Vs Experience (Training Set)')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()


#Visualise the Test set results
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_test, y_pred, color = 'blue')
# plt.title('Salary Vs Experience (Test Set)')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()
