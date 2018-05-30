import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# input data from train.csv
def inputDataSet():
  train_data = pd.read_csv('train.csv',dtype={'x':np.float32,'y':np.float32,'accuracy':np.int16,
                                              'time':np.int,'place_id':np.int},index_col = 0)
  return train_data

# processing dataset; removing the place_id that appear less than 3 times in train data to speed up computation and
# reduce the number of output classes that we have; extracting Hours, Day, weekday, month from time.
def preprocessData(train_data):

  place_count = train_data.place_id.value_counts()
  mask = place_count[train_data.place_id.values] > 500
  train_data = train_data.loc[mask.values]

  hour = []
  weekday = []
  month = []
  year = []
  day = []

  for i in train_data.time.values:
    hour.append((i/60)%24)
    weekday.append((i/(60*24))%7)
    month.append((i/(60*24*30)) % 12)
    year.append(i/(60*24*365))
    day.append(i/(60*24) % 365)

  train_data['hour'] = hour
  train_data['weekday'] = weekday
  train_data['month'] = month
  train_data['year'] = year
  train_data['day'] = day
  train_data = train_data.drop('time', 1)
  return train_data

# creating test and train data set randomly 90% traindata and 10% testdata
def createDataset(df):
  msk = np.random.rand(len(df)) < 0.9
  train = df[msk]
  test = df[~msk]
  return train, test

# writing the created train and test data into seperate files to avoid redundant processing
def writeProcessedData(train_data, test_data):
  train_data.to_csv('train1.csv')
  test_data.to_csv('test1.csv')

# reading the data from the created files written earlier after processing
def readProcessedData():
  train_data = pd.read_csv('train1.csv',dtype = {'x':np.float32,'y':np.float32,'accuracy':np.int16,
                                               'hour':np.float32, 'weekday':np.float32, 'month':np.float32,
                                               'year':np.float32, 'day':np.float32, 'place_id':np.int},index_col = 0)
  test_data = pd.read_csv('test1.csv',dtype = {'x':np.float32,'y':np.float32,'accuracy':np.int16,
                                               'hour':np.float32, 'weekday':np.float32, 'month':np.float32,
                                               'year':np.float32, 'day':np.float32, 'place_id':np.int},index_col = 0)
  return train_data, test_data


# Creating data matrix for fast computation of grid data for clustering
def divideData(train_data):
    df_matrix = [[list() for x in range(101)] for y in range(101)]

    print "Creating Matrix..."
    print len(train_data)
    index = 0
    for i, row in train_data.iterrows():
        df_matrix[int(row[0]*10)][int(row[1]*10)].append(list(row))
        # train_data.drop(i, inplace=True)
        print index
        index += 1
    print "##### DONE #####"

    return df_matrix

# Applying KNN-Classifier
def classifyRowKNN(row, train_data):

    clf = KNeighborsClassifier(n_neighbors=len(train_data)/2, weights='distance',
                               metric='manhattan')

    X = train_data.drop('place_id', 1)
    clf.fit(X, train_data['place_id'])
    del row[-1]
    del row[3]
    y_pred = clf.predict(row)

    return y_pred

# Applying Random Forest
def classifyRowRandomForest(row, train_data):

    clf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1,
                                 min_samples_split=4,random_state=0)

    X = train_data.drop('place_id', 1)
    clf.fit(X, train_data['place_id'])
    del row[-1]
    del row[3]
    y_pred = clf.predict(row)

    return y_pred

# Applying Boosted trees classifier using XGBoost
def classifyRowBoostedTree(row, train_data):
    #Applying the classifier
    clf = xgb(loss='deviance',n_estimators=150, max_depth=None, min_samples_split=2, random_state=None)
    X = train_data.drop('place_id', 1)
    clf.fit(X, train_data['place_id'])
    del row[-1]
    del row[3]
    y_pred = clf.predict(row)
    return y_pred

def main():

  # # only run the folowing first time to generate the train and test files

  # train_data = inputDataSet()
  # train_data = preprocessData(train_data)
  # train_data, test_data = createDataset(train_data)
  # writeProcessedData(train_data, test_data)
  # # end

  # Read data from pre-written files
  train_data, test_data = readProcessedData()

  # Make new column to store predicted value for each column in test data

  predicted = []
  for i in test_data.x.values:
      predicted.append(-1)
  test_data['predicted'] = predicted
  # print test_data.head(n=10)

  # Divide data in matrix using grid formation for faster computation

  df_list = divideData(train_data)
  del train_data

  print len(test_data)
  counter = 0

  # Do prediction for each row in test data
  for row in test_data.values:
    x = int(row[0]*10)
    y = int(row[1]*10)

    new_train = list()
    if x == 0 or y == 0 or x == 100 or y == 100:
        new_train.extend(df_list[x][y])
    else:
        new_train.extend(df_list[x][y])
        new_train.extend(df_list[x-1][y])
        new_train.extend(df_list[x+1][y])
        new_train.extend(df_list[x][y-1])
        new_train.extend(df_list[x][y+1])
        new_train.extend(df_list[x+1][y+1])
        new_train.extend(df_list[x-1][y-1])
        new_train.extend(df_list[x+1][y-1])
        new_train.extend(df_list[x-1][y+1])

    print "CLASSIFY A ROW..."
    cols =['x','y','accuracy', 'place_id', 'hour', 'weekday', 'month', 'year', 'day']
    new_df = pd.DataFrame(new_train, columns=cols)
    pred = classifyRowKNN(list(row), new_df)
    print counter
    counter += 1

    row[-1] = pred
    # print row

    # Naive method of creating grid around location each time row is presented to us

    # xLow = row[0] - 0.01
    # yLow = row[1] - 0.01
    # xHigh = row[0] + 0.01
    # yHigh = row[1] + 0.01
    # new_train = pd.DataFrame(columns=['x','y','accuracy', 'place_id', 'hour', 'weekday', 'month', 'year', 'day'])
    # index = 0
    # print "PREPARE NEW TRAIN DATA..."
    # for r in train_data.values:
    #     if (xHigh > r[0] > xLow) and (yHigh > r[1] > yLow):
    #         new_train.loc[index] = list(r)
    #         index += 1
    # print "##### DONE #####"
    #
    # print "CLASSIFY A ROW..."
    # cols =['x','y','accuracy', 'place_id', 'hour', 'weekday', 'month', 'year', 'day']
    # new_df = pd.DataFrame(new_train, columns=cols)
    # pred = classifyRowRandomForest(list(row), new_df)
    #
    # print counter
    # counter += 1
    # print "##### DONE #####"

  # Writing results to a file

  test_data.to_csv('testResult.csv')
  # print test_data.head(n=10)

if __name__ == '__main__':
  main()