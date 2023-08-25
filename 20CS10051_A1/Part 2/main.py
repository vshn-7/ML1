
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


### spliting datasets
def splitting_data(df_insurance):
    training_data = df_insurance.sample(frac=0.8,random_state=100)
    testing_data = df_insurance.drop(training_data.index)
    return training_data, testing_data

### detecting outliers and removing them
def outliers(df_insurance, columns):
    limit =  [0]*(len(columns))
    for i in range(1, len(columns)):
        limit[i] = df_insurance[columns[i]].mean() + (3 * df_insurance[columns[i]].std())
    for j in range(1, len(columns)):
        index = df_insurance[(df_insurance[columns[i]]) > limit[j]].index
    if len(index) != 0:
        #print("OK")
        df_insurance.drop(index, inplace=True)
    return


def if_value_null(df_insurance, columns):

    for i in range(1,len(columns)-1):
        for j in range(0,df_insurance.shape[0]):
            if df_insurance.loc[j][columns[i]] == None:
                df_insurance.loc[j][columns[i]] = df_insurance[columns[i]].mean()
                #print("OK")


def k_fold_train(training,laplace):
    num_folds = 10
    subset_size = len(training) / num_folds
    all_accuracy = [0]*num_folds
    training = training.reindex(np.random.permutation(training.index))
    training = training.reset_index(drop=True)
    fold = [0]*10
    for i in range(num_folds):
        fold[i] = training.loc[i*subset_size:((i+1)*subset_size)-1]

    train_val = [0]*10
    train_val[0] = training.drop(fold[0].index)
    train_val[1] = training.drop(fold[1].index)
    train_val[2] = training.drop(fold[2].index)
    train_val[3] = training.drop(fold[3].index)
    train_val[4] = training.drop(fold[4].index)
    train_val[5] = training.drop(fold[5].index)
    train_val[6] = training.drop(fold[6].index)
    train_val[7] = training.drop(fold[7].index)
    train_val[8] = training.drop(fold[8].index)
    train_val[9] = training.drop(fold[9].index)

    for i in range(num_folds):
        y_train = train_val[i]["Response"]
        x_train = train_val[i].drop("Response", axis=1)

        y_test = fold[i]["Response"]
        x_test = fold[i].drop("Response", axis=1)

        means = train_val[i].groupby(["Response"]).mean()                                               # Find mean of each class

        var = train_val[i].groupby(["Response"]).var()                                                  # Find variance of each class
        prior = ((train_val[i].groupby("Response").count() ) / (len(train_val[i]) )).iloc[:, 1]                # Find prior probability of each class    len(train_val[i].columns)
        classes = np.unique(train_val[i]["Response"].tolist())                                          # Storing all possible classes

        pred_x_test = predict(x_test, means, var, prior, classes, x_test,laplace)

        all_accuracy[i] = round(100*accuracy(y_test, pred_x_test), 5)
        #print(round(100*Accuracy(y_train, pred_x_train), 5))
        print(all_accuracy[i])
    max_value = max(all_accuracy)
    index = all_accuracy.index(max_value)

    return train_val[index]


def normal(x, mu, var):                                       #

    stdev = np.sqrt(var)
    pdf = (np.e ** (-0.5 * ((x - mu) / stdev) ** 2)) / (stdev * np.sqrt(2 * np.pi))

    return pdf


def predict(X, means, var, prior, classes, x_train, laplace):
    predictions = []
    #print(x_train)
    #print(X.loc[3])
    #print(X)
    for i in X.index:                                           # Loop through each instances

        classlikelihood = []
        instance = X.loc[i]
        #print(classes)
        for cls in classes:                                     # Loop through each class

            featurelikelihoods = []
            featurelikelihoods.append(np.log(prior[cls]))       # Append log prior of class 'cls'

            for col in x_train.columns:                         # Loop through each feature

                data = instance[col]
                #print(data)
                mean = means[col].loc[cls]                      # Find the mean of column 'col' that are in class 'cls'
                variance = var[col].loc[cls]                    # Find the variance of column 'col' that are in class 'cls'
                #print("OK")
                likelihood = normal(data, mean, variance)

                if likelihood != 0:
                    likelihood = np.log((likelihood + laplace)/ 1 + laplace)             # Finding the log-likelihood evaluated at x
                else:
                    likelihood = 1 / len(train)

                featurelikelihoods.append(likelihood)

            totallikelihood = sum(featurelikelihoods)            # Calculate posterior
            classlikelihood.append(totallikelihood)

        MaxIndex = classlikelihood.index(max(classlikelihood))  # Find the largest posterior position
        prediction = classes[MaxIndex]
        predictions.append(prediction)
    return predictions


def accuracy(y, prediction):
    y = list(y)
    prediction = list(prediction)
    score = 0

    for i, j in zip(y, prediction):
        if i == j:
            score += 1

    return score / len(y)


if __name__ == "__main__":
    #### reading data from CSV

    df_insurance = pd.read_csv(r'Dataset_C.csv')
    df_orig = df_insurance
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    le = LabelEncoder()
    df_insurance['Gender'] = le.fit_transform(df_insurance['Gender'])
    df_insurance['Vehicle_Age'] = le.fit_transform(df_insurance['Vehicle_Age'])
    df_insurance['Vehicle_Damage'] = le.fit_transform(df_insurance['Vehicle_Damage'])

    outliers(df_insurance, df_insurance.columns)

    df_insurance = df_insurance.drop("id",axis=1)
    print(df_insurance.head(1))
    f = open("output.txt", "w")
    f.write("Final Set of Features with first row:\n")
    f.write(str(df_insurance.head(1)))

    train, test = splitting_data(df_insurance)
    #print(len(train))

    laplace = 1
    k_fold_train_ds = k_fold_train(train,laplace)

    k_fold_train_y = k_fold_train_ds["Response"]
    k_fold_train_x = k_fold_train_ds.drop("Response",axis=1)

    df_means = k_fold_train_ds.groupby(["Response"]).mean()                                                          # Find mean of each class
    df_var = k_fold_train_ds.groupby(["Response"]).var()                                                             # Find variance of each class
    df_prior = ((k_fold_train_ds.groupby("Response").count() ) / ((len(k_fold_train_ds)))).iloc[:, 1]                # Find prior probability of each class      len(k_fold_train_ds.columns)
    df_classes = np.unique(k_fold_train_ds["Response"].tolist())                                                     # Storing all possible classes


    y_test = test["Response"]
    x_test = test.drop("Response",axis=1)

    org_y_train = train["Response"]
    org_x_train = train.drop("Response",axis=1)

    PredictTest = predict(x_test, df_means, df_var, df_prior, df_classes, x_test,laplace)
    final_acc = accuracy(y_test, PredictTest)
    f.write("\n\nFinal Accuracy is:")
    f.write(str(final_acc))
    f.close()
    print("Accuracy:", round(100 * final_acc, 5))


