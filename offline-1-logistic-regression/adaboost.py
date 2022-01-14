import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.impute import SimpleImputer

def convert(x):
    if x > 0:
        return 1
    return -1

def binarize(df, col_list):
    for col in col_list:
        uniq = df[col].unique()
        
        df[col] = df[col].apply(lambda x: 0 if x == uniq[0] else 1)

def normalize(df, col_list):
    scaler = MinMaxScaler()
#     scaler = StandardScaler()
    df.loc[:, col_list] = scaler.fit_transform(df.loc[:, col_list])
    # df = scaler.fit_transform(df)

def transform_data(df, bin_columns, onehot_columns, value_columns):
    binarize(df, bin_columns)
    df = pd.get_dummies(df, columns=onehot_columns)
    normalize(df, value_columns)
    return df

def preprocess_1():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    nan_v = [i for i in range(len(df.TotalCharges)) if np.isnan(df.TotalCharges[i])]
    for i in nan_v:
        df.at[i, 'TotalCharges']=  0

    df.loc[nan_v]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = np.array([convert(yi) for yi in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    bin_columns = [col for col in X if X[col].nunique() == 2]
    onehot_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']
    value_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    X_train = transform_data(X_train, bin_columns, onehot_columns, value_columns)
    X_test = transform_data(X_test, bin_columns, onehot_columns, value_columns)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def preprocess_2():
    column_heads = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df_train = pd.read_csv("adult.csv")
    df_train.columns = column_heads

    df_test = pd.read_csv("adult.test.csv")
    df_test.columns = column_heads

    m_train = df_train.shape[0]
    m_test = df_test.shape[0]

    df = pd.concat([df_train, df_test])

    del df_train
    del df_test

    df["income"].replace([' <=50K', ' >50K', ' <=50K.', ' >50K.'], [-1, 1, -1, 1], inplace=True)
    df.replace(" ?", np.nan, inplace=True)
    simputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    df = pd.DataFrame(simputer.fit_transform(df), columns=df.columns, index=df.index)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X.drop("education-num", axis=1, inplace=True)
    X["sex"].replace([' Male', ' Female'], [0, 1], inplace=True)
    onehot_columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    X = pd.get_dummies(X, columns=onehot_columns)

    X_train = X.iloc[:m_train]
    y_train = y.iloc[:m_train]
    X_test = X.iloc[m_train:]
    y_test = y.iloc[m_train:]

    value_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    normalize(X_train, value_columns)
    normalize(X_test, value_columns)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.astype(int).reshape(-1, 1)
    y_test = y_test.values.astype(int).reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def preprocess_3():
    df = pd.read_csv("creditcard.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    df_sub = pd.concat([df.loc[df["Class"] == 1], df.loc[df["Class"] == 0].iloc[:10000, :]])
    del df

    X = df_sub.iloc[:, :-1]
    y = df_sub.iloc[:, -1]
    y.replace(0, -1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def meanloss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def h(X, w):
    return np.tanh(X @ w)

def gradient(X, y, y_hat, w):
    m = X.shape[0]
    dw = -(1/m) * ( X.T @ ((y - y_hat) * (1 - y_hat**2)) )
    return dw

def logistic_regression(X, y, bs, epochs, lr):
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.
    m, n = X.shape
    
    # Initializing weights to zeros.
    w = np.zeros((n+1,1))
#     w = np.full((n+1, 1), 1/m)
    
    X_train = np.concatenate((np.ones((m, 1)), X), axis=1)
    
    # Empty list to store losses.
    losses = []
    
    for epoch in range(epochs):
        for i in range((m-1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            
            Xb = X_train[start_i:end_i]
            yb = y[start_i:end_i]
            
            y_hat = h(Xb, w)
            
            dw = gradient(Xb, yb, y_hat, w)
            
            w -= lr * dw
            
#             print(f"Epoch {epoch}: [{start_i}:{end_i}]: {w.T}")
        
        # Calculating loss and appending it in the list.
        l = meanloss(y, h(X_train, w))
        losses.append(l)
    
    return w, losses

def predict(X_test, w, returns_prob=False):
    m, n = X_test.shape
    X_test = np.concatenate((np.ones((m, 1)), X_test), axis=1)
    pred = h(X_test, w)
    
    if returns_prob:
        return pred
    
#     pred = np.squeeze(pred)
    return np.array([convert(pi) for pi in pred.reshape(-1)]).reshape(-1, w.shape[1])

def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)

def adaboost(X_train, y_train, K):
    m, n = X_train.shape
    pw = np.array([1/m] * m)
    ws = np.zeros((K, (n+1)))
    z = np.zeros(K)
    
    examples = np.arange(m)

    for k in range(K):
        data = np.random.choice(examples, (m,), p=pw)
        Xk_train = X_train[data, :]
        yk_train = y_train[data, :]
        
        # w, losses = logistic_regression(Xk_train, yk_train, Xk_train.shape[0], 20, 0.1)
        w, losses = logistic_regression(Xk_train, yk_train, 100, 20, 0.1)
        yk_pred = predict(Xk_train, w)
        print(f"k={k}, accuracy={accuracy(yk_train, yk_pred) * 100}%")
        
        ws[k] = np.squeeze(w)
        
        y_hat = predict(Xk_train, w)
        
        mask = np.squeeze((y_train != y_hat))
        error = np.sum(pw[mask])
        
        if error > 0.5:
            continue
            
        mask = ~mask
        pw[mask] = pw[mask] * error / (1-error)
        
        pw = pw / np.sum(pw)
        
        z[k] = np.log((1-error) / error)
    
    return ws, z

def weighted_majority(X, ws, z):
    preds = predict(X, ws.T)
    y_hat = preds @ z.reshape(-1, 1)
    return np.array([convert(pi) for pi in y_hat.reshape(-1)]).reshape(-1, 1)

def report(y, y_pred):
    acc = accuracy(y, y_pred)
    [[TN, FP], [FN, TP]] = cm(y, y_pred)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    false_discovery_rate = FP / (FP + TP)
    f1 = 2*TP / (2*TP + FP + FN)
    
    print(f"Accuracy: {acc}")
    print(f"True positive rate (sensitivity, recall, hit rate): {recall}")
    print(f"True negative rate (specificity): {specificity}")
    print(f"Positive predictive value (precision): {precision}")
    print(f"False discovery rate: {false_discovery_rate}")
    print(f"F1 score: {f1}")
    print()


def run_logistic(X_train, y_train, X_test, y_test):
    w, losses = logistic_regression(X_train, y_train, 100, 20, 0.1)
    y_pred = predict(X_train, w)
    report(y_train, y_pred)

    y_pred = predict(X_test, w)
    report(y_test, y_pred)

def run_adaboost(X_train, y_train, X_test, y_test, K):
    ws, z = adaboost(X_train, y_train, K)

    # Using 1 and -1
    y_pred = weighted_majority(X_train, ws, z)
    report(y_train, y_pred)

    y_pred = weighted_majority(X_test, ws, z)
    report(y_test, y_pred)

# X_train, X_test, y_train, y_test = preprocess_1()
# X_train, X_test, y_train, y_test = preprocess_2()
# X_train, X_test, y_train, y_test = preprocess_3()

# run_logistic(X_train, y_train, X_test, y_test)

# K = 10
# run_adaboost(X_train, y_train, X_test, y_test, K)

# for K in [5, 10, 15, 20]:
#     print(f"K = {K}")
#     run_adaboost(X_train, y_train, X_test, y_test, K)