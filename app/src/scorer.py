from catboost import CatBoostClassifier
import pandas as pd


def preparing_data(path_to_file):
    # Get input dataframe
    input_df = pd.read_csv(path_to_file)
    df = input_df.drop(['binary_target', 'client_id', 'mrg_', 'pack'], axis=1)
    df[["регион", "использование"]] = df[["регион", "использование"]].fillna("other")
    return df

# Make prediction
def make_pred(dt, path_to_file):
    import pandas as pd
    train = pd.read_csv('train.csv')
    y = train.binary_target
    train = train.drop(['binary_target', 'client_id', 'mrg_', 'pack'], axis=1)
    train[["регион", "использование"]] = train[["регион", "использование"]].fillna("other")
    model = CatBoostClassifier(iterations=60, depth=10, class_weights=(1,2), learning_rate=0.1,
       logging_level='Silent', cat_features=["регион", "использование"]).fit(train, y)

    f_imp_list3 = list(zip(train.columns, model.feature_importances_))
    f_imp_list3.sort(key=lambda x: x[1], reverse=True)
    list2 = f_imp_list3[:5]
    dict1 = dict(list2)
    print(dict1)

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': model.predict(dt)
    })
    print('Prediction complete!')


    # Return class
    return submission
