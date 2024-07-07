from catboost import CatBoostClassifier
import pandas as pd

# Make prediction
def make_pred(path_to_file):
    import pandas as pd
    from catboost import CatBoostClassifier
    input_df = pd.read_csv(path_to_file)
    s = input_df.client_id
    input_df = input_df.drop(['client_id', 'mrg_', 'pack'], axis=1)
    input_df[["регион", "использование"]] = input_df[["регион", "использование"]].fillna("other")
    fe1 = input_df.groupby("регион").size() / len(input_df)
    input_df["регион_freq"] = input_df["регион"].map(fe1)
    fe2 = input_df.groupby("использование").size() / len(input_df)
    input_df["использование_freq"] = input_df["использование"].map(fe2)
    input_df = input_df.drop(["регион", 'использование'], axis=1)
    train = pd.read_csv('train.csv')
    y = train.binary_target
    train = train.drop(['binary_target', 'client_id', 'mrg_', 'pack'], axis=1)
    train[["регион", "использование"]] = train[["регион", "использование"]].fillna("other")
    fe3 = train.groupby("регион").size() / len(train)
    train["регион_freq"] = train["регион"].map(fe3)
    fe4 = train.groupby("использование").size() / len(train)
    train["использование_freq"] = train["использование"].map(fe4)
    train = train.drop(["регион", 'использование'], axis=1)
    model = CatBoostClassifier(iterations=60, depth=10, class_weights=(1, 2), learning_rate=0.1,
                                logging_level='Silent').fit(train, y)
    # Make submission dataframe
    pred1 = model.predict(input_df)
    submission = pd.DataFrame({
        'client_id':  s,
        'preds': pred1
    })
    print('Prediction complete!')

    # Return class
    return submission
