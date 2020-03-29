import argparse
import pandas as pd
import os

from sklearn import tree
from sklearn.externals import joblib
from sklearn.ensemble import IsolationForest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_features', type=float, default=1.0)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])    

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_file = os.path.join(args.train, "nyc_taxi.csv")
    train_data = pd.read_csv(input_file, delimiter=',', usecols=["value"])
    
    print(train_data.head())

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = IsolationForest(random_state=0).fit(train_data) # TODO: pass here arguments
    
    
    
    sample_input = [[0], # very low value, expect it to be identified as outlier
                    [train_data.mean(axis=0)],  # mean value, extect it to be identifies as non-outlier
                    [10e5]]  # very high value, expect to be identified as non-outlier
    prediction = clf.predict(sample_input)
    
    print("""Sample data for prediction:
          [[0], # very low value, expect it to be identified as outlier
          [train_data.mean(axis=0)],  # mean value, expect it to be identified as non-outlier \n
          [10e5]]  # high value, expect to be identified as outlier \n
          """)
    print(prediction)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
