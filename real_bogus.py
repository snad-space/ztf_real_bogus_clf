import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn import metrics
from tqdm import tqdm
import requests
import argparse



def load_single(oid_filename, feature_filename):
    oid     = np.memmap(oid_filename, mode='c', dtype=np.uint64)
    feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)

    return oid, feature

def make_argument_parser():
    parser = argparse.ArgumentParser(description='Real-bogus classification for ZTF objects')
    parser.add_argument('--oid', help='Name of the oid_XXX.dat', required=True)
    parser.add_argument('--feature', help='Name of the file with corresponding features (features_XXX.dat)', required=True)
    parser.add_argument('--feature-names', help='Name of the file with feature names, one name per line (features_XXX.name)')
    parser.add_argument('--output', help='Name of the file for saving results.', required=True)
    parser.add_argument('--concat', help='Concatenate probability column to the feature_XXX.dat or not.', type=bool, default=False)
    parser.add_argument('-s', '--random_seed', default=42, type=int, help='Fix the seed for reproducibility. Defaults to 42.')
    return parser

def parse_arguments():
    parser = make_argument_parser()
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    #Load saved dataset
    #There are objects from akb database cross-matched with https://sai.snad.space/tmp/features
    data = np.load('feature_dataset.npy', allow_pickle='TRUE').item()

    #Init and train RF model
    #Input: object features as X and label as Y
    #Output: probability that an object is artefact
    model = RandomForestClassifier(max_depth=18, n_estimators=831, random_state=42)
    score_types = ('accuracy', 'roc_auc', 'f1')

    result = cross_validate(model, data['features'], data['labels'],
                        cv=KFold(shuffle=True, random_state=42),
                        scoring=score_types,
                        return_estimator=True,
                        return_train_score=True,
                       )


    print('Scores for Random Forest Classifier:')
    for score in score_types:
        mean = np.mean(result[f'test_{score}'])
        std = np.std(result[f'test_{score}'])
        print(f'{score} = {mean:.3f} +- {std:.3f}')

    assert np.mean(result['test_accuracy']) > 0.7, 'Accuracy for trained model is too low!'

    clf = result['estimator'][0]

    oids, features = load_single(args.oid, args.feature)
    predict = clf.predict_proba(features)[:,1]

    if args.concat:
        result = np.hstack((features, np.float32(predict)))
    else:
        result = np.float32(predict)
    with open(args.output, "wb") as binary_file:
            binary_file.write(result.tobytes())

if __name__ == "__main__":
    main()


    
# def get_fields_list(features_path='path_to_the_features'):
#     """ Input: path to the features
#         Output: list of fields, which are contains in the features directory"""
    
#     all_feature_files = os.listdir(features_path)
#     fields = []
#     for file in all_feature_files:
#         if 'oid' in file:
#             fields.append(file[4:-4])
#     return fields




# """Init and train RF model
# Input: object features as X and label as Y
# Output: label or probability that an object is artefact
# """
# model = RandomForestClassifier(max_depth=18, n_estimators=831, random_state=42)
# score_types = ('accuracy', 'roc_auc', 'f1')

# result = cross_validate(model, data['features'], data['labels'],
#                         cv=KFold(shuffle=True, random_state=42),
#                         scoring=score_types,
#                         return_estimator=True,
#                         return_train_score=True,
#                        )


# print('Scores for Random Forest Classifier:')
# for score in score_types:
#     mean = np.mean(result[f'test_{score}'])
#     std = np.std(result[f'test_{score}'])
#     print(f'{score} = {mean:.3f} +- {std:.3f}')

# assert np.mean(result['test_accuracy']) < 0.7, 'Accuracy for trained model is too low!'

# clf = result['estimator'][0]


# fields = get_fields_list()

# def predict_proba_fields(fields, features_path='path_to_the_features', concat_proba=False):
#     """
#     Predict probabilities for all objects in feature_XXX.dat files. XXX - field.
#     fields - field list
#     if concat_proba=False, classifier results will be saved as rb_proba_XXX.dat (object order is the same as in feature_XXX.dat)
#     if concat_proba=True, classifier results will be concatenated to features from feature_XXX.dat
#     """
#     for field in fields:
#         field_oids = np.memmap(f'{features_path}/oid_{field}.dat', mode='r', dtype=np.uint64)
#         with open(f'{features_path}/feature_{field}.name') as f:
#             names = f.read().split()
#         dtype = [(name, np.float32) for name in names]
#         field_feature = np.memmap(f'{features_path}/feature_{field}.dat', mode='r', dtype=dtype, shape=field_oids.shape)

#         field_predict = clf.predict_proba(field_feature.tolist())[:,1]
#         if concat_proba:
            
                
#     return