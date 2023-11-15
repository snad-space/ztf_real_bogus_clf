import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn import metrics
from tqdm import tqdm
import requests
import argparse
import onnxruntime as rt
import time

def load_single(oid_filename, feature_filename):
    oid     = np.memmap(oid_filename, mode='c', dtype=np.uint64)
    feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)
    return oid, feature

def make_argument_parser():
    parser = argparse.ArgumentParser(description='Real-bogus classification for ZTF objects')
    parser.add_argument('--oid', help='Name of the file with OIDs', required=True)
    parser.add_argument('--feature', help='Name of the file with corresponding features.', required=True)
    parser.add_argument('--featurenames', help='Name of the file with feature names, one name per line.', required=True)
    parser.add_argument('--modelname', help='Trained model name.', required=True)
    parser.add_argument('--output', help='Name of the file for saving results.', required=True)
    parser.add_argument('--concat', help='Concatenate probability column to the features or not.', type=bool, default=False)
    return parser

def parse_arguments():
    parser = make_argument_parser()
    args = parser.parse_args()
    return args


def load_rbmodel(model_name):
    """
    Load real-bogus model
    Input: object features
    Output: probability that an object is artefact
    """
    sess = rt.InferenceSession(model_name, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name
    return (sess, input_name, label_name, prob_name)

def pred_from_onnx(model, data, return_label=False):
    """Prediction from onnx session"""
    session, input_name, label_name, prob_name = model
    if return_label:
        pred_label = session.run([label_name], {input_name: np.array(data).astype(np.float32)})[0]
    pred_proba = session.run([prob_name], {input_name: np.array(data).astype(np.float32)})[0]
    pred_proba = np.float32([pred[1] for pred in pred_proba])
    return (pred_proba, pred_label) if return_label else pred_proba


def main():
    args = parse_arguments()

    model = load_rbmodel(args.modelname)

    oids, features = load_single(args.oid, args.feature)
    print('Predicting labels...')
    t = time.monotonic()
    predict = pred_from_onnx(model, features)
    t = (time.monotonic() - t) / 60
    print(f'Predicted probabilities for {len(oids)} objects in {t:.0f} m')


    if not os.path.exists('expanded_features'):
        os.mkdir('expanded_features')

    with open(args.featurenames) as f:
            names = f.read().split()
    exp_names = names + ['RB_clf_proba']

    if args.concat:
        result = np.hstack((features, predict.reshape((-1,1))))
        with open(f'expanded_features/{args.output}.name', 'w') as f:
            for line in exp_names:
                f.write(f"{line}\n")
    else:
        result = predict
    
        
    with open(f'expanded_features/{args.output}.dat', "wb") as binary_file:
            binary_file.write(result.tobytes())

if __name__ == "__main__":
    main()

