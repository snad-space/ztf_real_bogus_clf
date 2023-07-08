import time
import sys
import argparse
import requests
from pprint import pprint
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn import metrics
import pandas as pd

API_URL_EXTR = 'http://features.lc.snad.space/api/latest'

def make_argument_parser():
    parser = argparse.ArgumentParser(description='Train real-bogus classification model for ZTF objects')
    parser.add_argument('--featurenames', help='Name of the file with feature names, one name per line', default='snad6_features/feature_snad6_r_100.name')
    parser.add_argument('--extrname', help='Name of the extractor file', default='feat_extr_snad6.json')
    parser.add_argument('--modelname', help='Name for trained model.', default='rf_RBclf.onnx')
    parser.add_argument('--d', help='Download LCs for akb objects or not.', type=bool, default=False)
    parser.add_argument('--extr', help='Extract features for akb objects or not.', type=bool, default=False)
    parser.add_argument('--lcdir', help='Directory for saving LCs.', default='LCs')
    parser.add_argument('-s', '--random_seed', default=42, type=int, help='Fix the seed for reproducibility. Defaults to 42.')
    return parser

def parse_arguments():
    parser = make_argument_parser()
    args = parser.parse_args()
    return args


def download_lc(oid, dir):
    url = 'http://db.ztf.snad.space/api/v3/data/dr4/oid/full/json?oid='
    with requests.get(f'{url}{oid}') as response:
        response.raise_for_status()
        open(f'{dir}/{oid}.json', 'wb').write(response.content)


def lc_from_json(oid, dir='LCs'):
    path = f'{dir}/{oid}.json'
    file = open(path)
    data = json.load(file)[f'{oid}']['lc']
    file.close()
    light_curve = [dict(t=obs['mjd'], m=obs['mag'], err=obs['magerr']) for obs in data]
    return light_curve

def magn_to_flux(light_curve, zp):
    flux_lc = []
    for obs in light_curve:
        flux = 10**(-0.4 * (obs['m'] - zp))
        # |d flux / d magn| = 0.4 * ln(10) * flux
        flux_sigma = 0.4 * np.log(10) * flux * obs['err']
        flux_lc.append(dict(t=obs['t'], m=flux, err=flux_sigma))
    return flux_lc

def get_feat_single(magn_lc, extractors):
    data_magn = dict(
                        light_curve=magn_lc,
                        extractor=extractors['magn']
                    )
    
    flux_lc = magn_to_flux(magn_lc, zp=extractors['flux']['zero_point'])
    data_flux = dict(
                        light_curve=flux_lc,
                        extractor=extractors['flux']['extractor']
                    )
    resp = requests.post(f'{API_URL_EXTR}/features', json=data_magn)
    resp.raise_for_status()
    magn_features = resp.json()

    resp = requests.post(f'{API_URL_EXTR}/features', json=data_flux)
    resp.raise_for_status()
    flux_features = resp.json()

    return magn_features, flux_features


def make_feature_vector(magn_lc, flux_lc, feature_names):
    feature_vector = []
    for name in feature_names:
        if name[-6:-2] == 'magn':
            feature_vector.append(magn_lc[name[:-7]])
        elif name[-6:-2] == 'flux':
            feature_vector.append(flux_lc[name[:-7]])
    return feature_vector


def get_oids(filepath):
    file = open(filepath)
    obj_list = json.load(file)
    file.close()

    oids = []
    tags = []
    for data in obj_list:
        oids.append(data['oid'])
        tags.append(data['tags'])

    targets = [] # 1-artefact,  0-transient
    for tag_list in tags:
        if 'artefact' in tag_list:
            targets.append(1)
        else:
            targets.append(0)
    
    return oids, targets


def convert_to_onnx(model, input_shape, name):    
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(name, "wb") as f:
        f.write(onx.SerializeToString())


def get_feat(oids, labels, extractors, names, args):
    data = []
    t = time.monotonic()
    for oid, label in zip(oids, labels):
        try:
            magn_lc = lc_from_json(oid, dir=args.lcdir)
            magn_features, flux_features = get_feat_single(magn_lc, extractors)
            feat_vec = make_feature_vector(magn_features, flux_features, names)
            data.append([oid, label] + feat_vec)
        except:
            continue
    print(f'{len(oids) - len(data)} objects have problems with extracting features')
    all_features = pd.DataFrame(data=data, columns=['oid', 'label'] + names)
    
    all_features.to_csv('akb_features.csv', index=False)
    t = (time.monotonic() - t) / 60
    print(f'Features for {len(data)} extracted in {t:.0f} m')
    return all_features



def main():
    args = parse_arguments()

    #Download LCs for akb objects if needed
    if args.d:
        url = f'https://akb.ztf.snad.space/objects/'
        with requests.get(url) as response:
            response.raise_for_status()
            open(f'akb.ztf.snad.space.json', 'wb').write(response.content)
        oids, labels = get_oids('akb.ztf.snad.space.json')
        
        t = time.monotonic()
        for oid in oids:
            download_lc(oid, dir=args.lcdir)
        t = (time.monotonic() - t) / 60
        print(f'LCs downloaded in {t:.0f} m')

    oids, labels = get_oids('akb.ztf.snad.space.json')
    #extractors
    file = open(args.extrname)
    extractors = json.load(file)
    file.close()

    with open(args.featurenames) as f:
            names = f.read().split()

    #Extract features from LCs for all akb objects
    if args.extr:
        data = get_feat(oids, labels, extractors, names, args)
    else:
        data = pd.read_csv('akb_features.csv')

    # Train and validate real-bogus model
    t = time.monotonic()
    model = RandomForestClassifier(max_depth=18, n_estimators=831, random_state=args.random_seed)
    score_types = ('accuracy', 'roc_auc', 'f1')

    result = cross_validate(model, data[names], data['label'],
                        cv=KFold(shuffle=True, random_state=args.random_seed),
                        scoring=score_types,
                        return_estimator=True,
                        return_train_score=True,
                       )

    print('Scores for Random Forest Classifier:')
    for score in score_types:
        mean = np.mean(result[f'test_{score}'])
        std = np.std(result[f'test_{score}'])
        print(f'{score} = {mean:.3f} +- {std:.3f}')
    t = (time.monotonic() - t) / 60
    print(f'RF trained (with cross-validation) in {t:.0f} m')
    
    assert np.mean(result['test_accuracy']) > 0.7, 'Accuracy for trained model is too low!'
    clf = result['estimator'][0]

    convert_to_onnx(clf, len(names), name=args.modelname)



if __name__ == "__main__":
    main()