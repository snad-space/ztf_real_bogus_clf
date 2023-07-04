Objects OIDs and labels (artefact/non-artefact) are taken from 'akb.ztf.snad.space.json'. The idea is to use object features from [https://sai.snad.space/tmp/features](https://sai.snad.space/tmp/features) and labels for training real-bogus classifier.

'download_features.ipynb' -- download oid_field.dat, features_field.dat, features_field.name for all fields in 'akb.ztf.snad.space.json'. 

'feature_dataset.npy' -- dictionary that contains OIDs from 'akb.ztf.snad.space.json' cross-matched with downloaded files, corresponding features and labels. (so it is data that used for training model)

'train_rf.ipynb' contains code for training RandomForestClassifier, optimizing hyperparameters using optuna framework and kfold cross-validation, converting trained model to ONNX format.
