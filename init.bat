python ./dataset_download.py -dataset teddy
python ./isotonic_regression_one_to_one.py -dataset teddy_data.csv
python ./dataset_split.py -dataset teddy_data_ir_experrs.csv -p 60:20:20
python ./dataset_scaling.py -datafiles teddy_data_ir_experrs_train.csv teddy_data_ir_experrs_val.csv teddy_data_ir_experrs_test.csv -scaler StandardScaler
python ./training.py -n teddy_experimento_1 -e 10 -dp ErrorBasedInvertedRandomDropout -runs 1 -trainset teddy_data_ir_experrs_train_scaled.csv -valset teddy_data_ir_experrs_val_scaled.csv
