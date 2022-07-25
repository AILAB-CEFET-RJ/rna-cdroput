python -m src.strategies.dt_m_x_1 -dataset teddy_data.csv

python -m src.pipeline.dataset_split -dataset teddy_data_dt_m_x_1_experrs.csv -p 60:20:20

python -m src.pipeline.dataset_scaling -datafiles teddy_data_dt_m_x_1_experrs_train.csv teddy_data_dt_m_x_1_experrs_val.csv teddy_data_dt_m_x_1_experrs_test.csv -scaler StandardScaler

python -m src.training -n teddy_training_dt_m_x_1 -e 10 -dp ErrorBasedInvertedRandomDropout -runs 1 -trainset teddy_data_dt_m_x_1_experrs_train_scaled.csv -valset teddy_data_dt_m_x_1_experrs_val_scaled.csv
