# declare -a REGRESSORS=("dt" "knn" "mlp" "rf" "xgb")
# declare -a STRATEGIES=("1_x_1" "m_x_1" "m_x_m")
# declare -a DATASETS=("teddy" "happy")

# for dataset in "${DATASETS[@]}"
# do
#     for regressor in "${REGRESSORS[@]}"
#     do
#         for strategy in "${STRATEGIES[@]}"
#         do
#             eval "python -m src.strategies.${regressor}_${strategy} -dataset ${dataset}_data.csv"
#             eval "python -m src.strategies.${regressor}_${strategy} -dataset ${dataset}_test_data_B.csv"
#             eval "python -m src.strategies.${regressor}_${strategy} -dataset ${dataset}_test_data_C.csv"
#             eval "python -m src.strategies.${regressor}_${strategy} -dataset ${dataset}_test_data_D.csv"

#             eval "python -m src.pipeline.dataset_split -dataset ${dataset}_data_${regressor}_${strategy}_experrs.csv -p 80:20"

#             eval "python -m src.pipeline.dataset_scaling -datafiles ${dataset}_data_${regressor}_${strategy}_experrs_train.csv ${dataset}_data_${regressor}_${strategy}_experrs_val.csv -scaler StandardScaler"

#             eval "python -m src.training -n ${dataset}_training_${regressor}_${strategy} -e 5000 -dp ErrorBasedInvertedRandomDropout -runs 3 -trainset ${dataset}_data_${regressor}_${strategy}_experrs_train_scaled.csv -valset ${dataset}_data_${regressor}_${strategy}_experrs_val_scaled.csv"

#             BEST_MODELS=$(python -m src.modules.best_model_selector -n ${dataset}_training_${regressor}_${strategy} -e 5000)

#             eval "python -m src.predict -n ${dataset}_B -models ${BEST_MODELS} -testset ${dataset}_test_data_B_${regressor}_${strategy}_experrs.csv -dp y"
#             eval "python -m src.predict -n ${dataset}_C -models ${BEST_MODELS} -testset ${dataset}_test_data_C_${regressor}_${strategy}_experrs.csv -dp y"
#             eval "python -m src.predict -n ${dataset}_D -models ${BEST_MODELS} -testset ${dataset}_test_data_D_${regressor}_${strategy}_experrs.csv -dp y"
#         done
#     done
# done

### ISOTONIC REGRESSOR ###
# ISOTONIC REGRESSOR DOES NOT SUPPORT MULTIPLE INPUT/OUTPUT!

### TEDDY ###
python -m src.strategies.ir_1_x_1 -dataset teddy_data.csv
python -m src.strategies.ir_1_x_1 -dataset teddy_test_data_B.csv
python -m src.strategies.ir_1_x_1 -dataset teddy_test_data_C.csv
python -m src.strategies.ir_1_x_1 -dataset teddy_test_data_D.csv

python -m src.pipeline.dataset_split -dataset teddy_data_ir_1_x_1_experrs.csv -p 80:20

python -m src.pipeline.dataset_scaling -datafiles teddy_data_ir_1_x_1_experrs_train.csv teddy_data_ir_1_x_1_experrs_val.csv -scaler StandardScaler

python -m src.training -n teddy_training_ir_1_x_1 -e 5000 -dp ErrorBasedInvertedRandomDropout -runs 3 -trainset teddy_data_ir_1_x_1_experrs_train_scaled.csv -valset teddy_data_ir_1_x_1_experrs_val_scaled.csv

BEST_MODELS=$(python -m src.modules.best_model_selector -n teddy_training_ir_1_x_1 -e 5000)

eval "python -m src.predict -n teddy_B -models ${BEST_MODELS} -testset teddy_test_data_B_ir_1_x_1_experrs.csv -dp y"
eval "python -m src.predict -n teddy_C -models ${BEST_MODELS} -testset teddy_test_data_B_ir_1_x_1_experrs.csv -dp y"
eval "python -m src.predict -n teddy_D -models ${BEST_MODELS} -testset teddy_test_data_B_ir_1_x_1_experrs.csv -dp y"

# ### HAPPY ###
# python -m src.strategies.ir_1_x_1 -dataset happy_data.csv
# python -m src.strategies.ir_1_x_1 -dataset happy_test_data_B.csv
# python -m src.strategies.ir_1_x_1 -dataset happy_test_data_C.csv
# python -m src.strategies.ir_1_x_1 -dataset happy_test_data_D.csv

# python -m src.pipeline.dataset_split -dataset happy_data_ir_1_x_1_experrs.csv -p 80:20

# python -m src.pipeline.dataset_scaling -datafiles happy_data_ir_1_x_1_experrs_train.csv happy_data_ir_1_x_1_experrs_val.csv -scaler StandardScaler

# python -m src.training -n happy_training_ir_1_x_1 -e 5000 -dp ErrorBasedInvertedRandomDropout -runs 3 -trainset happy_data_ir_1_x_1_experrs_train_scaled.csv -valset happy_data_ir_1_x_1_experrs_val_scaled.csv

# BEST_MODELS=$(python -m src.modules.best_model_selector -n happy_training_ir_1_x_1 -e 5000)

# eval "python -m src.predict -n happy_B -models ${BEST_MODELS} -testset happy_test_data_B_ir_1_x_1_experrs.csv -dp y"
# eval "python -m src.predict -n happy_C -models ${BEST_MODELS} -testset happy_test_data_C_ir_1_x_1_experrs.csv -dp y"
# eval "python -m src.predict -n happy_D -models ${BEST_MODELS} -testset happy_test_data_D_ir_1_x_1_experrs.csv -dp y"

# ### GENERATE ERROR RESULTS ###
# python -m src.report.error_results
