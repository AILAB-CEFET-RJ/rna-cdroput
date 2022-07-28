# DOWNLOAD.SH MUST BE EXECUTED FIRST!

## RUN ALL STRATEGIES IN ALL DATASETS
declare -a REGRESSORS=("dt" "knn" "mlp" "rf" "xgb")
declare -a STRATEGIES=("1_x_1" "m_x_1" "m_x_m")
declare -a DATASETS=("sdss")

for dataset in "${DATASETS[@]}"
do
    for regressor in "${REGRESSORS[@]}"
    do
        for strategy in "${STRATEGIES[@]}"
        do
            eval "python -m src.pipeline.dataset_clean -dataset ${dataset}_data.csv"

            eval "python -m src.strategies.${regressor}_${strategy} -dataset ${dataset}_data_clean.csv"

            eval "python -m src.pipeline.dataset_split -dataset ${dataset}_data_clean_${regressor}_${strategy}_experrs.csv -p 60:20:20"

            eval "python -m src.pipeline.dataset_scaling -datafiles ${dataset}_data_clean_${regressor}_${strategy}_experrs_train.csv ${dataset}_data_clean_${regressor}_${strategy}_experrs_val.csv ${dataset}_data_clean_${regressor}_${strategy}_experrs_test.csv -scaler StandardScaler"

            eval "python -m src.training -n ${dataset}_training_${regressor}_${strategy} -e 10 -dp ErrorBasedInvertedRandomDropout -runs 1 -trainset ${dataset}_data_clean_${regressor}_${strategy}_experrs_train_scaled.csv -valset ${dataset}_data_clean_${regressor}_${strategy}_experrs_val_scaled.csv"

            BEST_MODEL=$(python -m src.modules.best_model_selector -n ${dataset}_training_${regressor}_${strategy})

            eval "python -m src.predict -model ${BEST_MODEL} -testset ${dataset}_data_clean_${regressor}_${strategy}_experrs_test.csv -dp y"
        done
    done
done

### ISOTONIC REGRESSOR ###
# ISOTONIC REGRESSOR DOES NOT SUPPORT MULTIPLE INPUT/OUTPUT!
source ./src/scripts/strategies/ir_1_x_1.sh

### GENERATE ERROR RESULTS ###
python -m src.report.error_results
