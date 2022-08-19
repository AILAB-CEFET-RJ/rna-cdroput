declare -a REGRESSORS=("dt" "ir" "knn" "mlp" "rf" "xgb")
declare -a STRATEGIES=("1_x_1" "m_x_1" "m_x_m")
declare -a DATASETS=("teddy" "happy")

# declare -a REGRESSORS=("xgb")
# declare -a STRATEGIES=("m_x_m")
# declare -a DATASETS=("happy")

for dataset in "${DATASETS[@]}"
do
    for regressor in "${REGRESSORS[@]}"
    do
        for strategy in "${STRATEGIES[@]}"
        do
            eval "python -m src.xgb_learn -trainset ${dataset}_data_${regressor}_${strategy}_experrs.csv -testset ${dataset}_test_data_B_${regressor}_${strategy}_experrs.csv ${dataset}_test_data_C_${regressor}_${strategy}_experrs.csv ${dataset}_test_data_D_${regressor}_${strategy}_experrs.csv"
        done
    done
done

### GENERATE XGB RESULTS ###
python -m src.report.xgb_results
