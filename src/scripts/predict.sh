# THIS SCRIPTS TEST ALL STRATEGIES FOR TEDDY
# DOWNLOAD.SH MUST BE EXECUTED FIRST!

### DECISION TREE ###
source ./src/scripts/strategies/dt_1_x_1.sh
source ./src/scripts/strategies/dt_m_x_1.sh
source ./src/scripts/strategies/dt_m_x_m.sh

### ISOTONIC REGRESSOR ###
source ./src/scripts/strategies/ir_1_x_1.sh
source ./src/scripts/strategies/ir_m_x_1.sh

### KNEIGHBORS REGRESSOR ###
source ./src/scripts/strategies/knn_1_x_1.sh
source ./src/scripts/strategies/knn_m_x_1.sh
source ./src/scripts/strategies/knn_m_x_m.sh

### MULTI LAYER PERCEPTRON REGRESSOR ###
source ./src/scripts/strategies/mlp_1_x_1.sh
source ./src/scripts/strategies/mlp_m_x_1.sh
source ./src/scripts/strategies/mlp_m_x_m.sh

### DECISION TREE REGRESSOR ###
source ./src/scripts/strategies/dt_1_x_1.sh
source ./src/scripts/strategies/dt_m_x_1.sh
source ./src/scripts/strategies/dt_m_x_m.sh

### XGBOOST REGRESSOR ###
source ./src/scripts/strategies/xgb_1_x_1.sh
source ./src/scripts/strategies/xgb_m_x_1.sh
source ./src/scripts/strategies/xgb_m_x_m.sh
