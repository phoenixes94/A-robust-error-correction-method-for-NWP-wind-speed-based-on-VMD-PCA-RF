A hybrid method for numerical weather prediction wind speed based on Bayesian optimization (version 1.2.0) and error correction: First release of my code
1、pip3 install -r /ws_correct_ML/requirements.txt
####2、Run a scheduled task in the background to start the prediction and plotting tasks(crontab -e)
####2.1、Process path 1
#31 17 * * * sh /ws_correct_ML/machinelearning/ml_410_master_mlp.sh > /ws_correct_ML/machinelearning/ml_410_master_mlp.log 2>&1
#31 17 * * * sh /ws_correct_ML/machinelearning/ml_410_master.sh > /ws_correct_ML/machinelearning/ml_410_master.log 2>&1
#31 17 * * * sh /ws_correct_ML/machinelearning/ml_410_master_dbn.sh > /ws_correct_ML/machinelearning/ml_410_master_dbn.log 2>&1
#31 17 * * * sh /ws_correct_ML/machinelearning/ml_410_master_lightgbm.sh > /ws_correct_ML/machinelearning/ml_410_master_lightgbm.log 2>&1
#31 17 * * * sh /ws_correct_ML/machinelearning/ml_410_master_xgboost.sh > /ws_correct_ML/machinelearning/ml_410_master_xgboost.log 2>&1
####2.2、Process path 2
#45 17 * * * sh /ws_correct_ML/vmd_pca/ml_410_master_mlp.sh > /ws_correct_ML/vmd_pca/ml_410_master_mlp.log 2>&1
#45 17 * * * sh /ws_correct_ML/vmd_pca/ml_410_master.sh > /ws_correct_ML/vmd_pca/ml_410_master.log 2>&1
#45 17 * * * sh /ws_correct_ML/vmd_pca/ml_410_master_dbn.sh > /ws_correct_ML/vmd_pca/ml_410_master_dbn.log 2>&1
#45 17 * * * sh /ws_correct_ML/vmd_pca/ml_410_master_lightgbm.sh  > /ws_correct_ML/vmd_pca/ml_410_master_lightgbm.log 2>&1
#45 17 * * * sh /ws_correct_ML/vmd_pca/ml_410_master_xgboost.sh > /ws_correct_ML/vmd_pca/ml_410_master_xgboost.log 2>&1

####3、You can start the evaluation task by running a scheduled task in the background
#15 11 * * * sh /ws_correct_ML/vmd_pca/test_10models.sh
#15 11 * * * sh /ws_correct_ML/machinelearning/test_10models.sh


####4、Calculate all the metrics
python /ws_correct_ML/vmd_pca/evaluate_all_10models.py


####5、Missing data file please go to the website for: https://doi.org/10.5281/zenodo.7940686

####6、Preprint link: https://egusphere.copernicus.org/preprints/2023/egusphere-2023-945/#discussion
