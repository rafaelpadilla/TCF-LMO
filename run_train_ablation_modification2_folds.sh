#Modificação 2: Avaliar contribuição da consistência temporal​
#Não tem consistencia temporal

# RODANDO MUNIQUE DEVICE 0 (mar 1st 00:07)
python train_ablation.py --fold 1 --ablation modification2 --alignment temporal --name_experiment fold_1_ablation_modification2 --device 0 --run_once_without_training --perform_validation;

# RODANDO MUNIQUE DEVICE 0 (mar 1st 00:07)
python train_ablation.py --fold 2 --ablation modification2 --alignment temporal --name_experiment fold_2_ablation_modification2 --device 0 --run_once_without_training --perform_validation;

# RODANDO MUNIQUE DEVICE 1 (mar 1st 00:09)
python train_ablation.py --fold 3 --ablation modification2 --alignment temporal --name_experiment fold_3_ablation_modification2 --device 1 --run_once_without_training --perform_validation;

# RODANDO MUNIQUE DEVICE 1 (mar 1st 00:09)
python train_ablation.py --fold 4 --ablation modification2 --alignment temporal --name_experiment fold_4_ablation_modification2 --device 1 --run_once_without_training --perform_validation;


# RODANDO HELSINQUE DEVICE 0 (mar 1st 00:12)
python train_ablation.py --fold 5 --ablation modification2 --alignment temporal --name_experiment fold_5_ablation_modification2 --device 0 --run_once_without_training --perform_validation;

# RODANDO HELSINQUE DEVICE 0 (mar 1st 00:12)
python train_ablation.py --fold 6 --ablation modification2 --alignment temporal --name_experiment fold_6_ablation_modification2 --device 0 --run_once_without_training --perform_validation;

# RODANDO HELSINQUE DEVICE 1 (mar 1st 00:12)
python train_ablation.py --fold 7 --ablation modification2 --alignment temporal --name_experiment fold_7_ablation_modification2 --device 1 --run_once_without_training --perform_validation;

# RODANDO HELSINQUE DEVICE 1 (mar 1st 00:14)
python train_ablation.py --fold 8 --ablation modification2 --alignment temporal --name_experiment fold_8_ablation_modification2 --device 1 --run_once_without_training --perform_validation;

# RODANDO HELSINQUE DEVICE 1 (mar 1st 00:16)
python train_ablation.py --fold 9 --ablation modification2 --alignment temporal --name_experiment fold_9_ablation_modification2 --device 1 --run_once_without_training --perform_validation;

