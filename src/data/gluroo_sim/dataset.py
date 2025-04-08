import numpy as np
from torch.utils.data import Dataset
import logging
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, QuantileTransformer

from src.data.dataset_collection import SyntheticDatasetCollection
# from src.data.cancer_sim.cancer_simulation import TUMOUR_DEATH_THRESHOLD
# from src.data.cancer_sim.cancer_simulation import generate_params, get_scaling_params, simulate_factual, \
#     simulate_counterfactual_1_step, simulate_counterfactuals_treatment_seq


logger = logging.getLogger(__name__)


class SyntheticGlurooDataset(Dataset):
    """
    Pytorch-style Dataset of Tumor Growth Simulator datasets
    """

    def __init__(self,
                 subset_name: str,
                 mode,
                 file_path='/Users/sneha/Downloads/full_dataset.csv',
                 seq_length=12):

        self.subset_name = subset_name

        if mode == 'factual':
            self.data = self.simulate_factual(file_path, seq_length)
        elif mode == 'counterfactual_one_step':
            self.data = self.simulate_counterfactual_1_step(file_path, seq_length)
        # elif mode == 'counterfactual_treatment_seq':
        #     assert projection_horizon is not None
        #     self.data = simulate_counterfactuals_treatment_seq(self.params, seq_length, projection_horizon, cf_seq_mode)
        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        # self.treatment_mode = treatment_mode
        # self.exploded = False

        self.norm_const = 1.0

    def simulate_factual(self, file_path, seq_length):
        """
        Simulates glucose level sequences based on synthetic dataset factors.
        
        :param data: DataFrame containing columns ['glucose', 'carbs', 'insulin', 'exercise',
                    'stress', 'meal_insulin_delay', 'active_insulin', 'carb_impact']
        :param seq_length: Length of each sequence
        :return: Dictionary containing simulated sequences and related factors
        """
        data = pd.read_csv(file_path)
        num_samples = data.shape[0] - seq_length + 1
    
        glucose_seq = np.zeros((num_samples, seq_length))
        carbs_seq = np.zeros((num_samples, seq_length))
        insulin_seq = np.zeros((num_samples, seq_length))
        exercise_seq = np.zeros((num_samples, seq_length))
        stress_seq = np.zeros((num_samples, seq_length))
        meal_insulin_delay_seq = np.zeros((num_samples, seq_length))
        active_insulin_seq = np.zeros((num_samples, seq_length))
        carb_impact_seq = np.zeros((num_samples, seq_length))
        
        for i in range(num_samples):
            glucose_seq[i] = data['glucose'].iloc[i:i+seq_length].values
            carbs_seq[i] = data['carbs'].iloc[i:i+seq_length].values
            insulin_seq[i] = data['insulin'].iloc[i:i+seq_length].values
            exercise_seq[i] = data['exercise'].iloc[i:i+seq_length].values
            stress_seq[i] = data['stress'].iloc[i:i+seq_length].values
            meal_insulin_delay_seq[i] = data['meal_insulin_delay'].iloc[i:i+seq_length].values
            active_insulin_seq[i] = data['active_insulin'].iloc[i:i+seq_length].values
            carb_impact_seq[i] = data['carb_impact'].iloc[i:i+seq_length].values
        
        output = {
            'glucose': glucose_seq,
            'carbs': carbs_seq,
            'insulin': insulin_seq,
            'exercise': exercise_seq,
            'stress': stress_seq,
            'meal_insulin_delay': meal_insulin_delay_seq,
            'active_insulin': active_insulin_seq,
            'carb_impact': carb_impact_seq,
            'sequence_lengths': np.full(num_samples, seq_length)
        }
        
        return output

    def simulate_counterfactual_1_step(self, file_path, seq_length=24):
        """
        Simulates factual and counterfactual glucose trajectories under three interventions:
        1. First intake carb dosage is halved
        2. First meal is delayed by 1 time step
        3. First insulin intake is increased by 50%
        """
        data = pd.read_csv(file_path)
        num_samples = data.shape[0] - seq_length + 1
        
        glucose_seq = np.zeros((num_samples, seq_length))
        carbs_seq = np.zeros((num_samples, seq_length))
        insulin_seq = np.zeros((num_samples, seq_length))
        exercise_seq = np.zeros((num_samples, seq_length))
        stress_seq = np.zeros((num_samples, seq_length))
        meal_insulin_delay_seq = np.zeros((num_samples, seq_length))
        active_insulin_seq = np.zeros((num_samples, seq_length))
        carb_impact_seq = np.zeros((num_samples, seq_length))
        
        for i in tqdm(range(num_samples)):
            selected_data = data.iloc[i:i + seq_length]
            glucose_factual = selected_data['glucose'].values.copy()
            factual_carbs = selected_data['carbs'].values.copy()
            factual_insulin = selected_data['insulin'].values.copy()
            carb_absorption = selected_data['carb_impact'].values.copy()
            insulin_sensitivity = selected_data['active_insulin'].values.copy()
            
            # Counterfactual 1: Halving First Carb Intake
            counterfactual_carbs = factual_carbs.copy()
            counterfactual_carbs[0] *= 0.5
            counterfactual_carb_impact = carb_absorption.copy()
            counterfactual_carb_impact[0] *= 0.5
            
            glucose_half_carb = glucose_factual.copy()
            for t in range(seq_length - 1):
                target_glucose = (
                    glucose_half_carb[t] + counterfactual_carb_impact[t + 1] * counterfactual_carbs[t] -
                    insulin_sensitivity[t + 1] * factual_insulin[t]
                )
                glucose_half_carb[t + 1] = 0.9 * glucose_half_carb[t] + 0.1 * target_glucose
            
            # Counterfactual 2: Delaying meal by 1 time step
            counterfactual_carbs_1 = np.zeros_like(factual_carbs)
            counterfactual_carbs_1[1:] = factual_carbs[:-1]
            
            glucose_delayed_meal = glucose_factual.copy()
            for t in range(seq_length - 1):
                target_glucose = (
                    glucose_delayed_meal[t] + carb_absorption[t + 1] * counterfactual_carbs_1[t] -
                    insulin_sensitivity[t + 1] * factual_insulin[t]
                )
                glucose_delayed_meal[t + 1] = 0.9 * glucose_delayed_meal[t] + 0.1 * target_glucose
                glucose_delayed_meal[t + 1] = np.clip(glucose_delayed_meal[t + 1], 50, 300)
            
            # Counterfactual 3: Doubling First Insulin Dose
            counterfactual_insulin = factual_insulin.copy()
            counterfactual_insulin[0] *= 2.0
            
            glucose_double_insulin = glucose_factual.copy()
            for t in range(seq_length - 1):
                target_glucose = (
                    glucose_double_insulin[t] + carb_absorption[t + 1] * factual_carbs[t] -
                    insulin_sensitivity[t + 1] * counterfactual_insulin[t]
                )
                glucose_double_insulin[t + 1] = 0.9 * glucose_double_insulin[t] + 0.1 * target_glucose
                
            glucose_seq[i] = glucose_factual
            carbs_seq[i] = factual_carbs
            insulin_seq[i] = factual_insulin
            exercise_seq[i] = selected_data['exercise'].values
            stress_seq[i] = selected_data['stress'].values
            meal_insulin_delay_seq[i] = selected_data['meal_insulin_delay'].values
            active_insulin_seq[i] = insulin_sensitivity
            carb_impact_seq[i] = carb_absorption
        
        output = {
            'glucose': glucose_seq,
            'carbs': carbs_seq,
            'insulin': insulin_seq,
            'exercise': exercise_seq,
            'stress': stress_seq,
            'meal_insulin_delay': meal_insulin_delay_seq,
            'active_insulin': active_insulin_seq,
            'carb_impact': carb_impact_seq,
            'sequence_lengths': np.full(num_samples, seq_length),
        }
        return output

    # def simulate_counterfactual_1_step(self, file_path, seq_length=12):
    #     """
    #     Simulates factual and counterfactual glucose trajectories under three interventions:
    #     1. First intake carb dosage is halved
    #     2. First meal is delayed by 1 time step
    #     3. First insulin intake is increased by 50%

    #     :param data: DataFrame containing glucose-related time series data
    #     :param seq_length: Length of sequence to simulate
    #     :return: Dictionary with factual and counterfactual glucose trajectories
    #     """
    #     data = pd.read_csv(file_path)
    #     carb_intake_index = data['carbs'].gt(0).idxmax()


    #     # Identify first carb intake        
    #     if carb_intake_index is None:
    #         raise ValueError("No carb intake detected in the dataset.")

    #     # Extract sequence starting from first carb intake
    #     selected_data = data.iloc[carb_intake_index:carb_intake_index + 36]

    #     # Initialize storage for glucose levels
    #     glucose_factual = np.zeros(seq_length*3)
    #     glucose_factual[0] = selected_data.iloc[0]['glucose']

    #     # Extract values for simulation
    #     factual_carbs = selected_data['carbs'].values
    #     factual_insulin = selected_data['insulin'].values
    #     carb_absorption = selected_data['carb_impact'].values
    #     insulin_sensitivity = selected_data['active_insulin'].values

    #     # Simulate **factual glucose trajectory**
    #     # for t in range(seq_length*5 - 1):
    #     #     target_glucose = (
    #     #         glucose_factual[t] + carb_absorption[t+1] * factual_carbs[t] -
    #     #         insulin_sensitivity[t+1] * factual_insulin[t]
    #     #     )
    #     #     glucose_factual[t + 1] = 0.9 * glucose_factual[t] + 0.1 * target_glucose
    #         # glucose_factual[t + 1] = np.clip(glucose_factual[t + 1], 50, 300)

    #     # factual_glucose_seq = np.zeros((num_samples, seq_length))
    #     # for i in range(num_samples):
    #     #     factual_glucose_seq[i] = data.iloc[i:i+seq_length, 1].values
    #     # ------------------------------
    #     # **Counterfactual 1: Halving First Carb Intake**
    #     # ------------------------------
    #     glucose_half_carb = np.zeros(seq_length*3)
    #     glucose_half_carb[0] = glucose_factual[0]
    #     counterfactual_carbs = factual_carbs.copy()
    #     counterfactual_carbs[0] *= 0.5  # Reduce first carb intake by 50%
    #     counterfactual_carb_impact = carb_absorption.copy()
    #     counterfactual_carb_impact[0] *= 0.5

    #     for t in range(seq_length*3 - 1):
    #         target_glucose = (
    #             glucose_half_carb[t] + carb_absorption[t+1] * counterfactual_carbs[t] -
    #             insulin_sensitivity[t+1] * factual_insulin[t]
    #         )
    #         glucose_half_carb[t + 1] = 0.9 * glucose_half_carb[t] + 0.1 * target_glucose
    #         # glucose_half_carb[t + 1] = np.clip(glucose_half_carb[t + 1], 50, 300)

    #     data.iloc[carb_intake_index:carb_intake_index + 36, 1] = glucose_half_carb
    #     data.iloc[carb_intake_index:carb_intake_index + 36, 2] = counterfactual_carbs

    #     # ------------------------------
    #     # **Counterfactual 2: Delaying second meal by half an hr **
    #     # ------------------------------      
    #     print('---- meal indexes', data[data['carbs']>0].index)
    #     print('---- second smallest index', data[data['carbs']>0].index[1]) 
    #     second_meal_index = data[data['carbs']>0].index[1] 
    #     selected_meal_data = data[second_meal_index: second_meal_index+48]

    #     # Initialize storage for glucose levels
    #     glucose_factual_1 = np.zeros(seq_length*4)
    #     glucose_factual_1[0] = selected_meal_data.iloc[0]['glucose']

    #     # Extract values for simulation
    #     factual_carbs_1 = selected_meal_data['carbs'].values
    #     factual_insulin_1 = selected_meal_data['insulin'].values
    #     carb_absorption_1 = selected_meal_data['carb_impact'].values
    #     insulin_sensitivity_1 = selected_meal_data['active_insulin'].values

    #     # Simulate **factual glucose trajectory**
    #     for t in range(seq_length*4 - 1):
    #         target_glucose_1 = (
    #             glucose_factual_1[t] + carb_absorption_1[t+1] * factual_carbs_1[t] -
    #             insulin_sensitivity_1[t+1] * factual_insulin_1[t]
    #         )
    #         glucose_factual_1[t + 1] = 0.9 * glucose_factual_1[t] + 0.1 * target_glucose_1

    #     glucose_delayed_meal = np.zeros(seq_length*4)
    #     glucose_delayed_meal[0] = glucose_factual_1[0]
    #     counterfactual_carbs_1 = np.zeros_like(factual_carbs_1)
    #     counterfactual_carbs_1[6:] = factual_carbs_1[:-6]  # Delay meal by 6 steps (30 min)

    #     for t in range(seq_length*4 - 1):
    #         target_glucose_1 = (
    #             glucose_delayed_meal[t] + carb_absorption_1[t+1] * counterfactual_carbs_1[t] -
    #             insulin_sensitivity_1[t+1] * factual_insulin_1[t]
    #         )
    #         glucose_delayed_meal[t + 1] = 0.9 * glucose_delayed_meal[t] + 0.1 * target_glucose_1
    #         glucose_delayed_meal[t + 1] = np.clip(glucose_delayed_meal[t + 1], 50, 300)

    #     data.iloc[second_meal_index:second_meal_index + 48, 1] = glucose_factual_1
    #     data.iloc[second_meal_index:second_meal_index + 48, 2] = counterfactual_carbs_1

    #     # ------------------------------
    #     # **Counterfactual 3: Increasing First Insulin Intake by 50%**
    #     # ------------------------------
    #     # insulin_index = data[data['insulin']>0.0].index.min()
    #     # selected_insulin_data = data[insulin_index: insulin_index+48]

    #     # # Initialize storage for glucose levels
    #     # glucose_factual_2 = np.zeros(seq_length*4)
    #     # glucose_factual_2[0] = selected_insulin_data.iloc[0]['glucose']

    #     # # Extract values for simulation
    #     # factual_carbs_2 = selected_insulin_data['carbs'].values
    #     # factual_insulin_2 = selected_insulin_data['insulin'].values
    #     # carb_absorption_2 = selected_insulin_data['carb_impact'].values
    #     # insulin_sensitivity_2 = selected_insulin_data['active_insulin'].values

    #     # # Simulate **factual glucose trajectory**
    #     # for t in range(seq_length*4 - 1):
    #     #     target_glucose_2 = (
    #     #         glucose_factual_2[t] + carb_absorption_2[t+1] * factual_carbs_2[t] -
    #     #         insulin_sensitivity_2[t+1] * factual_insulin_2[t]
    #     #     )
    #     #     glucose_factual_2[t + 1] = 0.9 * glucose_factual_2[t] + 0.1 * target_glucose_2

    #     # glucose_more_insulin = np.zeros(seq_length*4)
    #     # glucose_more_insulin[0] = glucose_factual_2[0]
    #     # counterfactual_insulin = factual_insulin_2.copy()
    #     # counterfactual_insulin[0] *= 1.5  # Increase first insulin intake by 50%

    #     # for t in range(seq_length*4 - 1):
    #     #     target_glucose_2 = (
    #     #         glucose_more_insulin[t] + carb_absorption_2[t+1] * factual_carbs_2[t] -
    #     #         insulin_sensitivity_2[t+1] * counterfactual_insulin[t]
    #     #     )
    #     #     glucose_more_insulin[t + 1] = 0.9 * glucose_more_insulin[t] + 0.1 * target_glucose_2
    #     #     # glucose_more_insulin[t + 1] = np.clip(glucose_more_insulin[t + 1], 50, 300)

    #     # data.iloc[insulin_index:insulin_index + 48, 1] = glucose_more_insulin
    #     # data.iloc[insulin_index:insulin_index + 48, 3] = counterfactual_insulin

    #     # ------------------------------
    #     # Plot results
    #     # ------------------------------
    #     # data = data.rename(columns={'Unnamed: 0': 'Timestamp'})
    #     # print(data)
    #     # data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    #     # print('---- columns',data.columns)
    #     # print('--- data head')
    #     # print( data.head())
    #     # plt.figure(figsize=(10, 5))
    #     # plt.plot(data["Timestamp"], data['glucose'], label="Factual", marker='o')
    #     # plt.plot(data["Timestamp"], data['carbs'], label="Half Carb Intake", linestyle='dashed', marker='s')
    #     # plt.plot(data["Timestamp"], data['active_insulin'], label="Delayed First Meal", linestyle='dashed', marker='^')
    #     # plt.plot(data["Timestamp"], data['insulin'], label="Increased First Insulin", linestyle='dashed', marker='v')

    #     # plt.title("Glucose Trajectories with Counterfactual Interventions")
    #     # plt.xlabel("Time Step")
    #     # plt.ylabel("Glucose Level")
    #     # plt.legend()
    #     # plt.show()    

    #     # # Return dictionary of results
    #     # return {
    #     #     "glucose": glucose_factual,
    #     #     "half_carb": glucose_half_carb,
    #     #     "delayed_meal": glucose_delayed_meal,
    #     #     "more_insulin": glucose_more_insulin
    #     # }

    #     num_samples = data.shape[0] - seq_length + 1 

    #     glucose_seq = np.zeros((num_samples, seq_length))
    #     carbs_seq = np.zeros((num_samples, seq_length))
    #     insulin_seq = np.zeros((num_samples, seq_length))
    #     exercise_seq = np.zeros((num_samples, seq_length))
    #     stress_seq = np.zeros((num_samples, seq_length))
    #     meal_insulin_delay_seq = np.zeros((num_samples, seq_length))
    #     active_insulin_seq = np.zeros((num_samples, seq_length))
    #     carb_impact_seq = np.zeros((num_samples, seq_length))
        
    #     for i in range(num_samples):
    #         glucose_seq[i] = data.iloc[i:i+seq_length, 1].values
    #         carbs_seq[i] = data.iloc[i:i+seq_length, 2].values
    #         insulin_seq[i] = data.iloc[i:i+seq_length, 3].values
    #         exercise_seq[i] = data.iloc[i:i+seq_length, 4].values
    #         stress_seq[i] = data.iloc[i:i+seq_length, 5].values
    #         meal_insulin_delay_seq[i] = data.iloc[i:i+seq_length, 6].values
    #         active_insulin_seq[i] = data.iloc[i:i+seq_length, 7].values
    #         carb_impact_seq[i] = data.iloc[i:i+seq_length, 8].values

    #     output = {
    #         'glucose': glucose_seq,
    #         'carbs': carbs_seq,
    #         'insulin': insulin_seq,
    #         'exercise': exercise_seq,
    #         'stress': stress_seq,
    #         'meal_insulin_delay': meal_insulin_delay_seq,
    #         'active_insulin': active_insulin_seq,
    #         'carb_impact': carb_impact_seq,
    #         'sequence_lengths': np.full(num_samples, seq_length),
    #     }
    #     return output

        # # Identify first instance where carbs > 0
        # carb_intake_index = data[data['carbs'] > 0].index.min()

        # if carb_intake_index is None:
        #     raise ValueError("No carb intake detected in the dataset.")

        # # Extract glucose trajectory from that point onward
        # seq_length = 12  # Define sequence length
        # selected_data = data.iloc[carb_intake_index:carb_intake_index + seq_length].copy()

        # # Extract initial values
        # glucose_levels = np.zeros(seq_length)
        # glucose_levels[0] = selected_data.iloc[0]['glucose']
        # carb_absorption = selected_data['carb_impact'].values
        # insulin_sensitivity = selected_data['active_insulin'].values

        # # Factual scenario
        # factual_carbs = selected_data['carbs'].values
        # print('+++', factual_carbs)
        # factual_insulin = selected_data['insulin'].values

        # for t in range(seq_length - 1):

        #     target_glucose = (
        #         glucose_levels[t] + carb_absorption[t+1] * factual_carbs[t] -
        #         insulin_sensitivity[t+1] * factual_insulin[t]
        #     )
        #     glucose_levels[t + 1] = 0.9 * glucose_levels[t] + 0.1 * target_glucose
        #     # glucose_levels[t + 1] = np.clip(glucose_levels[t + 1], 50, 300)

        # # Counterfactual scenario (Halving Carb Intake)
        # counterfactual_glucose = np.zeros(seq_length)
        # counterfactual_glucose[0] = glucose_levels[0]
        # counterfactual_carbs = factual_carbs * 0.5
        # print('+++', counterfactual_carbs)
        # print('+++', carb_absorption)
        # for t in range(seq_length - 1):

        #     target_glucose = (
        #         glucose_levels[t] + carb_absorption[t+1] * counterfactual_carbs[t] -
        #         insulin_sensitivity[t+1] * factual_insulin[t]
        #     )
        #     counterfactual_glucose[t + 1] = 0.9 * counterfactual_glucose[t] + 0.1 * target_glucose
        #     # counterfactual_glucose[t + 1] = np.clip(counterfactual_glucose[t + 1], 50, 300)

        # # Plot results
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(seq_length), glucose_levels, label="Factual", marker='o')
        # plt.plot(range(seq_length), counterfactual_glucose, label="Half Carb Dosage", linestyle='dashed', marker='s')
        # plt.title("Glucose Trajectory with Carb Intake Intervention")
        # plt.xlabel("Time Step")
        # plt.ylabel("Glucose Level")
        # plt.legend()
        # plt.show()
            

        # num_patients = data.shape[0]
        # num_scenarios = 2  # Carb dosage and Meal timing interventions
        # num_test_points = num_patients * seq_length * num_scenarios
        
        # glucose_levels = np.zeros((num_test_points, seq_length))
        # sequence_lengths = np.zeros(num_test_points)
        # patient_ids = np.zeros(num_test_points)
        # test_idx = 0
        
        # for i in tqdm(range(num_patients), total=num_patients):
        #     noise = 0.05 * np.random.randn(seq_length)  # 5% variability
            
        #     factual_glucose = np.zeros(seq_length)
        #     factual_glucose[0] = data.iloc[i, 1]  # Initial glucose
        #     carb_absorption = data.iloc[i, 8]  # Carb impact
        #     insulin_sensitivity = data.iloc[i, 7]  # Active insulin
        #     meal_delay = data.iloc[i, 6]  # Meal insulin delay
            
        #     factual_carbs = data.iloc[i, 2]  # Carbs intake
        #     factual_insulin = data.iloc[i, 3]  # Insulin
        #     factual_meal_timing = meal_delay  # Meal timing
            
        #     # Simulate factual glucose
        #     for t in range(seq_length - 1):
        #         factual_glucose[t + 1] = (
        #             factual_glucose[t] + carb_absorption * factual_carbs -
        #             insulin_sensitivity * factual_insulin +
        #             noise[t + 1]
        #         )
        #         factual_glucose[t + 1] = np.clip(factual_glucose[t + 1], 50, 300)
            
        #     # Store factual data
        #     glucose_levels[test_idx] = factual_glucose
        #     patient_ids[test_idx] = i
        #     sequence_lengths[test_idx] = seq_length
        #     test_idx += 1
            
        #     # Counterfactual scenarios
        #     for scenario in [(0.5, 1), (1, 1.5)]:  # (Carb dosage factor, Meal timing factor)
        #         counterfactual_carbs = factual_carbs * scenario[0]
        #         counterfactual_meal_timing = factual_meal_timing * scenario[1]
                
        #         counterfactual_glucose = np.zeros(seq_length)
        #         counterfactual_glucose[0] = factual_glucose[0]
                
        #         for t in range(seq_length - 1):
        #             counterfactual_glucose[t + 1] = (
        #                 counterfactual_glucose[t] + carb_absorption * counterfactual_carbs -
        #                 insulin_sensitivity * factual_insulin +
        #                 noise[t + 1]
        #             )
        #             counterfactual_glucose[t + 1] = np.clip(counterfactual_glucose[t + 1], 50, 300)
                
        #         glucose_levels[test_idx] = counterfactual_glucose
        #         patient_ids[test_idx] = i
        #         sequence_lengths[test_idx] = seq_length
        #         test_idx += 1
        
        # outputs = {
        #     'glucose_levels': glucose_levels[:test_idx],
        #     # 'sequence_lengths': sequence_lengths[:test_idx],
        #     'patient_ids': patient_ids[:test_idx]
        # }
        # self.plot_glucose_interventions(outputs)
        # return outputs

    # def plot_glucose_interventions(self, outputs, num_patients=5):
    #     glucose_levels = outputs['glucose_levels']
    #     patient_ids = outputs['patient_ids']
        
    #     unique_patients = np.unique(patient_ids)[:num_patients]
    #     fig, axes = plt.subplots(len(unique_patients), 3, figsize=(15, 5 * len(unique_patients)))
        
    #     for i, patient in enumerate(unique_patients):
    #         idx = np.where(patient_ids == patient)[0]
            
    #         axes[i, 0].plot(glucose_levels[idx[0]], label='Factual')
    #         axes[i, 1].plot(glucose_levels[idx[1]], label='Carb Halved', linestyle='dashed')
    #         axes[i, 2].plot(glucose_levels[idx[2]], label='Meal Delayed', linestyle='dotted')
            
    #         for j in range(3):
    #             axes[i, j].set_ylim(50, 300)
    #             axes[i, j].legend()
    #             axes[i, j].set_xlabel('Time Steps')
    #             axes[i, j].set_ylabel('Glucose Level')
    #             axes[i, j].set_title(f'Patient {int(patient)} - Scenario {j}')
        
    #     plt.tight_layout()
    #     plt.show()

    def __getitem__(self, index) -> dict:
        result = {k: v[index] for k, v in self.data.items() if hasattr(v, '__len__') and len(v) == len(self)}
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return self.data['current_covariates'].shape[0]

    def get_scaling_params(self):
        real_idx = ['glucose', 'carbs', 'insulin', 'exercise', 'stress', 'meal_insulin_delay', 'active_insulin', 'carb_impact']

        means = {}
        stds = {}
        seq_lengths = self.data['sequence_lengths']

        for k in real_idx:
            active_values = []
            for i in range(seq_lengths.shape[0]):
                end = int(seq_lengths[i])  # Only consider non-padded values
                active_values += list(self.data[k][i, :end])

            means[k] = np.mean(active_values)
            stds[k] = np.std(active_values)

        return pd.Series(means), pd.Series(stds)

    '''
    key steps:

    1 - Normalization: Standardizes all continuous features (glucose, carbs, insulin, etc.) using mean and standard deviation from scaling_params.
    2 - Treatment Processing: Extracts previous and current carb_impact as treatments.
    3 - Target Variable: Uses future glucose values as the output.
    4 - Active Entries: Marks available data points to handle variable sequence lengths.
    5 - Data Storage: Structures data in a format suitable for transformer training.
    6 - Unified Data Format: Ensures consistency in the dataset with prev_outputs, static_features, and prev_treatments.
    '''

    def process_data(self, scaling_params):
        """
        Pre-process dataset for one-step-ahead prediction
        Args:
            scaling_params: dict of standard normalization parameters (calculated with train subset)
        """
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            mean, std = scaling_params

            horizon = 1
            offset = 1

            print('**** self.data', self.data)

            # Use RobustScaler to handle outliers
            scaler = RobustScaler()
            self.data['glucose'] = scaler.fit_transform(self.data['glucose'])
            self.data['insulin'] = scaler.fit_transform(self.data['insulin'])
            self.data['carbs'] = scaler.fit_transform(self.data['carbs'])
            self.data['exercise'] = scaler.fit_transform(self.data['exercise'])
            self.data['stress'] = scaler.fit_transform(self.data['stress'])
            self.data['meal_insulin_delay'] = scaler.fit_transform(self.data['meal_insulin_delay'])
            self.data['active_insulin'] = scaler.fit_transform(self.data['active_insulin'])

            # Apply log transformation for highly skewed features
            self.data['carb_impact'] = np.log1p(self.data['carb_impact'])

            sequence_lengths = self.data['sequence_lengths']

            # Stack input features
            current_covariates = np.stack([
                self.data['glucose'], self.data['carbs'], self.data['insulin'],
                self.data['exercise'], self.data['stress'], self.data['meal_insulin_delay'],
                self.data['active_insulin']], axis=-1)
            
            # Previous and current treatments
            prev_treatments = self.data['carb_impact'][:, :-2, np.newaxis]  # Previous carb impact
            current_treatments = self.data['carb_impact'][:, :-1, np.newaxis]  # Current carb impact

            outputs = self.data['glucose'][:, horizon:, np.newaxis]  # Future glucose values

            output_means = np.mean(self.data['glucose'])
            output_stds = np.std(self.data['glucose'])

            # Add active entries
            active_entries = np.zeros(outputs.shape)
            for i in range(sequence_lengths.shape[0]):
                sequence_length = int(sequence_lengths[i])
                active_entries[i, :sequence_length, :] = 1

            # Store processed data
            self.data['current_covariates'] = current_covariates[:, :-offset, :]
            self.data['outputs'] = outputs
            self.data['active_entries'] = active_entries
            self.data['prev_treatments'] = prev_treatments
            self.data['current_treatments'] = current_treatments
            self.data['unscaled_outputs'] = (outputs * output_stds) + output_means

            self.scaling_params = {
                'input_means': mean.values.flatten(),
                'inputs_stds': std.values.flatten(),
                'output_means': output_means,
                'output_stds': output_stds
            }

            # Unified data format
            self.data['prev_outputs'] = self.data['current_covariates'][:, :, :1]
            self.data['static_features'] = self.data['current_covariates'][:, 0, 1:]
            zero_init_treatment = np.zeros(shape=[current_covariates.shape[0], 1, self.data['prev_treatments'].shape[-1]])
            self.data['prev_treatments'] = np.concatenate([zero_init_treatment, self.data['prev_treatments']], axis=1)

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data

    # def process_data(self, scaling_params):
    #     """
    #     Pre-process dataset for one-step-ahead prediction
    #     Args:
    #         scaling_params: dict of standard normalization parameters (calculated with train subset)
    #     """
    #     if not self.processed:
    #         logger.info(f'Processing {self.subset_name} dataset before training')

    #         mean, std = scaling_params

    #         horizon = 1
    #         offset = 1

    #         print('**** self.data', self.data)
    #         # Normalize continuous features
    #         glucose = (self.data['glucose'] - mean['glucose']) / std['glucose']
    #         # factual_glucose = (self.data['factual_glucose'] - mean['factual_glucose']) / std['factual_glucose']
    #         carbs = (self.data['carbs'] - mean['carbs']) / std['carbs']
    #         insulin = (self.data['insulin'] - mean['insulin']) / std['insulin']
    #         exercise = (self.data['exercise'] - mean['exercise']) / std['exercise']
    #         stress = (self.data['stress'] - mean['stress']) / std['stress']
    #         meal_insulin_delay = (self.data['meal_insulin_delay'] - mean['meal_insulin_delay']) / std['meal_insulin_delay']
    #         active_insulin = (self.data['active_insulin'] - mean['active_insulin']) / std['active_insulin']

    #         sequence_lengths = self.data['sequence_lengths']

    #         # Stack input features
    #         current_covariates = np.stack([glucose, carbs, insulin, exercise, stress, meal_insulin_delay, active_insulin], axis=-1)
            
    #         # Previous and current treatments
    #         prev_treatments = self.data['carb_impact'][:, :-2, np.newaxis]  # Previous carb impact
    #         current_treatments = self.data['carb_impact'][:, :-1, np.newaxis]  # Current carb impact

    #         outputs = glucose[:, horizon:, np.newaxis]  # Future glucose values
    #         # real_outputs = factual_glucose[:, horizon:, np.newaxis]

    #         output_means = mean['glucose']
    #         output_stds = std['glucose']

    #         # Add active entries
    #         active_entries = np.zeros(outputs.shape)
    #         for i in range(sequence_lengths.shape[0]):
    #             sequence_length = int(sequence_lengths[i])
    #             active_entries[i, :sequence_length, :] = 1

    #         # Store processed data
    #         self.data['current_covariates'] = current_covariates[:, :-offset, :]
    #         self.data['outputs'] = outputs
    #         # self.data['real_outputs'] = real_outputs
    #         self.data['active_entries'] = active_entries
    #         self.data['prev_treatments'] = prev_treatments
    #         self.data['current_treatments'] = current_treatments
    #         self.data['unscaled_outputs'] = (outputs * std['glucose']) + mean['glucose']
    #         # self.data['unscaled_real_outputs'] = (real_outputs * std['factual_glucose']) + mean['factual_glucose']

    #         self.scaling_params = {
    #             'input_means': mean.values.flatten(),
    #             'inputs_stds': std.values.flatten(),
    #             'output_means': output_means,
    #             'output_stds': output_stds
    #         }

    #         # Unified data format
    #         self.data['prev_outputs'] = self.data['current_covariates'][:, :, :1]
    #         self.data['static_features'] = self.data['current_covariates'][:, 0, 1:]
    #         zero_init_treatment = np.zeros(shape=[current_covariates.shape[0], 1, self.data['prev_treatments'].shape[-1]])
    #         self.data['prev_treatments'] = np.concatenate([zero_init_treatment, self.data['prev_treatments']], axis=1)

    #         data_shapes = {k: v.shape for k, v in self.data.items()}
    #         logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

    #         self.processed = True
    #     else:
    #         logger.info(f'{self.subset_name} Dataset already processed')

    #     return self.data


    def explode_trajectories(self, projection_horizon):
        assert self.processed

        logger.info(f'Exploding {self.subset_name} dataset before testing (multiple sequences)')

        outputs = self.data['outputs']  # Future glucose values
        prev_outputs = self.data['prev_outputs']  # Past glucose values
        sequence_lengths = self.data['sequence_lengths']  # Length of each patient sequence
        active_entries = self.data['active_entries']  # Mask for valid data points
        current_treatments = self.data['current_treatments']  # Insulin and carb intake at each step
        previous_treatments = self.data['prev_treatments']  # Previous insulin and carb intake
        static_features = self.data['static_features']  # Static data (e.g., patient demographics, if available)

        num_patients, max_seq_length, num_features = outputs.shape
        num_seq2seq_rows = num_patients * max_seq_length

        seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros((num_seq2seq_rows, max_seq_length, current_treatments.shape[-1]))
        seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
        seq2seq_outputs = np.zeros((num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
        seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
        seq2seq_active_entries = np.zeros((num_seq2seq_rows, max_seq_length, active_entries.shape[-1]))
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)

        total_seq2seq_rows = 0

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])

            for t in range(projection_horizon, sequence_length):  # Shift outputs back by projection horizon
                seq2seq_active_entries[total_seq2seq_rows, :(t + 1), :] = active_entries[i, :(t + 1), :]
                seq2seq_previous_treatments[total_seq2seq_rows, :(t + 1), :] = previous_treatments[i, :(t + 1), :]
                seq2seq_current_treatments[total_seq2seq_rows, :(t + 1), :] = current_treatments[i, :(t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, :(t + 1), :] = outputs[i, :(t + 1), :]
                seq2seq_prev_outputs[total_seq2seq_rows, :(t + 1), :] = prev_outputs[i, :(t + 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                total_seq2seq_rows += 1

        # Trim arrays to the actual number of rows generated
        seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
        seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
        seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

        new_data = {
            'prev_treatments': seq2seq_previous_treatments,
            'current_treatments': seq2seq_current_treatments,
            'static_features': seq2seq_static_features,
            'prev_outputs': seq2seq_prev_outputs,
            'outputs': seq2seq_outputs,
            'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
            'sequence_lengths': seq2seq_sequence_lengths,
            'active_entries': seq2seq_active_entries,
        }

        self.data = new_data
        self.exploded = True

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

    '''
    process_sequential (For Training)
        This function restructures the dataset to create multiple overlapping sequences for training a model.

        Key Steps:
        1 - Initialize Output Data Structures:
        It creates zero-filled NumPy arrays to store different components of the processed dataset (e.g., seq2seq_state_inits, seq2seq_outputs, seq2seq_current_treatments).
        
        2 - Iterate Over Patients:
        It loops over all patients in the dataset.
        For each patient, it considers their full sequence length and applies a rolling-window approach to generate multiple overlapping sub-sequences.
        
        3 - Generate Multi-Step Prediction Data:
        It creates new sequences by shifting time steps and extracting the necessary features (encoder state, outputs, treatments, covariates, etc.).
        Ensures each generated sequence has projection_horizon steps into the future.
        
        4 - Filter Out Short Sequences:
        Any sequences that do not have enough future steps (projection_horizon) are removed.
        
        5 - Store Processed Data:
        Packages the transformed data into a dictionary.
        If save_encoder_r is enabled, it saves the encoder representations.
        
        6 - Mark Dataset as Processed:
        Updates flags self.processed_sequential = True and self.exploded = True, indicating that the dataset has been transformed.
    '''

    '''
    process_autoregressive_test

        1 - Check if Data is Already Processed

        Ensures that the dataset has been sequentially processed before proceeding.
        If already processed for autoregressive testing, logs a message and returns.
        
        2 - Extract Relevant Data

        Retrieves current_treatments, prev_treatments, and sequence_lengths from the original dataset.
        Initializes a dictionary current_dataset to store processed data.
        
        3 - Initialize Data Structures

        Creates zero-filled arrays for current_covariates, prev_treatments, current_treatments, init_state, and active_encoder_r.
        Sets active_entries to ones for all projection horizon steps.
        
        4 - Iterate Over Each Patient

        Determines the index (fact_length) where forecasting starts (last projection_horizon steps).
        Fills init_state with encoder representations at the last observed timestep.
        Assigns corresponding prev_treatments and current_treatments.
        Copies encoder output as current_covariates for autoregressive prediction.
        Marks active time steps in active_encoder_r.
        
        5 - Finalize Data Processing

        Adds prev_outputs (past values for forecasting) and static_features.
        Stores the original dataset as data_processed_seq before updating the active dataset.
        Logs the shapes of processed data for verification.
        
        6 - Handle Encoder Representations

        If save_encoder_r is enabled, saves encoder representations.
        Marks dataset as processed for autoregressive testing.
        This function prepares the test dataset for autoregressive sequence prediction, ensuring that the last n steps of each sequence are correctly structured for forecasting.
    '''

class SyntheticGlurooDatasetCollection(SyntheticDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
    """

    def __init__(self, seed=100, projection_horizon=5, **kwargs):
        """
        Args:
            chemo_coeff: Confounding coefficient of chemotherapy
            radio_coeff: Confounding coefficient of radiotherapy
            num_patients: Number of patients in dataset
            window_size: Used for biased treatment assignment
            max_seq_length: Max length of time series
            subset_name: train / val / test
            mode: factual / counterfactual_one_step / counterfactual_treatment_seq
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            seed: Random seed
            lag: Lag for treatment assignment window
            cf_seq_mode: sliding_treatment / random_trajectories
            treatment_mode: multiclass / multilabel
        """
        super(SyntheticGlurooDatasetCollection, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        
        self.train_f = SyntheticGlurooDataset('train', mode='factual')
        self.val_f = SyntheticGlurooDataset('val', mode='factual')
        self.test_cf_one_step = SyntheticGlurooDataset('test', mode='counterfactual_one_step')
        # self.test_cf_treatment_seq = SyntheticCancerDataset(chemo_coeff, radio_coeff, num_patients['test'], window_size,
        #                                                     max_seq_length, 'test', mode='counterfactual_treatment_seq',
        #                                                     projection_horizon=projection_horizon, lag=lag,
        #                                                     cf_seq_mode=cf_seq_mode, treatment_mode=treatment_mode)
        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = False
        self.train_scaling_params = self.train_f.get_scaling_params()
