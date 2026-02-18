import os
import pandas as pd

def save_result(result, multihead, tasks_number, classes_per_task, hidden_dim, topk, layer_w_thresholding, fixed_threshold, repeat, root_dir):
    multihead_str = "multihead" if multihead else "singlehead"
    layer_w_thresholding_str = "None" if layer_w_thresholding == [[]] else "_".join(map(str, layer_w_thresholding))
    adaptive_threshold_str = f"fixed_threshold_{fixed_threshold}" if fixed_threshold else "increasing_threshold"
    if classes_per_task == 0:
        classes_per_task = 100 // tasks_number
    for result_type in result:
        file_name = f'MLP_GPM_{result_type}_{multihead_str}_{tasks_number}_tasks_{classes_per_task}_classes_hidden_dim_{hidden_dim}_topk_{topk}_layer_idx_w_thresholding_{layer_w_thresholding_str}_{adaptive_threshold_str}_repeat_{repeat}.csv'

        result_dir = os.path.join(root_dir, 'result', multihead_str, f'{tasks_number}_tasks', f'{classes_per_task}_classes', f'{hidden_dim}_hidden_dim')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        file_path = os.path.join(result_dir, file_name)

        if result_type == "acc_matrix":
            index = [f'After learn task {i}' for i in range(tasks_number)]
            columns = [f'Task {i}' for i in range(tasks_number)]
        elif result_type == "gradient_basis_number":
            index = [f'Layer {i}' for i in range(5)]
            columns = [f'After learn task {i}' for i in range(tasks_number)]
        elif result_type == "best_final":
            index = [f'Task {i}' for i in range(tasks_number)]
            columns = ['Best Model Performance', 'Final Performance', 'Best-Final Gap']

        df = pd.DataFrame(result[result_type], index=index, columns=columns)
        df.to_csv(file_path)