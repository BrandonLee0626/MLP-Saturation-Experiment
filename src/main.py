import torch
import argparse
import numpy as np

from copy import deepcopy
from dataloader import get_cifar100
from models import *
from utils import save_result

def run_experiment(args, seed, device, data, taskcla, tasks_number, classes_per_task, multihead, hidden_dim, layer_w_thresholding, fixed_threshold, k, epochs):
    lr = args.lr
    lr_patience = args.lr_patience
    lr_factor = args.lr_factor
    lr_min = args.lr_min

    # Set device and random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    acc_matrix = np.zeros((tasks_number, tasks_number))
    best_acc = list()
    gradient_basis_number = np.zeros((5, tasks_number))

    criterion = torch.nn.CrossEntropyLoss()

    for task_id, classes_per_task in taskcla:
        if fixed_threshold:
            threshold = np.array([fixed_threshold] * 5)
        else:
            threshold = np.array([0.97] * 5) + task_id*np.array([0.03 / tasks_number] * 5)

        xtrain=data[task_id]['train']['x']
        ytrain=data[task_id]['train']['y']
        xvalid=data[task_id]['valid']['x']
        yvalid=data[task_id]['valid']['y']
        xtest =data[task_id]['test']['x']
        ytest =data[task_id]['test']['y']

        best_loss = np.inf

        if task_id == 0:
            model = MLP(
                hidden_dim=hidden_dim,
                output_dim=tasks_number * classes_per_task,
                multihead=multihead,
                taskcla=taskcla
            ).to(device)

            base_model = deepcopy(model)
            best_model = get_model(model)

            feature_list = list()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            for epoch in range(epochs):
                train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, classes_per_task)

                valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion, task_id, classes_per_task)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= lr_factor
                        if lr < lr_min:
                            break
                    patience = lr_patience
                    adjust_learning_rate(optimizer, epoch, lr, lr_factor)

            set_model_(model, best_model)

            _, test_acc = test(args, model, device, xtest, ytest, criterion, task_id, classes_per_task)
            best_acc.append(test_acc)

            representation_matrix = get_representation_matrix(model, device, xtrain)
            feature_list = update_GPM(representation_matrix, threshold, k, layer_w_thresholding, feature_list)
        
        else:
            temp_model = deepcopy(base_model)

            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr)

            feature_mat = list()
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                feature_mat.append(Uf)
            
            for epoch in range(epochs):
                train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, classes_per_task, projected=True, feature_mat=feature_mat)
                train(args, temp_model, device, xtrain, ytrain, temp_optimizer, criterion, task_id, classes_per_task)

                valid_loss, _ = test(args, model, device, xvalid, yvalid, criterion, task_id, classes_per_task)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= lr_factor
                        if lr < lr_min:
                            break
                    patience = lr_patience
                    adjust_learning_rate(optimizer, epoch, lr, lr_factor)

            set_model_(model, best_model)

            _, temp_test_acc = test(args, temp_model, device, xtest, ytest, criterion, task_id, classes_per_task)
            best_acc.append(temp_test_acc)

            representation_matrix = get_representation_matrix(model, device, xtrain)
            feature_list = update_GPM(representation_matrix, threshold, k, layer_w_thresholding, feature_list)

        for l in range(len(feature_list)):
            gradient_basis_number[l, task_id] = feature_list[l].shape[1]

        for t in range(tasks_number):
            xtest=data[t]['test']['x']
            ytest=data[t]['test']['y']

            _, acc_matrix[task_id, t] = test(args, model, device, xtest, ytest, criterion, t, classes_per_task)

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()

    best_acc=np.array(best_acc)
    final_acc = acc_matrix[-1, :].T
    best_final_gap = best_acc - final_acc

    return acc_matrix, gradient_basis_number, best_acc, final_acc, best_final_gap

def main(args):
    dataset = args.dataset
    seed = args.seed
    pc_valid = args.pc_valid
    thresholding_layer = args.thresholding_layer
    tasks_number = args.tasks_number
    classes_per_task = args.classes_per_task
    multihead = args.multihead
    hidden_dim = args.hidden_dim
    topk = args.topk
    repeat = args.repeat
    epochs = args.epochs
    fixed_threshold = args.fixed_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'cifar100':
        data, taskcla, _ = get_cifar100(
            device=device,
            tasks_number=tasks_number,
            classes_per_task=classes_per_task,
            get_feature=True,
            for_multihead=multihead,
            pc_valid=pc_valid,
            seed=seed
        )

    seeds = range(1, args.repeat + 1)
    if thresholding_layer == 'all':
        thresholding_layer = [[0,1,2,3,4]]
    elif thresholding_layer == 'one':
        thresholding_layer = [[i] for i in range(5)]
    elif thresholding_layer == 'none':
        thresholding_layer = [[]]
    elif thresholding_layer == 'accumulative':
        thresholding_layer = [list(range(0,i+1)) for i in range(5)]
        thresholding_layer += [list(range(i,5)) for i in range(5)]
    else:
        thresholding_layer = [[]]

    for layer_w_thresholding in thresholding_layer:
        all_acc_matrix = list()
        all_gradient_basis_numbers = list()
        all_best_acc = list()
        all_final_acc = list()
        all_best_final_gap = list()

        for seed in seeds:
            args.seed = seed

            acc_matrix, gradient_basis_number, best_acc, final_acc, best_final_gap = run_experiment(args, seed, device, data, taskcla, tasks_number, classes_per_task, multihead, hidden_dim, layer_w_thresholding, fixed_threshold, topk, epochs)

            all_acc_matrix.append(acc_matrix)
            all_gradient_basis_numbers.append(gradient_basis_number)
            all_best_acc.append(best_acc)
            all_final_acc.append(final_acc)
            all_best_final_gap.append(best_final_gap)

        avg_acc_matrix = np.mean(all_acc_matrix, axis=0)
        avg_gradient_basis_number = np.mean(all_gradient_basis_numbers, axis=0)
        avg_best_acc = np.mean(all_best_acc, axis=0)
        avg_final_acc = np.mean(all_final_acc, axis=0)
        avg_best_final_gap = np.mean(all_best_final_gap, axis=0)
        
        result = {
            "acc_matrix": avg_acc_matrix,
            "gradient_basis_number": avg_gradient_basis_number,
            "best_final": np.column_stack((avg_best_acc, avg_final_acc, avg_best_final_gap))
        }

        save_result(result, multihead, tasks_number, classes_per_task, hidden_dim, topk, layer_w_thresholding, fixed_threshold, repeat, './')

if __name__ == '__main__':
    # Training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of training epochs/task (default: 20)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid', default=0.05,type=float,
                        help='fraction of training data used for validation')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    
    # Experiment parameters
    parser.add_argument('--dataset', type=str, default='cifar100', metavar='D',
                        help='dataset for experiments')
    parser.add_argument('--hidden_dim', type=int, default=125, metavar='HD',
                        help='hidden dimension of MLP')
    parser.add_argument('--tasks_number', type=int, default=10, metavar='TN',
                        help='number of tasks (default: 10)')
    parser.add_argument('--classes_per_task', type=int, default=0, metavar='CPT',
                        help='number of classes per task')
    parser.add_argument('--topk', type=int, default=0, metavar='k',
                        help='fixed number of bases which added to GPM')
    parser.add_argument('--repeat', type=int, default=1, metavar='R',
                        help='how many times repeat experiments')
    parser.add_argument('--multihead', action='store_true',
                        help='Use multihead model')
    parser.add_argument('--thresholding_layer', default='all', type=str, metavar='T', choices=['none', 'one', 'accumulative', 'all'],
                        help='thresholding type (default: all) [candidates: none, one, accumulative, all]')
    parser.add_argument('--fixed_threshold', type=float, default=0., metavar='FT',
                        help='Setting threshold for determine number of bases for adding to GPM (default: linearly increase from 0.97)')

    args = parser.parse_args()

    main(args)