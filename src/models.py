import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy

import numpy as np

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, multihead, taskcla):
        super(MLP, self).__init__()
        self.act = OrderedDict()
        self.map = list()

        self.map.append(512)
        self.fc1 = nn.Linear(512, hidden_dim)

        self.map.append(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.map.append(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.map.append(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.multihead = multihead

        if multihead:
            self.taskcla = taskcla
            self.mh=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.mh.append(torch.nn.Linear(hidden_dim,n,bias=False))
        else:
            self.map.append(hidden_dim)
            self.sh = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.relu(x)

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.relu(x)

        self.act['fc3'] = x
        x = self.fc3(x)
        x = self.relu(x)

        self.act['fc4'] = x
        x = self.fc4(x)
        x = self.relu(x)

        if self.multihead:
            y=[]
            for t,i in self.taskcla:
                y.append(self.mh[t](x))
        
        else:
            self.act['sh'] = x
            y = self.sh(x)
            
        return y

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, lr, lr_factor):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=lr
        else:
            param_group['lr'] /= lr_factor  

def train(args, model, device, x,y, optimizer,criterion, task_id, classes_per_task, projected=False, feature_mat=[]):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b.cpu()]
        data, target = data.to(device), y[b.cpu()].to(device)
        optimizer.zero_grad()        
        output = model(data)

        if model.multihead:
            output = output[task_id]
        else:
            start_idx = task_id * classes_per_task
            end_idx = (task_id + 1) * classes_per_task

            output = output[:, start_idx:end_idx]
            target = target - start_idx  # Adjust target for current task

        loss = criterion(output, target)        
        loss.backward()

        if projected:
            # Gradient Projections 
            kk = 0
            for k, (name, params) in enumerate(model.named_parameters()):
                if params.grad is None:
                    continue

                if kk < len(feature_mat) and len(params.size()) == 2:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())

                    kk +=1
                elif (kk < len(feature_mat) and len(params.size()) == 1) and task_id != 0:
                    params.grad.data.fill_(0)


        optimizer.step()

def test(args, model, device, x, y, criterion, task_id, classes_per_task):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b.cpu()]
            data, target = data.to(device), y[b.cpu()].to(device)
            output = model(data)

            if model.multihead:
                output = output[task_id]
            else:
                start_idx = task_id * classes_per_task
                end_idx = (task_id + 1) * classes_per_task

                output = output[:, start_idx:end_idx]
                target = target - start_idx  # Adjust target for current task

            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def get_representation_matrix (model, device, x): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b.cpu()]
    example_data = example_data.to(device)

    model.eval()
    model(example_data)
    
    mat_list=[]
    act_key=list(model.act.keys())
    for i in range(len(model.map)):
        act = model.act[act_key[i]].detach().cpu().numpy()
        activation = act.transpose()
        mat_list.append(activation)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list    

def update_GPM (mat_list, threshold, k, thresholding_layer, feature_list=[]):
    print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            if i in thresholding_layer:
                topk = 0
            else:
                topk = k
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1 
            if topk == 0: 
                feature_list.append(U[:,0:r])
            elif topk > 0:
                k = min(U.shape[1], k)
                feature_list.append(U[:,0:k])
                
    else:
        for i in range(len(mat_list)):
            if i in thresholding_layer:
                topk = 0
            else:
                topk = k
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break

            # update GPM
            if topk == 0:
                Ui=np.hstack((feature_list[i],U[:,0:r]))  
            elif topk > 0:
                topk = min(U.shape[1], k)
                Ui=np.hstack((feature_list[i],U[:,0:k])) 
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list  