import pdb 
import scipy.io
from scipy.stats import pearsonr
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
# from torch_geometric.data import Data, DataLoader, DenseDataLoader
from tensorboardX import SummaryWriter

#### 设置随机种子
import random
# random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

import swats

############################## 数据读取

graph_convolution_data = scipy.io.loadmat('graph_convolution_data_face360.mat')


graph_convolution_data.keys()
C_resting = graph_convolution_data['C_resting']
task = graph_convolution_data['task']

C_resting = torch.from_numpy(C_resting).to(torch.float32)
task = torch.from_numpy(task).to(torch.float32)
task = task.unsqueeze(2) if task.dim()==2 else task


############################################### parameter 参数设置
random_seeds = np.load('random_seeds.npy')     ########### 96 
rep_num = 100; print(rep_num)
single_run = 0; rep_base = 0
epochs = 2000
milestones = [1900,6000,9000]; gamma=0.1
stepnum = 2
layernum = 1
lr = [0.01,0.001,0.001,0.001,0.001]  ## 0.003
nesterov = True  
amsgrad = True  ### False
weight_decay = 0
use_SGD = 1;  use_adam = 0; use_RMSprop = 0; use_swats = 0;
residue = 1   
add_loop = True 
nonsharing = 1    
division = [8,1,1]  #[128*6, 128*1, 1] ##[32*20, 32*7, 1] ##### [6,2,2]  # 
division_method = 1 if division[0]>10 else 0
add_noise = 1; add_noise_y = 0;   
initial_gain = 0.1  
dropout = 0  
perm_individual = 0

print(stepnum,residue,add_loop,nonsharing,division,l1_reg,lr,add_noise,nesterov,initial_gain,dropout)
print(add_noise,noise1,noise2,add_noise_y)

# prefix = 'NET'+'step'+str(stepnum)+'ns'+str(nonsharing)+'noise'+str(add_noise)+'Res'+str(residue)+'L'
prefix = 'NET'+'step'+str(stepnum)+'ns'+str(nonsharing)+'noise'+str(add_noise)+'Res'+str(residue
    )+'iG'+str(initial_gain)+'SpC'+str(sparse_correction)+'Nes'+str(np.sum(nesterov))+'L'
print(prefix)
save_net = 0
full_batch = 0; BATCH_SIZE = 128 #256    # 批训练的数据个数
if full_batch==0:
 

    epochs = 500
    lr = [0.01,0.001,0.001,0.001,0.001]  ## 0.003
    milestones = [300,400,6000,8000]; gamma=0.1

   

myloss = 1; only_sse = 1; only_R2 = 0; both_R2_sse = 0  
l1loss = 0

out_old = -1
torch.autograd.set_detect_anomaly(True)

task_num = 0; print(task_num)
target_region = 35; print(target_region)
resting = 1
sparse_thresh_percentile = 0
outmask = torch.zeros(task.size(1),dtype=torch.bool); outmask[target_region] = 1
# outmask = torch.ones(246,dtype=torch.bool);

group_initial = 0; ones_initial = 1  

use_writer = 0

multi = 1
if multi > 1:
    featureTrans = True
else:
    featureTrans = False

val_use = 0
val_thresh = -1
early_stopping = 0

sub_net = 1  
sub_net_num = 320#280 #180 
activation_thresh = 0.9750   #face-shape:  0.9750  face-avg: 0.9557

sub_net2 = 0
sub_net_num2 = 2
# region_num = 147

############################################################### 数据预处理
task_one = task[:,:,task_num]
task_one = task_one.unsqueeze(2)


if norm:
    
    task_norm = torch.std(task_one, dim=0, keepdim=True)  
    task_one = task_one/task_norm


C_data = C_resting


################################ 激活子网络
if sub_net:
    group = torch.mean(task_one.abs(),0)
    # group = torch.mean(task_one,0)
    # _,sort_index = torch.sort(group.abs(),0,descending=True) 
    group_sort,sort_index = torch.sort(group.abs(),0,descending=False)
    sub_net_num = torch.sum(group_sort < activation_thresh)
    print(sub_net_num)
    sort_index_truncate = sort_index.squeeze()[0:sub_net_num]
    
    C_data[:,sort_index_truncate,:]=0
    C_data[:,:,sort_index_truncate]=0


if sub_net2:
    group = torch.mean(task_one,0)
    _,sort_index = torch.sort(group.abs(),0,descending=True) 
    sort_index_truncate = sort_index.squeeze()[0:sub_net_num2]
    outmask = torch.zeros(246,dtype=torch.bool);
    outmask[sort_index_truncate] = True
    # lr = [1,0.1,0.01,0.01,0.01]
    lr = [0.01,0.001,0.001,0.001,0.001]

################################# 准备数据集
weight_mask = torch.mean(C_data,0) != 0
if predict_behavior:
    task_one = torch.ones(y.size(0),task_one.size(1))
    task_one[:,target_region] = y[:,0]
    task_one = task_one.unsqueeze(2)

num_data = task_one.size(0)
if division_method == 0:
    num_train = int(num_data*division[0]//10)  ##division[0]  ##
    num_val = int(num_data*division[1]//10)  ##division[1]   ##
    num_test = num_data - num_train - num_val
elif division_method == 1:
    num_train = division[0]  ##
    num_val = division[1]   ##  ##
    num_test = num_data - num_train - num_val

def seed_torch(seed=0):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

class myLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        sse = torch.sum(torch.pow((x - y), 2))
        sst = torch.sum(vy ** 2)
        
        corr = torch.sum(vx * vy) / ( torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) )
        R2 = corr ** 2
        mse = torch.mean(torch.pow((x - y), 2))

        # loss = -1*corr + mse
        # loss = -1*corr * mse
        if only_R2:
            loss = (1-R2) 
        elif only_sse:
            loss = sse/sst
        elif both_R2_sse:
            loss = ( sse/sst + (1-R2) ) / 2

        return loss


############################ 网络搭建
num_nodes = C_data.size(1)
from wsgcn_multi import *   #wsgcn_multi
class wsgcn_multi_net(torch.nn.Module):
    def __init__(self, multi=1, in_channels=1, out_channels=1):
        super(wsgcn_multi_net, self).__init__()

        self.weight = Parameter(torch.Tensor(num_nodes, multi)) ##确定维度
        if multi > 1:
            self.weight2 = Parameter(torch.Tensor(multi, num_nodes, 1))
        else:
            self.weight2 = Parameter(torch.Tensor(num_nodes, num_nodes))       
        
        self.bias = Parameter(torch.Tensor(num_nodes,1))
        
        # pdb.set_trace()  

        self.conv = wsgcn_multi(in_channels, out_channels, num_nodes, multi=multi, 
                multi_aggre='concat', featureTrans=featureTrans, nonsharing=False) 

        if nonsharing:
            self.convs = nn.ModuleList([wsgcn_multi(in_channels, out_channels, num_nodes, 
                multi=multi, multi_aggre='concat', featureTrans=featureTrans, nonsharing=True,
                initial_gain=initial_gain,weight_mask=weight_mask) for i in range(stepnum)])

        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight, initial_gain)
        nn.init.xavier_normal_(self.weight2, initial_gain)
        
        nn.init.zeros_(self.bias)
        

    def forward(self, adj, x, timestep=1):
        if nonsharing:

            # if timestep==1:
                # x = F.dropout(x, p = dropout, training=self.training)  
            for i in range(timestep-1):          
                x = self.convs[i](x, adj, self.weight2, self.weight, self.bias, add_loop=add_loop, residue=residue)  #+ x
                x = F.dropout(x, p = dropout, training=self.training)                  
            x = self.convs[-1](x, adj, self.weight2, self.weight, self.bias, add_loop=add_loop, residue=residue)
            
               
        else:

            if timestep==1:
                x = F.dropout(x, p = dropout, training=self.training)  
            for i in range(timestep-1):
                x = self.conv(x, adj, self.weight2, self.weight, self.bias, add_loop=add_loop, residue=residue)   #+ x
                x = F.dropout(x, p = dropout, training=self.training)
            x = self.conv(x, adj, self.weight2, self.weight, self.bias, add_loop=add_loop, residue=residue)
            
            #################    
        return x  

#### build models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net = [wsgcn_multi_net(multi=multi, in_channels=1, out_channels=1).to(device)  for i in range(layernum)]
import torch.optim as optim
##############################
if l1loss:
    criterion = nn.L1Loss()  ## 采用MSE的损失函数
elif myloss:
    criterion = myLoss()
else:
    criterion = nn.MSELoss()  ## 采用MSE的损失函数

def test(layer, data_loader):
    with torch.no_grad():
        net[layer].eval()
        loss_all = 0
        corr_all = 0
        for i, batch in enumerate(data_loader):  # 每一步 loader 释放一小批数据用来学习
        
            batch_cdata, batch_task = batch[0].to(device), batch[1].to(device)             

            if group_initial:
                batch_task_initials = torch.mean(task_one[train_dataset.indices,:,:], 0).to(device)
            elif ones_initial:
                batch_task_initials = torch.ones(num_nodes,1).to(device) 

            for k in range(layer):                  
                batch_task_initials = net[k](batch_cdata,batch_task_initials, timestep=stepnum)
                    
            outputs = net[layer](batch_cdata,batch_task_initials, timestep=stepnum)
          
            outputs_old = outputs; batch_task_old = batch_task
            outputs = outputs[:,outmask,:]; batch_task = batch_task[:,outmask,:];
            
            if layer==out_old:
                loss = criterion(outputs_old, batch_task_old)
            else:
                loss = criterion(outputs, batch_task)
            
            loss_all += batch_task.size(0) * loss.item()

            outputs_numpy=outputs.detach().cpu().squeeze(-1).numpy()
            batch_task_numpy=batch_task.cpu().squeeze(-1).numpy()
            
            corr2 = np.corrcoef(outputs_numpy.T, batch_task_numpy.T)
            corr2_region = np.diag(corr2[0:outputs_numpy.shape[1],outputs_numpy.shape[1]:])
                                      
            SSE = np.sum( (batch_task_numpy - outputs_numpy)**2 )
            SST = np.sum( (batch_task_numpy - np.mean(batch_task_numpy))**2 ) 
            SSR = np.sum( (outputs_numpy - np.mean(batch_task_numpy))**2 ) 

        return loss_all / len(test_dataset), corr2_region, SSE/len(test_dataset)


##########################

test_accs = [ [[] for i in range(layernum)] for j in range(rep_num) ]
test_losses = [ [[] for i in range(layernum)] for j in range(rep_num) ]
test_loss_final = [ [[] for i in range(layernum)] for j in range(rep_num) ]
test_accs_final = [ [[] for i in range(layernum)] for j in range(rep_num) ]

net_weights_sharing = torch.zeros(rep_num, num_nodes, num_nodes)
if nonsharing:
    net_weights_nonsharing = torch.zeros(rep_num, stepnum, num_nodes, num_nodes)
    net_bias_nonsharing = torch.zeros(rep_num, stepnum, num_nodes, 1)

for rep in range(rep_num):

    if perm_individual:         ####### permutation test 随机数必须在外部生成，保证每次都一样
        # perm=torch.randperm(C_data.size(1))
        # C_data = C_data[:,perm,:][:,:,perm]

        perm=torch.randperm(C_data.size(0))
        C_data = C_data[perm,:,:]

    torch_dataset = Data.TensorDataset(C_data, task_one)

    rep += rep_base
    if single_run:
        rep = single_run
    print('#################### Repetition num:');print(rep)
    random_seed = random_seeds[rep] #torch.randint(2**30,(1,)).item()  # 
    # random_seeds.append(random_seed)
    seed_torch(seed=random_seed)
    train_dataset,val_dataset,test_dataset=Data.random_split(torch_dataset,[num_train,num_val,num_test])

    train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    # train_val_index = list(train_dataset.indices)
    # train_val_index.extend(list(val_dataset.indices))
    # train_val_dataset = torch.utils.data.Subset(torch_dataset,train_val_index)
    rep -= rep_base
    
    if full_batch:
        BATCH_SIZE = num_train + num_val      # 批训练的数据个数

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )
    # train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
    # 						 shuffle=True,num_workers=1)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=num_val, 
                             shuffle=True,num_workers=1)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=num_test, 
    						 shuffle=True,num_workers=1)
    train_val_loader = Data.DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True,num_workers=1)


    #################
    if use_writer:
        # writer = SummaryWriter(comment="-loss_LR100_batch_task")
        writer = SummaryWriter("runs/timestep3")
        n_iter = 0  ## for visualization
    
    for layer in range(layernum):
        net[layer].reset_parameters()
        if nonsharing:
            for step in range(stepnum):
                net[layer].convs[step].reset_parameters()

    for layer in range(layernum):

        # optimizer = optim.SGD([{'params': net[j].parameters()} for j in range(layer+1)], lr=lr[layer], momentum=0.9)
        if use_SGD:
            optimizer = optim.SGD(net[layer].parameters(), lr=lr[layer], momentum=0.9, nesterov=nesterov, weight_decay=weight_decay)  ##lr不要太小，否则学习的太慢
        elif use_adam:
            optimizer = optim.Adam(net[layer].parameters(), lr=lr[layer], amsgrad=amsgrad) ## adam虽然训练超快，但是过拟合严重
        elif use_swats:
            optimizer = swats.SWATS(net[layer].parameters(), lr=lr[layer], nesterov=nesterov, amsgrad=amsgrad, verbose=True)
        elif use_RMSprop:
            optimizer = optim.RMSprop(net[layer].parameters(), lr=lr[layer], momentum=0,centered=True)

        acc_best = 0
        loss_best = 10000

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        for epoch in range(epochs):   
            for i, batch in enumerate(train_val_loader):  
                
                # pdb.set_trace()
                net[layer].train()
                batch_cdata, batch_task = batch[0].to(device), batch[1].to(device)

                if add_noise:
                    if noise1:
                        var_batch = torch.var(batch_cdata)
                    elif noise2:
                        var_batch = torch.var(batch_cdata,0)
                    batch_cdata += torch.randn_like(batch_cdata)*torch.sqrt(var_batch/add_noise)
                if add_noise_y:
                    var_batch = torch.var(batch_task)
                    batch_task += torch.randn_like(batch_task)*torch.sqrt(var_batch/add_noise_y)

                ## zero the parameter gradients
                optimizer.zero_grad()

                ## forward + backward + optimize
                if group_initial:
                    batch_task_initials = torch.mean(task_one[train_dataset.indices,:,:], 0).to(device)
                elif ones_initial:
                    batch_task_initials = torch.ones(num_nodes,1).to(device) 
                
                for k in range(layer):                      
                    batch_task_initials = net[k](batch_cdata,batch_task_initials, timestep=stepnum)

                outputs = net[layer](batch_cdata,batch_task_initials, timestep=stepnum)
            
                
                outputs_old = outputs; batch_task_old = batch_task
                outputs = outputs[:,outmask,:]; batch_task = batch_task[:,outmask,:];

                if layer==out_old:
                    loss = criterion(outputs_old, batch_task_old)
                else:
                    loss = criterion(outputs, batch_task)

                

                loss.backward()
                optimizer.step()  

                
                outputs_numpy=outputs.detach().cpu().squeeze(-1).numpy()
                batch_task_numpy=batch_task.cpu().squeeze(-1).numpy()

                corr2 = np.corrcoef(outputs_numpy.T, batch_task_numpy.T)
                corr2_region = np.diag(corr2[0:outputs_numpy.shape[1],outputs_numpy.shape[1]:])

                if use_writer:
	                writer.add_scalars('Loss', {'train':loss.item()}, n_iter)
	                writer.add_scalars('Corr', {'train':corr2_region}, n_iter)
	                n_iter += 1
                # pdb.set_trace()
                # print('loss: %.3f' % (loss.item()))
                print('[Rep: %d,epoch: %d,iteration: %5d] train_loss: %.3f train_corr_individual: %.3f' %
                          (rep, epoch + 1, i + 1, loss.item()*100, corr2_region))

            test_loss, test_corr_individual, test_msst = test(layer,test_loader) 

            scheduler.step()
            
            if loss < loss_best:        #test_corr_individual > acc_best:
                acc_best = test_corr_individual
                loss_best = loss
                if save_net:
                    for sv in range(layer+1):
                        torch.save(net[sv], 'saved_net/'+prefix+str(sv)+'.pkl') 

            if use_writer:
	            writer.add_scalars('Loss', {'test':test_loss}, n_iter)
	            writer.add_scalars('Corr', {'test':test_corr_individual}, n_iter)

            print('layer: %d epoch: %d test_loss: %.3f test_corr_individual: %.3f test_msst: %.3f'  %
                      (layer, epoch + 1, test_loss*100, test_corr_individual, test_msst))
            
            test_accs[rep][layer].append(test_corr_individual)
            test_losses[rep][layer].append(test_loss)
        
        if save_net:
            for sv in range(layer+1):
                net[sv] = torch.load('saved_net/'+prefix+str(sv)+'.pkl')
            test_loss, test_corr_individual, test_msst = test(layer,test_loader)

        test_loss_final[rep][layer] = test_loss
        test_accs_final[rep][layer] = test_corr_individual    

    net_weights_sharing[rep,:,:] = net[layer].weight2
    if nonsharing:
    	for step in range(stepnum):
            net_weights_nonsharing[rep,step,:,:] = net[layer].convs[step].weight_nonsharing
            net_bias_nonsharing[rep,step,:,:] = net[layer].convs[step].bias_nonsharing
    	
    if use_writer:
    	writer.close()


############################################################################  group result
if sub_net:
    prefix = 'NET'+'step'+str(stepnum)+'ns'+str(nonsharing)+'noise'+str(add_noise)+'Res'+str(residue
    )+'iG'+str(initial_gain)+'SpC'+str(sparse_correction)+'Nes'+str(np.sum(nesterov))+'L'+str(sub_net_num.item())
print(stepnum,residue,add_loop,nonsharing,division,l1_reg,lr,add_noise,nesterov,initial_gain,dropout)
print(add_noise,noise1,noise2,add_noise_y,element_sparse,group_sparse,l2_sparse,sparse_correction,milestones)
print(task_num,target_region,sub_net,sub_net_num,prefix)


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)
# individual=batch_task.cpu().squeeze().numpy()
individual = task_one.squeeze().numpy() 
group = np.nanmean(individual,0)
group_abs = np.nanmean(np.abs(individual),0)
indi_std = np.std(individual,0)
group_sort = np.sort(group)
sort_index = np.argsort(np.abs(group)) 
sort_index_abs = np.argsort(np.abs(group_abs)) 

from scipy import stats 
t_stat, pval = stats.ttest_1samp(individual,0,axis=0)
pval2 = pval < (0.01/360)


########################  结果统计

test_accs = np.asarray(test_accs)
test_losses = np.asarray(test_losses)

test_accs_final = np.asarray(test_accs_final)
test_loss_final = np.asarray(test_loss_final)
print('test_accs_final:');print(test_accs_final.T)
print('test_loss_final:');print(test_loss_final.T)

print('test_accs_final_mean:');print(np.mean(test_accs_final,0))
print('test_loss_final_mean:');print(np.mean(test_loss_final,0))

