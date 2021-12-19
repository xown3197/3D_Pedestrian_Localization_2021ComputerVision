
# pylint: disable=too-many-statements
"""
Training and evaluation of a neural network which predicts 3D localization and confidence intervals
given 2d joints
"""

import copy
import os
import datetime
import logging
from collections import defaultdict
import sys
import time
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from .datasets import KeypointsDataset
from ..network import LaplacianLoss
from ..network.process import unnormalize_bi
from ..network.architectures import LinearModel
from ..utils import set_logger

knee_point = [9, 10, 12, 13]

class Trainer:
    def __init__(self, joints, epochs=100, bs=256, dropout=0.2, lr=0.002,
                 sched_step=20, sched_gamma=1, hidden_size=256, n_stage=3, r_seed=1, n_samples=100,
                 baseline=False, save=False, print_loss=False, sub=False):
        """
        Initialize directories, load the data and parameters for the training
        """

        # Initialize directories and parameters
        dir_out = os.path.join('data', 'models')
        if not os.path.exists(dir_out):
            warnings.warn("Warning: output directory not found, the model will not be saved")
        dir_logs = os.path.join('data', 'logs')
        if not os.path.exists(dir_logs):
            warnings.warn("Warning: default logs directory not found")
        assert os.path.exists(joints), "Input file not found"

        self.joints = joints
        self.num_epochs = epochs
        self.save = save
        self.print_loss = print_loss
        self.baseline = baseline
        self.lr = lr
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma
        n_joints = 17
        input_size = n_joints * 2
        self.output_size = 2
        # self.clusters = ['10', '20', '30', '>30']
        self.clusters = ['10']
        self.hidden_size = hidden_size
        self.n_stage = n_stage
        self.dir_out = dir_out
        self.n_samples = n_samples
        self.r_seed = r_seed

        # Loss functions and output names
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]

        if baseline:
            name_out = 'baseline-' + now_time
            self.criterion = nn.L1Loss().cuda()
            self.output_size = 1
        else:
            name_out = 'monoloco-' + now_time
            self.criterion = LaplacianLoss().cuda()
            self.output_size = 2
        self.criterion_eval = nn.L1Loss().cuda()
        
        ## tjkim Submodel
        self.sub_criterion = nn.MSELoss().cuda()
        self.sub_criterion2 = nn.L1Loss().cuda()

        if self.save:
            self.path_model = os.path.join(dir_out, name_out + '.pkl')
            self.path_model_sub = os.path.join(dir_out, name_out + '_sub.pkl')
            self.logger = set_logger(os.path.join(dir_logs, name_out))
            self.logger.info("Training arguments: \nepochs: {} \nbatch_size: {} \ndropout: {}"
                             "\nbaseline: {} \nlearning rate: {} \nscheduler step: {} \nscheduler gamma: {}  "
                             "\ninput_size: {} \nhidden_size: {} \nn_stages: {} \nr_seed: {}"
                             "\ninput_file: {}"
                             .format(epochs, bs, dropout, baseline, lr, sched_step, sched_gamma, input_size,
                                     hidden_size, n_stage, r_seed, self.joints))
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        # Select the device and load the data
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Device: ', self.device)

        # Set the seed for random initialization
        torch.manual_seed(r_seed)
        if use_cuda:
            torch.cuda.manual_seed(r_seed)

        # Dataloader
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.joints, phase=phase),
                                              batch_size=bs, shuffle=True) for phase in ['train', 'val']}

        self.dataset_sizes = {phase: len(KeypointsDataset(self.joints, phase=phase))
                              for phase in ['train', 'val', 'test']}

        # Define the model
        self.logger.info('Sizes of the dataset: {}'.format(self.dataset_sizes))
        print(">>> creating model")
        self.model = LinearModel(input_size=input_size, output_size=self.output_size, linear_size=hidden_size,
                                 p_dropout=dropout, num_stage=self.n_stage)
        self.model.to(self.device)
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        
        ## tjkim
        self.sub = sub
        if self.sub:
            print(">>> creating Missing Part Model")
            # self.sub_model = LinearModel(input_size=input_size, output_size=input_size, linear_size=256,
            #                         p_dropout=0.5, num_stage=1)
            self.sub_model = LinearModel(input_size=input_size, output_size=8, linear_size=2048,
                                    p_dropout=0.5, num_stage=1)
            self.sub_model.to(self.device)
            print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.sub_model.parameters()) / 1000000.0))

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)
        
        if self.sub:
            # MPM ~ Optimizer and scheduler
            self.optimizer2 = torch.optim.Adam(params=self.sub_model.parameters(), lr=lr)
            self.scheduler2 = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

    def train(self):

        # Initialize the variable containing model weights
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 1e6
        best_epoch = 0
        loss = 0
        loss_eval = 0
        epoch_losses_tr, epoch_losses_val, epoch_norms, epoch_sis = [], [], [], []
        
        epoch_losses_tr2, epoch_losses_val2, epoch_norms2, epoch_sis2 = [], [], [], []
        
        # sub_model_path = 'data/models/monoloco-211218-1942_sub.pkl'
        # self.sub_model.load_state_dict(torch.load(sub_model_path, map_location=lambda storage, loc: storage))

        for epoch in range(self.num_epochs):

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                    ## tjkim
                    if self.sub:
                        self.sub_model.train()
                    ##
                else:
                    self.model.eval()  # Set model to evaluate mode
                    ## tjkim
                    if self.sub:
                        self.sub_model.eval()
                    ##

                running_loss_tr = running_loss_eval = norm_tr = bi_tr = 0.0
                running_loss_tr2 = running_loss_eval2 = norm_tr2 = bi_tr2 = 0.0
                                # Iterate over data.
                if self.sub and epoch < 150:                                
                    ## submodel !!
                    for inputs, labels, names, kps in self.dataloaders[phase]:
                        # if '000407.txt' in names:
                        #     import pdb;pdb.set_trace()
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        # import pdb;pdb.set_trace()
                        ## tjkim
                        inputs_ = copy.deepcopy(inputs) 
                        if torch.rand(1) > -10:
                            
                            ## knee
                            knee_point = [9, 10, 12, 13]
                            '''
                            0 ~ *2 + 1 ~ 0 1
                            1 ~ *2 + 1 ~ 2 3
                            2 ~ *2 + 1 ~ 4 5
                            ...
                            16 ~ *2 + 1 ~ 32 33
                            '''
                            for kn in knee_point:
                                inputs_[:, kn*2:kn*2+1] = 1.0000e-02
                            
                            ## random ~
                            # mask = torch.randn(inputs.size()) > torch.rand(1) 
                            # inputs_[mask] = 1.0000e-02
                            
                        
                            
                            
                            # inputs += noise 
                        # zero the parameter gradients
                        self.optimizer2.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.sub_model(inputs_)

                            # outputs_eval = outputs[:, 0:1] if self.output_size == 2 else outputs
                            outputs_eval = outputs
                            try:
                                for i, kn in enumerate(knee_point):
                                    if i==0:
                                        loss = self.sub_criterion(outputs[:, i:i+1], inputs[:, kn*2:kn*2+1])
                                    else:
                                        loss += self.sub_criterion(outputs[:, i:i+1], inputs[:, kn*2:kn*2+1])
                                # loss = self.sub_criterion(outputs, inputs)  # L2 loss to evaluation
                                loss /= len(knee_point) / 2
                            except:
                                import pdb;pdb.set_trace()
                                
                            for i, kn in enumerate(knee_point):
                                if i==0:
                                        loss_eval = self.sub_criterion2(outputs[:, i:i+1], inputs[:, kn*2:kn*2+1])
                                else:
                                    loss_eval += self.sub_criterion2(outputs[:, i:i+1], inputs[:, kn*2:kn*2+1])
                            loss_eval /= len(knee_point) / 2
                            # loss_eval = self.sub_criterion2(outputs, inputs)  # L2 loss to evaluation

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer2.step()

                        # statistics
                        running_loss_tr2 += loss.item() * inputs.size(0)
                        running_loss_eval2 += loss_eval.item() * inputs.size(0)

                    epoch_loss = running_loss_tr2 / self.dataset_sizes[phase]
                    epoch_acc = running_loss_eval2 / self.dataset_sizes[phase]  # Average distance in meters
                    epoch_norm = float(norm_tr2 / self.dataset_sizes[phase])
                    epoch_si = float(bi_tr2 / self.dataset_sizes[phase])
                    if phase == 'train':
                        epoch_losses_tr2.append(epoch_loss)
                        epoch_norms2.append(epoch_norm)
                        epoch_sis2.append(epoch_si)
                    else:
                        epoch_losses_val2.append(epoch_acc)

                    if epoch % 5 == 1:
                        sys.stdout.write('\r' + 'subnet !! Epoch: {:.0f}   Training Loss: {:.3f}   Val Loss {:.3f}'
                                        .format(epoch, epoch_losses_tr2[-1], epoch_losses_val2[-1]) + '\t')

                    # deep copy the model
                    if phase == 'val' and epoch_acc < best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                        best_model_wts2 = copy.deepcopy(self.sub_model.state_dict())
                    if epoch_losses_tr2[-1] > 0.08:
                        continue
                #### main model !!! 
                if self.sub:          
                    self.sub_model.eval()     
                # Iterate over data.
                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    ## tjkim
                    ## sub_model
                    if self.sub:
                        inputs_ = self.sub_model(inputs)
                        for i, kn in enumerate(knee_point):
                                inputs[:, kn*2:kn*2+1] = inputs_[:, i:i+1].detach()
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        # if self.sub:
                            # outputs = self.model(inputs_)    
                        # else:
                        outputs = self.model(inputs)

                        outputs_eval = outputs[:, 0:1] if self.output_size == 2 else outputs

                        loss = self.criterion(outputs, labels)
                        loss_eval = self.criterion_eval(outputs_eval, labels)  # L1 loss to evaluation

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss_tr += loss.item() * inputs.size(0)
                    running_loss_eval += loss_eval.item() * inputs.size(0)

                epoch_loss = running_loss_tr / self.dataset_sizes[phase]
                epoch_acc = running_loss_eval / self.dataset_sizes[phase]  # Average distance in meters
                epoch_norm = float(norm_tr / self.dataset_sizes[phase])
                epoch_si = float(bi_tr / self.dataset_sizes[phase])
                if phase == 'train':
                    epoch_losses_tr.append(epoch_loss)
                    epoch_norms.append(epoch_norm)
                    epoch_sis.append(epoch_si)
                else:
                    epoch_losses_val.append(epoch_acc)
                try:
                    if epoch % 5 == 1:
                        sys.stdout.write('\n' + 'Epoch: {:.0f}   Training Loss: {:.3f}   Val Loss {:.3f}'
                                        .format(epoch, epoch_losses_tr[-1], epoch_losses_val[-1]) + '\t')
                except:
                    if epoch % 5 == 1:
                        sys.stdout.write('\n' + 'Epoch: {:.0f}   Training Loss: {:.3f}   Val Loss {:.3f}'
                                        .format(epoch, epoch_losses_tr[-1], epoch_acc) + '\t')
                # deep copy the model
                if phase == 'val' and epoch_acc < best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('\n\n' + '-'*120)
        self.logger.info('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best validation Accuracy: {:.3f}'.format(best_acc))
        self.logger.info('Saved weights of the model at epoch: {}'.format(best_epoch))

        if self.print_loss:
            epoch_losses_val_scaled = [x - 4 for x in epoch_losses_val]  # to compare with L1 Loss
            plt.plot(epoch_losses_tr[10:], label='Training Loss')
            plt.plot(epoch_losses_val_scaled[10:], label='Validation Loss')
            plt.legend()
            plt.show()

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if self.sub:
            self.sub_model.load_state_dict(best_model_wts2)

        return best_epoch

    def evaluate(self, load=False, model=None, debug=False, sub_model=None):

        # To load a model instead of using the trained one
        if load:
            self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
            ## tjkim
            self.sub_model.load_state_dict(torch.load(sub_model, map_location=lambda storage, loc: storage))

        # Average distance on training and test set after unnormalizing
        self.model.eval()
        self.sub_model.eval()
        dic_err = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # initialized to zero
        phase = 'val'
        batch_size = 5000
        dataset = KeypointsDataset(self.joints, phase=phase)
        size_eval = len(dataset)
        start = 0
        
        # Save the model and the results
        if self.save and not load:
            torch.save(self.model.state_dict(), self.path_model)
            torch.save(self.sub_model.state_dict(), self.path_model_sub)
            print('-'*120)
            self.logger.info("\nmodel saved: {} \n".format(self.path_model + self.path_model_sub))
        else:
            self.logger.info("\nmodel not saved\n")
            
        with torch.no_grad():
            for end in range(batch_size, size_eval+batch_size, batch_size):
                end = end if end < size_eval else size_eval
                inputs, labels, _, _ = dataset[start:end]
                start = end
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Debug plot for input-output distributions
                if debug:
                    debug_plots(inputs, labels)
                    sys.exit()

                # Forward pass
                
                ## tjkim
                ## sub_model
                if self.sub:
                    inputs_ = self.sub_model(inputs)
                    for i, kn in enumerate(knee_point):
                        inputs[:, kn*2:kn*2+1] = inputs_[:, i:i+1].detach()
                ## 
                
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                if not self.baseline:
                    outputs = unnormalize_bi(outputs)

                dic_err[phase]['all'] = self.compute_stats(outputs, labels, dic_err[phase]['all'], size_eval)

            print('-'*120)
            self.logger.info("Evaluation:\nAverage distance on the {} set: {:.2f}"
                             .format(phase, dic_err[phase]['all']['mean']))

            self.logger.info("Aleatoric Uncertainty: {:.2f}, inside the interval: {:.1f}%\n"
                             .format(dic_err[phase]['all']['bi'], dic_err[phase]['all']['conf_bi']*100))

            # Evaluate performances on different clusters and save statistics
            for clst in self.clusters:
                inputs, labels, size_eval = dataset.get_cluster_annotations(clst)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass on each cluster
                ## tjkim
                ## sub_model
                if self.sub:
                    # inputs_ = self.sub_model(inputs)
                    outputs_ = self.sub_model(inputs)
                    for i, kn in enumerate(knee_point):
                        inputs[:, kn*2:kn*2+1] = outputs_[i:i+1]
                ## 
                
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)                    
                # outputs = self.model(inputs)
                if not self.baseline:
                    outputs = unnormalize_bi(outputs)

                    dic_err[phase][clst] = self.compute_stats(outputs, labels, dic_err[phase][clst], size_eval)

                self.logger.info("{} error in cluster {} = {:.2f} for {} instances. "
                                 "Aleatoric of {:.2f} with {:.1f}% inside the interval"
                                 .format(phase, clst, dic_err[phase][clst]['mean'], size_eval,
                                         dic_err[phase][clst]['bi'], dic_err[phase][clst]['conf_bi'] * 100))



        return dic_err, self.model

    def compute_stats(self, outputs, labels_orig, dic_err, size_eval):
        """Compute mean, bi and max of torch tensors"""

        labels = labels_orig.view(-1, )
        mean_mu = float(self.criterion_eval(outputs[:, 0], labels).item())
        max_mu = float(torch.max(torch.abs((outputs[:, 0] - labels))).item())

        if self.baseline:
            return (mean_mu, max_mu), (0, 0, 0)

        mean_bi = torch.mean(outputs[:, 1]).item()

        low_bound_bi = labels >= (outputs[:, 0] - outputs[:, 1])
        up_bound_bi = labels <= (outputs[:, 0] + outputs[:, 1])
        bools_bi = low_bound_bi & up_bound_bi
        conf_bi = float(torch.sum(bools_bi)) / float(bools_bi.shape[0])

        dic_err['mean'] += mean_mu * (outputs.size(0) / size_eval)
        dic_err['bi'] += mean_bi * (outputs.size(0) / size_eval)
        dic_err['count'] += (outputs.size(0) / size_eval)
        dic_err['conf_bi'] += conf_bi * (outputs.size(0) / size_eval)

        return dic_err


def debug_plots(inputs, labels):
    inputs_shoulder = inputs.cpu().numpy()[:, 5]
    inputs_hip = inputs.cpu().numpy()[:, 11]
    labels = labels.cpu().numpy()
    heights = inputs_hip - inputs_shoulder
    plt.figure(1)
    plt.hist(heights, bins='auto')
    plt.show()
    plt.figure(2)
    plt.hist(labels, bins='auto')
    plt.show()
