import argparse
from ast import Num
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import pdb

import sys
# sys.path.append('/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal')
# sys.path.append('/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood')
from opencood.models.sub_modules.DA_module import DomainAdaptationModule, DA_feature_Discriminator,DA_feature_Discriminator_V1,DA_feature_Discriminator_V2,DA_feature_Discriminator_V3,DA_feature_Discriminator_V4

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset#, build_dataset_da




def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str,required=True,#default="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_fax_deformable.yaml",
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--model', default='',
                        help='for fine-tuned training path')
    parser.add_argument('--model_source', default='',
                        help='for fine-tuned training path')
    parser.add_argument('--model_target', default='',
                        help='for fine-tuned training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")
    opt = parser.parse_args()
    return opt



def GRL_training(opt,hypes,device,source_train_loader,target_train_loader,val_loader):


    print('Creating Model')
    model = train_utils.create_model(hypes)
    DA_module = DomainAdaptationModule(hypes['model']['args'])

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        DA_module.to(device)
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(source_train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        print('Loaded model from {}'.format(saved_path))

    else:
        if opt.model:
            saved_path = train_utils.setup_train(hypes)
            model_path = opt.model
            init_epoch = 0
            # pretrained_state = torch.load(os.path.join(model_path,'latest.pth'))
            pretrained_state = torch.load(model_path)
            model_dict = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_state)
            model.load_state_dict(model_dict)
            print('Loaded pretrained model from {}'.format(model_path))
        else:
            init_epoch = 0
            # if we train the model from scratch, we need to create a folder
            # to save the model,
            saved_path = train_utils.setup_train(hypes)
    # record training
    writer = SummaryWriter(saved_path)

    #log in txt file
    # open file in write mode
    txt_path = os.path.join(saved_path, 'training_eval_log.txt')
    txt_log = open(txt_path, "w")

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('GRL Training start')

    epoches = hypes['train_params']['epoches']
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(source_train_loader), leave=True)
        target_train_loader_iter = iter(target_train_loader)
        # for iteration, (source_batch_data, target_batch_data) in enumerate(zip(source_train_loader, target_train_loader)):
        for iteration, source_batch_data in enumerate(source_train_loader):

            try:
                target_batch_data = next(target_train_loader_iter)
            except StopIteration:
                target_train_loader_iter = iter(target_train_loader)
                target_batch_data = next(target_train_loader_iter)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            source_batch_data = train_utils.to_device(source_batch_data, device)
            target_batch_data = train_utils.to_device(target_batch_data, device)

            total_batch_data = [source_batch_data['ego'], target_batch_data['ego']]

            if not opt.half:
                ouput_dict = model(total_batch_data)
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(total_batch_data)
                    final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                    
            ####DA Loss#####
            da_loss = DA_module(ouput_dict)
            losses = sum(loss for loss in da_loss.values())
            da_loss['DA_training'] = hypes['train_params']['DA_training']
            # print(final_loss.item())

            Total_loss = losses + final_loss
            da_loss['GRL'] = True
            criterion.logging_da(epoch, iteration, len(source_train_loader), writer, da_loss, pbar=pbar2)

            pbar2.update(1)   
            # back-propagation
            if not opt.half:
                Total_loss.backward()
                optimizer.step()
            else:
                scaler.scale(Total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + iteration)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])
                
                    final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                            valid_ave_loss))
            
            txt_log.write('At epoch' + str(epoch+1)+', the validation loss is '+ str(valid_ave_loss) + '    save in '+ os.path.join(saved_path,'net_epoch%d.pth' % (epoch + 1)) + '\n')

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                    os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

    print('Training Finished, checkpoints saved to %s' % saved_path)
    # close file
    txt_log.close()




def ADDA_training(opt,hypes,device,source_train_loader,target_train_loader,val_loader):

    print('Creating Model')
    model = train_utils.create_model(hypes)
    target_model = train_utils.create_model(hypes)### for taget domain model
    DA_Discriminator = DA_feature_Discriminator(hypes['model']['args']['DA_feature_head'])
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        target_model.to(device)
        # DA_module.to(device)
        DA_Discriminator.to(device)

    criterion = train_utils.create_loss(hypes)
    criterion_da = nn.BCEWithLogitsLoss()
    # criterion_da = nn.BCELoss()
    # criterion_da = nn.CrossEntropyLoss()

    optimizer_D = optim.Adam(DA_Discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(source_train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)
    scheduler_D = train_utils.setup_lr_schedular(hypes, optimizer_D, num_steps)

    if opt.model_target and opt.model_source:##### Loaded pretrained model
        saved_path = train_utils.setup_train(hypes)
        source_model_path = opt.model_source
        target_model_path = opt.model_target
        init_epoch = 0
        # pretrained_state = torch.load(os.path.join(model_path,'latest.pth'))
        source_pretrained_state = torch.load(source_model_path)
        model_dict = model.state_dict()
        source_pretrained_state = {k: v for k, v in source_pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(source_pretrained_state)
        model.load_state_dict(model_dict)
        print('Loaded pretrained source model from {}'.format(source_model_path))

        target_pretrained_state = torch.load(target_model_path)
        model_dict = target_model.state_dict()
        target_pretrained_state = {k: v for k, v in target_pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(target_pretrained_state)
        target_model.load_state_dict(model_dict)
        print('Loaded pretrained target model from {}'.format(target_model_path))
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # record training
    writer = SummaryWriter(saved_path)
    #log in txt file
    # open file in write mode
    txt_path = os.path.join(saved_path, 'training_eval_log.txt')
    txt_log = open(txt_path, "w")

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('ADDA Training start')


    da_loss = {}
    da_loss['DA_training'] = hypes['train_params']['DA_training']
    epoches = hypes['train_params']['epoches']
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(source_train_loader), leave=True)
        target_train_loader_iter = iter(target_train_loader)

        for iteration, source_batch_data in enumerate(source_train_loader):

            try:
                target_batch_data = next(target_train_loader_iter)
            except StopIteration:
                target_train_loader_iter = iter(target_train_loader)
                target_batch_data = next(target_train_loader_iter)
            

            target_model.train()
            DA_Discriminator.train()

            source_batch_data = train_utils.to_device(source_batch_data, device)
            target_batch_data = train_utils.to_device(target_batch_data, device)

            if not opt.half:
                with torch.no_grad():
                    ouput_dict_source = model(source_batch_data['ego'])
                ouput_dict_target_t = target_model(target_batch_data['ego'])
                ouput_dict_target_s = target_model(source_batch_data['ego'])
                final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])

                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        ouput_dict_source = model(source_batch_data['ego'])
                    ouput_dict_target_t = target_model(target_batch_data['ego'])
                    ouput_dict_target_s = target_model(source_batch_data['ego'])
                    final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
                    # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                    

            # create labels for source and target domains
            batch_size = hypes['train_params']['batch_size']
            source_labels = torch.ones(batch_size, 1).to(device)
            target_labels = torch.zeros(batch_size, 1).to(device)


            # train discriminator with source domain
            DA_Discriminator.zero_grad()
            outputs = DA_Discriminator(ouput_dict_source['fused_feature'])
            d_loss_source = criterion_da(outputs, source_labels)
            # d_loss_source.backward()

            # train discriminator with target domain
            outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'].detach())
            d_loss_target = criterion_da(outputs, target_labels)
            # d_loss_target.backward()

            # update discriminator parameters
            d_loss = d_loss_source + d_loss_target
            d_loss.backward()
            optimizer_D.step()
            da_loss['D_loss'] = d_loss
            da_loss['D_loss_s'] = d_loss_source
            da_loss['D_loss_t'] = d_loss_target

            if epoch <=1:#### In the initial stages of training, the discriminator can be trained for a few epochs before starting to train the generator.
                with torch.no_grad():
                    # train generator
                    DA_Discriminator.zero_grad()
                    optimizer.zero_grad()
                    target_model.zero_grad()
                    outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                    G_loss = criterion_da(outputs, source_labels)
                    # G_loss.backward()
                    Total_loss = G_loss*10.0 + final_loss
                    da_loss['G_loss'] = G_loss
                    # back-propagation
                    # Total_loss.backward() 
                    optimizer.step()
            else:
                    # train generator
                    DA_Discriminator.zero_grad()
                    optimizer.zero_grad()
                    target_model.zero_grad()
                    outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                    G_loss = criterion_da(outputs, source_labels)
                    # G_loss.backward()
                    Total_loss = G_loss + final_loss
                    da_loss['G_loss'] = G_loss
                    # back-propagation
                    if not opt.half:
                        Total_loss.backward() 
                        optimizer.step()
                    else:
                        scaler.scale(Total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            pbar2.update(1)   
            criterion.logging_da(epoch, iteration, len(source_train_loader), writer, da_loss, pbar=pbar2)
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + iteration)
                scheduler_D.step_update(epoch * num_steps + iteration)


        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    target_model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = target_model(batch_data['ego'])
                
                    final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                            valid_ave_loss))
            
            txt_log.write('At epoch' + str(epoch+1)+',  the validation loss is'+ str(valid_ave_loss) + 'save in '+ str(os.path.join(saved_path,'net_epoch%d.pth' % (epoch + 1))) + '\n')

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(target_model.state_dict(),
                    os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            torch.save(DA_Discriminator.state_dict(),
                    os.path.join(saved_path,
                                    'Dis_net_epoch%d.pth' % (epoch + 1)))

    print('Training Finished, checkpoints saved to %s' % saved_path)
    # close file
    txt_log.close()



def Dual_DA_training(opt,hypes,device,source_train_loader,target_train_loader,val_loader):

    print('Creating Model')
    model = train_utils.create_model(hypes)
    target_model = train_utils.create_model(hypes)### for taget domain model
    # DA_Discriminator = DA_feature_Discriminator(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator_E = DA_feature_Discriminator(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator = DA_feature_Discriminator_V1(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator_E = DA_feature_Discriminator_V1(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator = DA_feature_Discriminator_V2(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator_E = DA_feature_Discriminator_V2(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator = DA_feature_Discriminator_V3(hypes['model']['args']['DA_feature_head'])
    # DA_Discriminator_E = DA_feature_Discriminator_V3(hypes['model']['args']['DA_feature_head'])
    DA_Discriminator = DA_feature_Discriminator_V4(hypes['model']['args']['DA_feature_head'])
    DA_Discriminator_E = DA_feature_Discriminator_V4(hypes['model']['args']['DA_feature_head'])
    # DA_feature_Discriminator_V1
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        target_model.to(device)
        # DA_module.to(device)
        DA_Discriminator.to(device)
        DA_Discriminator_E.to(device)

    criterion = train_utils.create_loss(hypes)
    criterion_da = nn.BCEWithLogitsLoss()
    # criterion_da = nn.BCELoss()
    # criterion_da = nn.CrossEntropyLoss()

    optimizer_D = optim.Adam(DA_Discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    optimizer_D_E = optim.Adam(DA_Discriminator_E.parameters(), lr=0.00002, betas=(0.5, 0.999))

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(source_train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)
    scheduler_D = train_utils.setup_lr_schedular(hypes, optimizer_D, num_steps)
    scheduler_D_E = train_utils.setup_lr_schedular(hypes, optimizer_D_E, num_steps)

    if opt.model_target and opt.model_source:##### Loaded pretrained model
        saved_path = train_utils.setup_train(hypes)
        source_model_path = opt.model_source
        target_model_path = opt.model_target
        init_epoch = 0
        # pretrained_state = torch.load(os.path.join(model_path,'latest.pth'))
        source_pretrained_state = torch.load(source_model_path)
        model_dict = model.state_dict()
        source_pretrained_state = {k: v for k, v in source_pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(source_pretrained_state)
        model.load_state_dict(model_dict)
        print('Loaded pretrained source model from {}'.format(source_model_path))

        target_pretrained_state = torch.load(target_model_path)
        model_dict = target_model.state_dict()
        target_pretrained_state = {k: v for k, v in target_pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(target_pretrained_state)
        target_model.load_state_dict(model_dict)
        print('Loaded pretrained target model from {}'.format(target_model_path))
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # record training
    writer = SummaryWriter(saved_path)
    #log in txt file
    # open file in write mode
    txt_path = os.path.join(saved_path, 'training_eval_log.txt')
    txt_log = open(txt_path, "w")

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Dual_DA Training start')


    da_loss = {}
    da_loss['DA_training'] = hypes['train_params']['DA_training']
    epoches = hypes['train_params']['epoches']
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(source_train_loader), leave=True)
        target_train_loader_iter = iter(target_train_loader)

        for iteration, source_batch_data in enumerate(source_train_loader):

            try:
                target_batch_data = next(target_train_loader_iter)
            except StopIteration:
                target_train_loader_iter = iter(target_train_loader)
                target_batch_data = next(target_train_loader_iter)
            

            target_model.train()
            DA_Discriminator.train()
            DA_Discriminator_E.train()

            source_batch_data = train_utils.to_device(source_batch_data, device)
            target_batch_data = train_utils.to_device(target_batch_data, device)

            # if not opt.half:
            #     with torch.no_grad():
            #         ouput_dict_source = model(source_batch_data['ego'])
            #         ouput_dict_target_t = target_model(target_batch_data['ego'])
            #         ouput_dict_target_s = target_model(source_batch_data['ego'])
            #         final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])

            #     # first argument is always your output dictionary,
            #     # second argument is always your label dictionary.
            #     # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
            # else:
            #     with torch.cuda.amp.autocast():
            #         with torch.no_grad():
            #             ouput_dict_source = model(source_batch_data['ego'])
            #             ouput_dict_target_t = target_model(target_batch_data['ego'])
            #             ouput_dict_target_s = target_model(source_batch_data['ego'])
            #             final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
            #         # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                    

            # # create labels for source and target domains
            # # batch_size = hypes['train_params']['batch_size']
            # # source_labels = torch.ones(batch_size, 1).to(device)
            # # target_labels = torch.zeros(batch_size, 1).to(device)


            # # train discriminator with source domain
            # DA_Discriminator.zero_grad()
            # outputs = DA_Discriminator(ouput_dict_source['fused_feature'])
            # source_labels = torch.ones_like(outputs).to(device)
            # d_loss_source = criterion_da(outputs, source_labels)
            # ########Second DA
            # DA_Discriminator_E.zero_grad()
            # # pdb.set_trace()
            # outputs_E = DA_Discriminator_E(ouput_dict_source['all_feature'])
            # source_labels_E = torch.ones_like(outputs_E).to(device)
            # d_loss_source_E = criterion_da(outputs_E, source_labels_E)
            # # d_loss_source.backward()

            # # train discriminator with target domain
            # outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
            # target_labels = torch.zeros_like(outputs).to(device)
            # d_loss_target = criterion_da(outputs, target_labels)
            # ########Second DA
            # outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
            # target_labels_E = torch.zeros_like(outputs_E).to(device)
            # d_loss_target_E = criterion_da(outputs_E, target_labels_E)
            # # d_loss_target.backward()
            # # update discriminator parameters
            # # d_loss = d_loss_source + d_loss_target
            # d_loss = d_loss_source + d_loss_target + d_loss_source_E + d_loss_target_E
            # d_loss.backward()
            # optimizer_D.step()
            # optimizer_D_E.step()
            # da_loss['D_loss'] = d_loss
            # da_loss['D_loss_s'] = d_loss_source
            # da_loss['D_loss_t'] = d_loss_target 
            # da_loss['D_loss_s_e'] =  d_loss_source_E
            # da_loss['D_loss_t_e'] =  d_loss_target_E
            # da_loss['GRL'] = False
            

            '''
            ###1)
            # train generator
            optimizer.zero_grad()
            target_model.zero_grad()
            # ouput_dict_target_t = target_model(target_batch_data['ego'])
            # ouput_dict_target_s = target_model(source_batch_data['ego'])
            # with torch.no_grad():
            #     ouput_dict_target_s = model(source_batch_data['ego'])            
            ouput_dict_target_t = target_model(target_batch_data['ego'])
            ouput_dict_target_s = target_model(source_batch_data['ego'])
            final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
            outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
            outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
            source_labels = torch.ones_like(outputs).to(device)
            source_labels_E = torch.ones_like(outputs_E).to(device)
            G_loss = criterion_da(outputs, source_labels)
            G_loss_E = criterion_da(outputs_E, source_labels_E)
            da_loss['G_loss'] = G_loss + G_loss_E
            # Total_loss = G_loss*10 + G_loss_E*10 + final_loss
            Total_loss = G_loss + G_loss_E + final_loss
            # Total_loss = G_loss + G_loss_E 
            # Total_loss =final_loss
            da_loss['final_loss'] = final_loss
            # print(final_loss.item())

            Total_loss.backward()
            optimizer.step()
            '''


            ##2）
            if epoch > 2:#### In the initial stages of training, the discriminator can be trained for a few epochs before starting to train the generator.
                with torch.no_grad():
                    if not opt.half:
                        with torch.no_grad():
                            ouput_dict_source = model(source_batch_data['ego'])
                            ouput_dict_target_t = target_model(target_batch_data['ego'])
                            ouput_dict_target_s = target_model(source_batch_data['ego'])
                            final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])

                        # first argument is always your output dictionary,
                        # second argument is always your label dictionary.
                        # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                    else:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                ouput_dict_source = model(source_batch_data['ego'])
                                ouput_dict_target_t = target_model(target_batch_data['ego'])
                                ouput_dict_target_s = target_model(source_batch_data['ego'])
                                final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
                            # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                            

                    # create labels for source and target domains
                    # batch_size = hypes['train_params']['batch_size']
                    # source_labels = torch.ones(batch_size, 1).to(device)
                    # target_labels = torch.zeros(batch_size, 1).to(device)


                    # train discriminator with source domain
                    DA_Discriminator.zero_grad()
                    outputs = DA_Discriminator(ouput_dict_source['fused_feature'])
                    source_labels = torch.ones_like(outputs).to(device)
                    d_loss_source = criterion_da(outputs, source_labels)
                    ########Second DA
                    DA_Discriminator_E.zero_grad()
                    # pdb.set_trace()
                    outputs_E = DA_Discriminator_E(ouput_dict_source['all_feature'])
                    source_labels_E = torch.ones_like(outputs_E).to(device)
                    d_loss_source_E = criterion_da(outputs_E, source_labels_E)
                    # d_loss_source.backward()

                    # train discriminator with target domain
                    outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                    target_labels = torch.zeros_like(outputs).to(device)
                    d_loss_target = criterion_da(outputs, target_labels)
                    ########Second DA
                    outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
                    target_labels_E = torch.zeros_like(outputs_E).to(device)
                    d_loss_target_E = criterion_da(outputs_E, target_labels_E)
                    # d_loss_target.backward()
                    # update discriminator parameters
                    # d_loss = d_loss_source + d_loss_target
                    d_loss = d_loss_source + d_loss_target + d_loss_source_E + d_loss_target_E
                    # d_loss.backward()
                    optimizer_D.step()
                    optimizer_D_E.step()
                    da_loss['D_loss'] = d_loss
                    da_loss['D_loss_s'] = d_loss_source
                    da_loss['D_loss_t'] = d_loss_target 
                    da_loss['D_loss_s_e'] =  d_loss_source_E
                    da_loss['D_loss_t_e'] =  d_loss_target_E
                    da_loss['GRL'] = False
                # da_loss['D_loss'] = torch.tensor(1.0)
                # da_loss['D_loss_s'] = torch.tensor(1.0)
                # da_loss['D_loss_t'] = torch.tensor(1.0)
                # da_loss['D_loss_s_e'] =  torch.tensor(1.0)
                # da_loss['D_loss_t_e'] =  torch.tensor(1.0)
                # da_loss['GRL'] = False
                # train generator
                optimizer.zero_grad()
                target_model.zero_grad()
                # ouput_dict_target_t = target_model(target_batch_data['ego'])
                # ouput_dict_target_s = target_model(source_batch_data['ego'])
                with torch.no_grad():
                    ouput_dict_target_s = model(source_batch_data['ego'])            
                ouput_dict_target_t = target_model(target_batch_data['ego'])
                # ouput_dict_target_s = target_model(source_batch_data['ego'])
                final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
                # with torch.no_grad():
                #     outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                #     outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
                outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
                source_labels = torch.ones_like(outputs).to(device)
                source_labels_E = torch.ones_like(outputs_E).to(device)
                G_loss = criterion_da(outputs, source_labels)
                G_loss_E = criterion_da(outputs_E, source_labels_E)
                da_loss['G_loss'] = G_loss + G_loss_E
                # Total_loss = G_loss*10 + G_loss_E*10 + final_loss
                # Total_loss = G_loss + G_loss_E + final_loss
                Total_loss = G_loss + G_loss_E 
                # Total_loss =final_loss
                da_loss['final_loss'] = final_loss
                # print(final_loss.item())

                Total_loss.backward()
                optimizer.step()

            else:

                if not opt.half:
                    with torch.no_grad():
                        ouput_dict_source = model(source_batch_data['ego'])
                        ouput_dict_target_t = target_model(target_batch_data['ego'])
                        ouput_dict_target_s = target_model(source_batch_data['ego'])
                        final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])

                    # first argument is always your output dictionary,
                    # second argument is always your label dictionary.
                    # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                else:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            ouput_dict_source = model(source_batch_data['ego'])
                            ouput_dict_target_t = target_model(target_batch_data['ego'])
                            ouput_dict_target_s = target_model(source_batch_data['ego'])
                            final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
                        # final_loss = criterion(ouput_dict, source_batch_data['ego']['label_dict'])
                        

                # create labels for source and target domains
                # batch_size = hypes['train_params']['batch_size']
                # source_labels = torch.ones(batch_size, 1).to(device)
                # target_labels = torch.zeros(batch_size, 1).to(device)


                # train discriminator with source domain
                DA_Discriminator.zero_grad()
                outputs = DA_Discriminator(ouput_dict_source['fused_feature'])
                source_labels = torch.ones_like(outputs).to(device)
                d_loss_source = criterion_da(outputs, source_labels)
                ########Second DA
                DA_Discriminator_E.zero_grad()
                # pdb.set_trace()
                outputs_E = DA_Discriminator_E(ouput_dict_source['all_feature'])
                source_labels_E = torch.ones_like(outputs_E).to(device)
                d_loss_source_E = criterion_da(outputs_E, source_labels_E)
                # d_loss_source.backward()

                # train discriminator with target domain
                outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                target_labels = torch.zeros_like(outputs).to(device)
                d_loss_target = criterion_da(outputs, target_labels)
                ########Second DA
                outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
                target_labels_E = torch.zeros_like(outputs_E).to(device)
                d_loss_target_E = criterion_da(outputs_E, target_labels_E)
                # d_loss_target.backward()
                # update discriminator parameters
                # d_loss = d_loss_source + d_loss_target
                d_loss = d_loss_source + d_loss_target + d_loss_source_E + d_loss_target_E
                d_loss.backward()
                optimizer_D.step()
                optimizer_D_E.step()
                da_loss['D_loss'] = d_loss
                da_loss['D_loss_s'] = d_loss_source
                da_loss['D_loss_t'] = d_loss_target 
                da_loss['D_loss_s_e'] =  d_loss_source_E
                da_loss['D_loss_t_e'] =  d_loss_target_E
                da_loss['GRL'] = False
                da_loss['G_loss'] = torch.tensor(1.0)
                da_loss['final_loss'] = torch.tensor(1.0)
                # optimizer.zero_grad()
                # target_model.zero_grad()
                # with torch.no_grad():
                #     # train generator
                #     optimizer.zero_grad()
                #     target_model.zero_grad()
                #     # ouput_dict_target_t = target_model(target_batch_data['ego'])
                #     # ouput_dict_target_s = target_model(source_batch_data['ego'])
                #     # with torch.no_grad():
                #     #     ouput_dict_target_s = model(source_batch_data['ego'])            
                #     ouput_dict_target_t = target_model(target_batch_data['ego'])
                #     ouput_dict_target_s = target_model(source_batch_data['ego'])
                #     final_loss = criterion(ouput_dict_target_s, source_batch_data['ego']['label_dict'])
                #     outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
                #     outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
                #     source_labels = torch.ones_like(outputs).to(device)
                #     source_labels_E = torch.ones_like(outputs_E).to(device)
                #     G_loss = criterion_da(outputs, source_labels)
                #     G_loss_E = criterion_da(outputs_E, source_labels_E)
                #     da_loss['G_loss'] = G_loss + G_loss_E
                #     # Total_loss = G_loss*10 + G_loss_E*10 + final_loss
                #     # Total_loss = G_loss + G_loss_E + final_loss
                #     Total_loss = G_loss + G_loss_E 
                #     # Total_loss =final_loss
                #     da_loss['final_loss'] = final_loss
                    # print(final_loss.item())

                    # Total_loss.backward()
                    # optimizer.step()

            # if epoch <=1:#### In the initial stages of training, the discriminator can be trained for a few epochs before starting to train the generator.
            #     with torch.no_grad():
            #         # train generator
            #         DA_Discriminator.zero_grad()
            #         ########Second DA
            #         DA_Discriminator_E.zero_grad()
            #         optimizer.zero_grad()
            #         outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
            #         outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
            #         source_labels_E = torch.ones(outputs_E.shape[0], 1).to(device)
            #         G_loss = criterion_da(outputs, source_labels)
            #         G_loss_E = criterion_da(outputs_E, source_labels_E)
            #         # G_loss.backward()
            #         Total_loss = G_loss + final_loss + G_loss_E
            #         da_loss['G_loss'] = G_loss + G_loss_E
            #         # back-propagation
            #         # Total_loss.backward() 
            #         optimizer.step()
            # else:
            #         # train generator
            #         DA_Discriminator.zero_grad()
            #         ########Second DA
            #         DA_Discriminator_E.zero_grad()
            #         optimizer.zero_grad()
            #         target_model.zero_grad()
            #         outputs = DA_Discriminator(ouput_dict_target_t['fused_feature'])
            #         outputs_E = DA_Discriminator_E(ouput_dict_target_t['all_feature'])
            #         source_labels_E = torch.ones(outputs_E.shape[0], 1).to(device)
            #         G_loss = criterion_da(outputs, source_labels)
            #         G_loss_E = criterion_da(outputs_E, source_labels_E)
            #         # G_loss.backward()
            #         # Total_loss = G_loss + final_loss + G_loss_E
            #         # da_loss['G_loss'] = G_loss + G_loss_E
            #         Total_loss = G_loss + final_loss + G_loss_E
            #         da_loss['G_loss'] = G_loss + G_loss_E
            #         # back-propagation
            #         if not opt.half:
            #             Total_loss.backward() 
            #             optimizer.step()
            #         else:
            #             scaler.scale(Total_loss).backward()
            #             scaler.step(optimizer)
            #             scaler.update()

            pbar2.update(1)   
            criterion.logging_da(epoch, iteration, len(source_train_loader), writer, da_loss, pbar=pbar2)
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + iteration)
                scheduler_D.step_update(epoch * num_steps + iteration)


        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    target_model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = target_model(batch_data['ego'])
                
                    final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                            valid_ave_loss))
            
            txt_log.write('At epoch' + str(epoch+1)+',  the validation loss is'+ str(valid_ave_loss) + 'save in '+ str(os.path.join(saved_path,'net_epoch%d.pth' % (epoch + 1))) + '\n')

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(target_model.state_dict(),
                    os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            # torch.save(DA_Discriminator.state_dict(),
            #         os.path.join(saved_path,
            #                         'Dis_net_epoch%d.pth' % (epoch + 1)))

    print('Training Finished, checkpoints saved to %s' % saved_path)
    # close file
    txt_log.close()

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    #########################################################################################
    # DA_Component: dataloader for source and target domain
    #########################################################################################

    source_opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, isSim=True)###### for source data
    target_opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, isSim=False)######for target data
    if source_opencood_train_dataset is not None:
        
        source_train_loader = DataLoader(source_opencood_train_dataset,
                                        batch_size=hypes['train_params']['batch_size'], 
                                        num_workers=8, 
                                        collate_fn=source_opencood_train_dataset.collate_batch_train, 
                                        shuffle=True, 
                                        pin_memory=False, 
                                        drop_last=True)
        print("source_opencood_train_dataset is loaded!")
    if target_opencood_train_dataset is not None:
        
        target_train_loader = DataLoader(target_opencood_train_dataset,
                                        batch_size=hypes['train_params']['batch_size'],
                                        num_workers=8,
                                        collate_fn=source_opencood_train_dataset.collate_batch_train, 
                                        shuffle=True, 
                                        pin_memory=False, 
                                        drop_last=True)
        print("target_opencood_train_dataset is loaded!")

    #########################################################################################
    # DA_Component: dataloader for source and target domain
    #########################################################################################
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False,
                                              isSim=False)
    if opencood_validate_dataset is not None:
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=source_opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)
        print("opencood_validate_dataset is loaded!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #########################################################################################
    # DA_Component: training for source and target domain
    #########################################################################################
    DA_training = hypes['train_params']['DA_training']
    if 'GRL' == DA_training:

        GRL_training(opt,hypes,device,source_train_loader,target_train_loader,val_loader)
    elif 'ADDA' == DA_training:

        ADDA_training(opt,hypes,device,source_train_loader,target_train_loader,val_loader)
    elif 'Dual_DA' == DA_training:

        Dual_DA_training(opt,hypes,device,source_train_loader,target_train_loader,val_loader)
        
    else:
        print('Dont detect the training strategy!!')

        




if __name__ == '__main__':
    main()
