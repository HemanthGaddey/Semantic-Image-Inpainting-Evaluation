import time
import os
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
#from model import create_model
from model.tdanet_model import TDAnet
from custom_options import Options

def print_current_errors(log_name, epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

if __name__ == '__main__':
    # get training options
    opt = opt=Options(name='tda_bird',model="tdanet",mask_type=[0,1,2,3],img_file='./datasets/CUB_200_2011/train.flist',mask_file='./datasets/CUB_200_2011/train_mask.flist',text_config='config.bird.yml') #TRAINING OPTIONS, CHANGE FOR TESTING!!!
#TrainOptions().parse()
    #initialize the log file
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    # create a dataset
    dataset = dataloader(opt) 
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)
    # create a model
    model = TDAnet(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter+opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count

    # training process
    while(keep_training):
        epoch_start_time = time.time()
        epoch+=1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            dataset.epoch = epoch - 1
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(input=data)
            model.optimize_parameters()

            #Start Logging
            with open(log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                print_current_errors(log_name, epoch, total_iteration, losses, t)

            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')

            # save the model every <save_iter_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0:
                print('saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break

        model.update_learning_rate()

        print('\nEnd training')
