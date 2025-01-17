import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from talkNet import talkNet
from torch import nn

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="/data08/AVA", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    parser.add_argument('--use_avdiar',      action='store_true', help='Train/test with avdiar data.')
    parser.add_argument('--finetune',      action='store_true', help="Finetune with TalkNet's pretrained weights.")
    parser.add_argument('--detector_arch', type=int, default=0, help='Choose the detector architecture. 0: default architecture. 1: Modified architecture.')
    parser.add_argument('--finetuned_model_path', type=str, default='Path not specified', help='Path to the saved model after finetuning')
    parser.add_argument('--num_blocks_unfrozen', type=int, default=0, help='The number of convolution blocks unfrozen in the feature extractors for finetuning. Max value is 4 since the number of Conv Blocks in VGGish is 4.')

    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()

    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)

    if args.evaluation == True:
        if args.finetune:
            s = talkNet(**vars(args))
            s.loadParameters(args.finetuned_model_path, map_location=torch.device(s.device))
            s = s.to(s.device)
            print("Model %s loaded from previous state!"%(args.finetuned_model_path))
        else:
            download_pretrain_model_AVA()
            s = talkNet(**vars(args))
            s.loadParameters('pretrain_AVA.model', map_location=torch.device(s.device))
            s = s.to(s.device)
            print("Model %s loaded from previous state!"%('pretrain_AVA.model'))
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()
    
    if args.finetune:
        download_pretrain_model_AVA()
        s = talkNet(**vars(args))
        s.loadParameters('pretrain_AVA.model', map_location=torch.device(s.device))
        for name, param in s.named_parameters():
            if 'FC' not in name:
                print('parameter name: ', name)
                param.requires_grad = False
        if args.detector_arch == 0:
            epoch = 14 # estimated epoch of pretrain_AVA.model
        else:
            epoch = 1
        s = s.to(s.device)
    else:
        modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
        modelfiles.sort()  
        if len(modelfiles) >= 1:
            print("Model %s loaded from previous state!"%modelfiles[-1])
            epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
            s = talkNet(epoch = epoch, **vars(args))
            s.loadParameters(modelfiles[-1], map_location=torch.device(s.device))
        else:
            epoch = 1
            s = talkNet(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    for name, param in s.named_parameters():
        print(f'parameter: {name}; requires_grad: {param.requires_grad}')
    
    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
