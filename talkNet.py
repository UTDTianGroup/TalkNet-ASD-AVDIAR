import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel
from torchaudio.prototype.pipelines import VGGISH
import torchvision

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_block=False, stride=(1,1), padding=(1,1)):
        super(ConvBlock, self).__init__()
        self.last_block = last_block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if not self.last_block:
            self.relu3 = nn.ReLU()
            self.bn3 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.conv3(x)
        if self.last_block:
            return x
        else:
            x = self.bn3(self.relu3(x))
            return x

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, detector_arch=0, num_blocks_unfrozen=0, **kwargs):
        super(talkNet, self).__init__()

        if torch.cuda.is_available(): #Set device to either gpu or cpu based on the runtime environment. Prefer using CUDA/GPU when available.
            self.device='cuda'
        else:
            self.device='cpu'        
        
        self.model = talkNetModel().to(self.device)
        self.visual_block_layer_dict = {0:31, 1:24, 2:17, 3:10, 4:5, 5:0}
        self.audio_block_layer_dict = {0:16, 1:11, 2:6, 3:3, 4:0}

        self.num_blocks_unfrozen = num_blocks_unfrozen

        vggish_model = VGGISH.get_model()
        self.audio_feature_extractor = vggish_model.features_network
        self.audio_feature_extractor = self.audio_feature_extractor.to(self.device)
        vgg_model = torchvision.models.vgg16(pretrained=True)
        self.visual_feature_extractor = vgg_model.features
        self.visual_feature_extractor = self.visual_feature_extractor.to(self.device)

        for name, param in self.visual_feature_extractor.named_parameters():
            name_ind=int(name.split('.')[0])
            if name_ind < self.visual_block_layer_dict[self.num_blocks_unfrozen]:
                print('Freezing visual weights of layer: ', name_ind)
                param.requires_grad=False

        for name, param in self.audio_feature_extractor.named_parameters():
            name_ind=int(name.split('.')[0])
            if name_ind < self.audio_block_layer_dict[self.num_blocks_unfrozen]:
                print('Freezing audio weights of layer: ', name_ind)
                param.requires_grad=False

        self.lossAV = lossAV().to(self.device)
        self.lossA = lossA().to(self.device)
        self.lossV = lossV().to(self.device)
        if detector_arch==1:
            self.lossAV.FC = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128), nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 2))
            self.lossA.FC = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 2))
            self.lossV.FC = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 2))
        

        self.visual_avg_pool = nn.AvgPool2d((3,3))
        self.audio_avg_pool = nn.AvgPool2d((6,4))
        self.visual_avg_pool = self.visual_avg_pool.to(self.device)
        self.audio_avg_pool = self.audio_avg_pool.to(self.device)

        self.visual_flatten = nn.Flatten()
        self.audio_flatten = nn.Flatten()
        self.visual_flatten = self.visual_flatten.to(self.device)
        self.audio_flatten = self.audio_flatten.to(self.device)

        self.visual_projector = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256,128))
        self.audio_projector = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256,128))
        self.visual_projector = self.visual_projector.to(self.device)
        self.audio_projector = self.audio_projector.to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)
        self.transformer_encoder = self.transformer_encoder.to(self.device)

        #self.audio_conv = [nn.Sequential(ConvBlock(512, 512, 3),nn.MaxPool2d(2), ConvBlock(512,512,3)) for i in range(25)]
        self.audio_conv = ConvBlock(512,512,3, last_block=True)
        self.visual_conv = ConvBlock(512, 512, 3, last_block=True)
        self.audio_conv = self.audio_conv.to(self.device)
        self.visual_conv = self.visual_conv.to(self.device)

        self.audio_fc_lst = [nn.Sequential(nn.Linear(512*24,1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Linear(1024,128)) for i in range(25)]
        self.audio_fc_lst = [audio_fc.to(self.device) for audio_fc in self.audio_fc_lst]
        self.visual_fc = nn.Sequential(nn.Linear(512*9,512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512,128))
        self.visual_fc = self.visual_fc.to(self.device)
        
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            # print('audioFeature shape: ', audioFeature[0].shape)
            # print('visualFeature shape: ', visualFeature[0].shape)
            # print('labels shape: ', labels[0].reshape((-1)).shape)
            audioFeature = audioFeature[0]
            visualFeature = visualFeature[0]
            B, T_a, d2_a, d3_a, d4_a = audioFeature.shape
            _, T_v, c_v, h_v, w_w = visualFeature.shape
            audioFeature_reshaped = torch.reshape(audioFeature, (B*T_a, d2_a, d3_a, d4_a))
            visualFeature_reshaped = torch.reshape(visualFeature, (B*T_v, c_v, h_v, w_w))

            audioFeature_reshaped = audioFeature_reshaped.to(self.device)
            visualFeature_reshaped = visualFeature_reshaped.to(self.device)

            # print('audio feature reshaped shape: ', audioFeature_reshaped.shape)
            # print('visual feature reshaped shape: ', visualFeature_reshaped.shape)
            # audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(self.device)) # feedForward
            # visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(self.device))
            audioEmbed_reshaped = self.audio_feature_extractor(audioFeature_reshaped)
            visualEmbed_reshaped = self.visual_feature_extractor(visualFeature_reshaped)
            # print('audio embed reshaped shape: ', audioEmbed_reshaped.shape)
            # print('visual embed reshaped shape: ', visualEmbed_reshaped.shape)

            # audioEmbed_reshaped = self.audio_conv(audioEmbed_reshaped)
            # audioEmbed_lst = [self.audio_conv[i](audioEmbed_reshaped) for i in range(25)]
            # audioEmbed_reshaped = torch.cat(audioEmbed_lst, dim=-3)
            # visualEmbed_reshaped = self.visual_conv(visualEmbed_reshaped)

            # audioEmbed_reshaped = self.audio_avg_pool(audioEmbed_reshaped)
            # visualEmbed_reshaped = self.visual_avg_pool(visualEmbed_reshaped)
            # print('audio embed reshaped shape after avg pool : ', audioEmbed_reshaped.shape)
            # print('visual embed reshaped shape after avg pool: ', visualEmbed_reshaped.shape)

            audioEmbed_reshaped = self.audio_flatten(audioEmbed_reshaped)
            # print(audioEmbed_reshaped.shape, audioEmbed_reshaped_flattened.shape)
            # audioEmbed_reshaped = torch.transpose(torch.transpose(audioEmbed_reshaped, 1,3), 1, 2)
            # print(audioEmbed_reshaped.shape)
            # audioEmbed_reshaped = torch.reshape(audioEmbed_reshaped, (B*T_a, 24, 512))
            # print(audioEmbed_reshaped.shape)
            # audioEmbed_reshaped = torch.cat([audioEmbed_reshaped, audioEmbed_reshaped_flattened.unsqueeze(1)], dim=1)
            # print(audioEmbed_reshaped.shape)
            visualEmbed_reshaped = self.visual_flatten(visualEmbed_reshaped)
            # print('audio embed reshaped shape after flatten : ', audioEmbed_reshaped.shape)
            # print('visual embed reshaped shape after flatten: ', visualEmbed_reshaped.shape)

            # audioEmbed_reshaped = self.audio_projector(audioEmbed_reshaped)
            # visualEmbed_reshaped = self.visual_projector(visualEmbed_reshaped)
            # print('audio embed reshaped shape after projection : ', audioEmbed_reshaped.shape)
            # print('visual embed reshaped shape after projection: ', visualEmbed_reshaped.shape)

            # audioEmbed_reshaped = torch.repeat_interleave(audioEmbed_reshaped, 25, dim=0)
            # print('audio embed reshaped shape after repeat : ', audioEmbed_reshaped.shape)
            audioEmbed_lst = [self.audio_fc_lst[i](audioEmbed_reshaped) for i in range(25)]
            audioEmbed_reshaped = torch.stack(audioEmbed_lst, dim=1)
            audioEmbed_reshaped = torch.reshape(audioEmbed_reshaped, (-1, 128))
            audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_v, 128))
            #audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_a, 512))
            #audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_a, 25, 512))
            #audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_v, 512))
            visualEmbed_reshaped = self.visual_fc(visualEmbed_reshaped)
            visualEmbed = torch.reshape(visualEmbed_reshaped, (B, T_v, 128))
            # visualEmbed = torch.reshape(visualEmbed, (B, T_a, 25, 512))
            # visualEmbed = torch.mean(visualEmbed, dim=2)


            # print('audio embed shape: ', audioEmbed.shape)
            # print('visual embed shape: ', visualEmbed.shape)

            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)

            # avembed = torch.cat([audioEmbed, visualEmbed], dim=2)
            # transformer_out = self.transformer_encoder(avembed)
            # transformer_out = torch.reshape(transformer_out, shape=(-1, 256))
            # print('transformer out shape: ', transformer_out.shape)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).to(self.device) # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioFeature = audioFeature[0]
                visualFeature = visualFeature[0]
                B, T_a, d2_a, d3_a, d4_a = audioFeature.shape
                _, T_v, c_v, h_v, w_w = visualFeature.shape
                audioFeature_reshaped = torch.reshape(audioFeature, (B*T_a, d2_a, d3_a, d4_a))
                visualFeature_reshaped = torch.reshape(visualFeature, (B*T_v, c_v, h_v, w_w))

                audioFeature_reshaped = audioFeature_reshaped.to(self.device)
                visualFeature_reshaped = visualFeature_reshaped.to(self.device)
                # audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].to(self.device))
                # visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(self.device))
                audioEmbed_reshaped = self.audio_feature_extractor(audioFeature_reshaped)
                visualEmbed_reshaped = self.visual_feature_extractor(visualFeature_reshaped)
                # audioEmbed_reshaped = self.audio_conv(audioEmbed_reshaped)
                # visualEmbed_reshaped = self.visual_conv(visualEmbed_reshaped)
                #audioEmbed_lst = [self.audio_conv[i](audioEmbed_reshaped) for i in range(25)]
                #audioEmbed_reshaped = torch.cat(audioEmbed_lst, dim=-3)
                # audioEmbed_reshaped = self.audio_avg_pool(audioEmbed_reshaped)
                # visualEmbed_reshaped = self.visual_avg_pool(visualEmbed_reshaped)
                audioEmbed_reshaped = self.audio_flatten(audioEmbed_reshaped)
                visualEmbed_reshaped = self.visual_flatten(visualEmbed_reshaped)
                audioEmbed_lst = [self.audio_fc_lst[i](audioEmbed_reshaped) for i in range(25)]
                audioEmbed_reshaped = torch.stack(audioEmbed_lst, dim=1)
                audioEmbed_reshaped = torch.reshape(audioEmbed_reshaped, (-1, 128))
                audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_v, 128))
                # audioEmbed_reshaped = torch.transpose(torch.transpose(audioEmbed_reshaped, 1,3), 1, 2)
                # audioEmbed_reshaped = torch.reshape(audioEmbed_reshaped, (B*T_a, 24, 512))
                # audioEmbed_reshaped = torch.cat([audioEmbed_reshaped, audioEmbed_reshaped_flattened.unsqueeze(1)], dim=1)
                #audioEmbed_reshaped = self.audio_projector(audioEmbed_reshaped)
                #visualEmbed_reshaped = self.visual_projector(visualEmbed_reshaped)
                # audioEmbed_reshaped = torch.repeat_interleave(audioEmbed_reshaped, 25, dim=0)
                # audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_v, 512))
                # visualEmbed = torch.reshape(visualEmbed_reshaped, (B, T_v, 512))
                #audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_v, 512))
                # audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_a, 25, 512))
                # audioEmbed = torch.reshape(audioEmbed_reshaped, (B, T_v, 512))
                visualEmbed_reshaped = self.visual_fc(visualEmbed_reshaped)
                visualEmbed = torch.reshape(visualEmbed_reshaped, (B, T_v, 128))
                # visualEmbed = torch.reshape(visualEmbed, (B, T_a, 25, 512))
                # visualEmbed = torch.mean(visualEmbed, dim=2)
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                # avembed = torch.cat([audioEmbed, visualEmbed], dim=2)
                # transformer_out = self.transformer_encoder(avembed)
                # transformer_out = torch.reshape(transformer_out, shape=(-1, 256))
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).to(self.device)             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        print('evaluation command: ', cmd)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path, map_location):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=map_location)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
