import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
from PIL import Image
import torchaudio, torchvision
from torchaudio.prototype.pipelines import VGGISH


def generate_audio_set(dataPath, batchList, use_avdiar=False):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        if use_avdiar:
            videoName = data[0][:13]
        else:
            videoName = data[0][:11]
        dataName = data[0]
        # _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audio, samplingRate = torchaudio.load(os.path.join(dataPath, videoName, dataName + '.wav'))
        audio = audio.squeeze(0)
        if(len(audio.shape) > 1):
            audio = torch.mean(audio, dim=0)
        audio = torchaudio.functional.resample(audio, samplingRate, VGGISH.sample_rate)
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = random.uniform(-5, 5)
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = torch.nn.functional.pad(noiseAudio, (0, shortage), mode='circular')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * torch.log10(torch.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * torch.log10(torch.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = torch.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None, audioProcessor=None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    # audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    audio = audioProcessor(audio)
    # maxAudio = int(numFrames * 4)
    maxAudio = int(numFrames/fps)
    # print(f'audio shapes: ', processed_audio.shape, numFrames)
    # if audio.shape[0] < maxAudio:
    #     shortage    = maxAudio - audio.shape[0]
    #     audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:maxAudio,:,:,:]  
    return audio

def load_visual(data, dataPath, numFrames, image_transform, use_avdiar=False): 
    dataName = data[0]
    if use_avdiar:
        videoName = data[0][:13]
    else:
        videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False)
    faces = []
    # H = 112
    # if visualAug == True:
    #     new = int(H*random.uniform(0.7, 1))
    #     x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
    #     M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
    #     augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    # else:
    #     augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        # face = cv2.imread(faceFile)
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # face = cv2.resize(face, (H,H))
        # if augType == 'orig':
        #     faces.append(face)
        # elif augType == 'flip':
        #     faces.append(cv2.flip(face, 1))
        # elif augType == 'crop':
        #     faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        # elif augType == 'rotate':
        #     faces.append(cv2.warpAffine(face, M, (H,H)))
        face = Image.open(faceFile)
        face = image_transform(face)
        faces.append(face)
    faces = torch.stack(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, use_avdiar, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end 
        self.use_avdiar=use_avdiar 
        self.image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(112,112)), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(degrees=30), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.audioProcessor = VGGISH.get_input_processor()    
        self.vgg_sample_rate = VGGISH.sample_rate   

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList, self.use_avdiar) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet, audioProcessor=self.audioProcessor))  
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, image_transform=self.image_transform, use_avdiar=self.use_avdiar))
            labels.append(load_label(data, numFrames))
        return torch.stack(audioFeatures), \
               torch.stack(visualFeatures), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, use_avdiar, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.use_avdiar=use_avdiar

        self.audioProcessor = VGGISH.get_input_processor()    
        self.vgg_sample_rate = VGGISH.sample_rate
        self.image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(112,112)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line, self.use_avdiar)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet, audioProcessor=self.audioProcessor)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, image_transform=self.image_transform, use_avdiar=self.use_avdiar)]
        labels = [load_label(data, numFrames)]         
        return torch.stack(audioFeatures, dim=0), \
               torch.stack(visualFeatures), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
