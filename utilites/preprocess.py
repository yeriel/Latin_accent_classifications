import os
import math
import shutil
import zipfile
import librosa
import pandas as pd
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, HighPassFilter

def unzip_folder(zip_folder, destination, pwd):
        with zipfile.ZipFile(zip_folder) as zf:
            zf.extractall(
                destination, pwd=pwd.encode())

def check_dataset(df):
    M = {'Argentina':[], 'Chile':[], 'Colombia':[], 'Peru':[], 'Venezuela':[]}
    F = {'Argentina':[],'Chile':[], 'Colombia':[], 'Peru':[], 'Venezuela':[]}
    sumM = 0
    sumF = 0

    for id,i in zip(df.Id, df.Expected):
    
        if i[0] == '0':
            sumF +=1
            if i[2] == '2':
                F['Argentina'].append(id)
            elif i[2] == '3':
                F['Chile'].append(id)
            elif i[2] == '4':
                F['Colombia'].append(id)
            elif i[2] == '5':
                F['Peru'].append(id)
            elif i[2] == '6':
                F['Venezuela'].append(id)
            else: 
                continue
    
        else:
            sumM +=1 
            if i[2] == '2':
                M['Argentina'].append(id)
            elif i[2] == '3':
                M['Chile'].append(id)
            elif i[2] == '4':
                M['Colombia'].append(id)
            elif i[2] == '5':
                M['Peru'].append(id)
            elif i[2] == '6':
                M['Venezuela'].append(id)
            else:
                continue
    
    return sumM, sumF, M, F

def mkdirs(newdir,mode=777):
    try:
        os.makedirs(newdir, mode)
    except OSError as err:
        return err

def new_order(M,F,src,dir):

    countries = M.keys()

    for file in os.listdir(src):
        for country in countries:
            if file in M[country]:
                dst = f'{dir}M/{country}'
                file_path = f'{src}{file}'
            
                if not os.path.exists(dst):
                    mkdirs(dst)
            
                if os.path.exists(dst):
                    shutil.move(file_path, dst)
        
            elif file in F[country]:
                dst = f'{dir}F/{country}'
                file_path = f'{src}{file}'
            
                if not os.path.exists(dst):
                    mkdirs(dst)
            
                if os.path.exists(dst):
                    shutil.move(file_path, dst)
            else:
                continue

def aumentation(path_file):
    augment_raw_audio = Compose([AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=1),
                                 HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1)])
    
    path_dst_audio = f'{path_file[:-4]}_{path_file[-4:]}'
    signal, sr = librosa.load(path_file)
    augmented_signal = augment_raw_audio(signal, sr)
    sf.write(path_dst_audio, augmented_signal, sr)

def upsampling(path):
    for country in os.listdir(path):
        if len(os.listdir(f'{path}/{country}')) < 144:
            diff =  144 - len(os.listdir(f'{path}/{country}'))
            for index, path_file in enumerate(os.listdir(f'{path}/{country}')):
                if index < diff:
                    aumentation(f'{path}/{country}/{path_file}')

def data_aumentation(path):
    for country in os.listdir(path):
        for path_file in os.listdir(f'{path}/{country}'):
            aumentation(f'{path}/{country}/{path_file}')

def new_dataset_csv(paths):
    columns_ = ['Id','Sex', 'Country']
    df = pd.DataFrame(columns = columns_)
    
    for path in paths:
        for country in os.listdir(path):
            for file in os.listdir(f'{path}/{country}'):
                if path[-1] == 'M':
                    if country == 'Argentina':
                        df_ = pd.DataFrame([[file,1,2]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Chile':
                        df_ = pd.DataFrame([[file,1,3]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Colombia':
                        df_ = pd.DataFrame([[file,1,4]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Peru':
                        df_ = pd.DataFrame([[file,1,5]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Venezula':
                        df_ = pd.DataFrame([[file,1,6]], columns=columns_)
                        df = df.append(df_)
                        
                elif path[-1] == 'F':
                    if country == 'Argentina':
                        df_ = pd.DataFrame([[file,0,2]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Chile':
                        df_ = pd.DataFrame([[file,0,3]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Colombia':
                        df_ = pd.DataFrame([[file,0,4]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Peru':
                        df_ = pd.DataFrame([[file,0,5]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Venezula':
                        df_ = pd.DataFrame([[file,0,6]], columns=columns_)
                        df = df.append(df_)
                else:
                    continue
    return df

def new_dataset(paths, dst, temp='../dataset/#Train/'):
    
    shutil.rmtree(dst)
    mkdirs(dst,mode=777)

    for path in paths:
        for country in os.listdir(path):
            for file in os.listdir(f'{path}/{country}'):
                shutil.move(f'{path}/{country}/{file}',dst)
    
    shutil.rmtree(temp)

def save_mfcc(dirpath, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    SAMPLE_RATE = 22050
    TRACK_DURATION = 7 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    mfcc_ = {}
    
    for f in os.listdir(dirpath):
        file_path = os.path.join(dirpath, f)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        for d in range(num_segments):
            start = samples_per_segment * d
            finish = start + samples_per_segment

            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_.update({str(f): mfcc.tolist()})
    return mfcc_

if __name__ == "__main__":
    
    print(f'\nthis process takes approximately 10 minutes\nunzip folder\n')
    unzip_folder('../dataset/clasificacion-de-acentos-latinos.zip','../dataset/',pwd='')
    print(f'done\n')
    
    df_train = pd.read_csv('../dataset/Train.csv')
    sumM,sumF,M,F = check_dataset(df_train)

    print(f'Original Dataset\nM\tF\tkey')
    for key,m,f in zip(M.keys(),M.values(),F.values()):
        print(f'{len(m)}\t{len(f)}\t{key}')
    print(f'M_total : {sumM}\tF_total : {sumF}\n')
    
    print(f'sorting the dataset files')
    new_order(M,F,'../dataset/Train/','../dataset/#Train/')
    print(f'sorting done\n')

    print(f'upsampling and data aumentation dataset')
    paths = ['../dataset/#Train/F','../dataset/#Train/M']
    for path in paths:
        upsampling(path)
        data_aumentation(path)
    print(f'done\n')

    paths = ['../dataset/#Train/F','../dataset/#Train/M']
    print(f'crate Train.csv')
    df = new_dataset_csv(paths)
    df.to_csv('../dataset/Train.csv',index=False)
    new_dataset(paths,'../dataset/Train/')
    print(f'done\n')
    print(f'New Dataset\n')
    print(f'{df.info()}')