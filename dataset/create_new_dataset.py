#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import zipfile

import numpy as np
import pandas as pd

import librosa
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

def new_order(M,F,src='Train/',dir='#Train/'):

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
    columns_ = ['id','F', 'M', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela']
    df = pd.DataFrame(columns = columns_)
    
    for path in paths:
        for country in os.listdir(path):
            for file in os.listdir(f'{path}/{country}'):
                if path[-1] == 'F':
                    if country == 'Argentina':
                        df_ = pd.DataFrame([[file,1,0,1,0,0,0,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Chile':
                        df_ = pd.DataFrame([[file,1,0,0,1,0,0,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Colombia':
                        df_ = pd.DataFrame([[file,1,0,0,0,1,0,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Peru':
                        df_ = pd.DataFrame([[file,1,0,0,0,0,1,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Venezula':
                        df_ = pd.DataFrame([[file,1,0,0,0,0,0,1]], columns=columns_)
                        df = df.append(df_)
                        
                elif path[-1] == 'M':
                    if country == 'Argentina':
                        df_ = pd.DataFrame([[file,0,1,1,0,0,0,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Chile':
                        df_ = pd.DataFrame([[file,0,1,0,1,0,0,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Colombia':
                        df_ = pd.DataFrame([[file,0,1,0,0,1,0,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Peru':
                        df_ = pd.DataFrame([[file,0,1,0,0,0,1,0]], columns=columns_)
                        df = df.append(df_)
                    elif country == 'Venezula':
                        df_ = pd.DataFrame([[file,0,1,0,0,0,0,1]], columns=columns_)
                        df = df.append(df_)
                else:
                    continue
    return df

def new_dataset(paths, dst='Train/'):
    
    shutil.rmtree(dst)
    mkdirs(dst,mode=777)

    for path in paths:
        for country in os.listdir(path):
            for file in os.listdir(f'{path}/{country}'):
                shutil.move(f'{path}/{country}/{file}',dst)
    
    shutil.rmtree('#Train/')

if __name__ == "__main__":

    print(f'this process takes approximately 10 minutes\nunzip folder')
    unzip_folder('clasificacion-de-acentos-latinos.zip','.',pwd='')
    print(f'done')
    
    df_train = pd.read_csv('Train.csv')
    sumM,sumF,M,F = check_dataset(df_train)

    print(f'Original Dataset\nM\tF\tkey')
    for key,m,f in zip(M.keys(),M.values(),F.values()):
        print(f'{len(m)}\t{len(f)}\t{key}')
    print(f'M_total : {sumM}\tF_total : {sumF}')
    
    print(f'sorting the dataset files')
    new_order(M,F)
    print(f'sorting done')

    print(f'upsampling and data aumentation dataset')
    paths = ['#Train/F','#Train/M']
    for path in paths:
        upsampling(path)
        data_aumentation(path)
    print(f'done')
    
    print(f'crate Train.csv')
    df = new_dataset_csv(paths)
    df.to_csv('Train.csv',index=False)
    new_dataset(paths)
    print(f'done')
    print(f'New Dataset\n{df.info()}')

