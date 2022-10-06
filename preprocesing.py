import pickle
import pandas as pd

from utilites.preprocess import unzip_folder, check_dataset, new_order, upsampling, data_aumentation, new_dataset_csv, new_dataset, save_mfcc

if __name__ == "__main__":

    dataset_csv = 'dataset/Train.csv'
    dataset_file = 'dataset/Train'
    dataset_file_test = 'dataset/Test'
    paths = ['./dataset/#Train/F','./dataset/#Train/M']

    print(f'\nThis process may take a few minutes\nunzip folder')
    unzip_folder('./dataset/clasificacion-de-acentos-latinos.zip','./dataset/',pwd='')
    print(f'done\n')
    
    df = pd.read_csv(f'./{dataset_csv}')
    sumM,sumF,M,F = check_dataset(df)

    print(f'Original Dataset\nM\tF\tkey')
    for key,m,f in zip(M.keys(),M.values(),F.values()):
        print(f'{len(m)}\t{len(f)}\t{key}')
    print(f'M_total : {sumM}\tF_total : {sumF}\n')
    
    print(f'sorting the dataset files')
    new_order(M,F,f'./{dataset_file}/','./dataset/#Train/')
    print(f'sorting done\n')

    print(f'upsampling and data aumentation dataset')
    for path in paths:
        upsampling(path)
        data_aumentation(path)
    print(f'done\n')

    print(f'crate Train.csv')
    df_ = new_dataset_csv(paths)
    df_.to_csv(f'./{dataset_csv}',index=False)
    new_dataset(paths,f'./{dataset_file}/','./dataset/#Train/')
    print(f'done\n')

    print(f'New Dataset\n')
    print(f'{df_.info()}')

    print(f'\nExtract features audio mfcc for train dataset')
    dic_mfcc_train = save_mfcc(dataset_file, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5)
    print(f'done\n')

    with open('./dataset/mfcc_train.json', 'wb') as fp:
        pickle.dump(dic_mfcc_train, fp)
    
    print(f'\nExtract features audio mfcc for test dataset')
    dic_mfcc_test = save_mfcc(dataset_file_test, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5)
    print(f'done\n')

    with open('./dataset/mfcc_test.json', 'wb') as fp:
        pickle.dump(dic_mfcc_test, fp)

    print(f'Everything is ready to train models')