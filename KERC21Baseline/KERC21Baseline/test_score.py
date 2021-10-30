import torch
from dataset import DatasetKERC21
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import os
import random
from utils.utils import read_config_file
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval(train_config, trained_model_path, device):
    sample_ids = []
    pred_labels = []
    test_sets = 'test'
    for test_set in test_sets:
        model = torch.load(trained_model_path)
        model.eval()
        val_dataset = DatasetKERC21(dataset_dir=train_config['dataset_dir'], data_type=test_set)
        val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

        with torch.no_grad():
            for eeg, eda, bvp, personality, _, sample_id in val_dataloader:
                eeg = eeg.to(device)
                eda = eda.to(device)
                bvp = bvp.to(device)
                personality = personality.to(device)
                outputs = model(eeg, eda, bvp, personality)
                sample_ids.extend(sample_id)
                preds = np.argmax(outputs.detach().cpu(), axis=1)
                pred_labels.extend(preds)
    submission_df = pd.DataFrame()
    quadrants = ['HAHV', 'HALV', 'LALV', 'LAHV']  # four quadrants in arousal, valence space
    submission_df['Id'] = sample_ids
    submission_df['Predicted'] = [quadrants[x.item()] for x in pred_labels]
    submission_df = submission_df[:90]
    test_label = pd.read_csv(f"D:\googleDrive\Korean-Emotion-Recognition\KERC21Dataset\KERC21Dataset\preprocessed/test_labels.csv")

    # test_label과 submission의 Id가 게 정리
    submission = []
    for i in range(len(test_label)):
        for j in range(len(submission_df)):
            if test_label['Id'][i] == submission_df['Id'][j]:
                submission.append(submission_df['Predicted'][j])

    train_f1 = f1_score(test_label['label'], submission, average='micro')

    print('F1-Score : ' + str(train_f1))



if __name__ == '__main__':
    seed_everything(17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs_dir = 'configs'
    logs_dir = 'logs'
    config = read_config_file(f'{configs_dir}/config.ini')
    train_configs = config['train']
    eval(train_configs, 'logs/saved_model/saved_model.pt', device)  # val or test