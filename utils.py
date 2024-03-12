from lxml.html.clean import Cleaner
from torch.utils.data import Dataset
import torch



def remove_comments(html_content):
    cleaner = Cleaner(comments=True)
    cleaned_html = cleaner.clean_html(html_content)
    return cleaned_html


class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return sequence, label


def get_split():
    file_paths = [f"cleaneval/ce_split/{part}.txt" for part in ["train_set", "dev_set", "test_set"]]
    res = []
    for file in file_paths:
        with open(file) as f:
            lines = f.readlines()

        curr = [line.split(".")[0].split("_")[1] for line in lines]
        res.append(curr)

    return res
