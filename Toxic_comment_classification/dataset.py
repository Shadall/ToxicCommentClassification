import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class ToxicCommentsDataset(Dataset):
    """Custom Dataset

    :returns: dictionary
    :rtype: Dict
    """

    def __init__(
            self,
            data: pd.DataFrame,
            target_columns: list,
            tokenizer: BertTokenizerFast,
            max_token_len: int = 128
    ):
        self.data = data
        self.target_columns = target_columns
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        _id = data_row['id']
        comment_text = data_row.comment_text
        labels = data_row[self.target_columns]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,  # [CLS] & [SEP]
            return_token_type_ids=False,
            return_attention_mask=True,  # attention_mask
            return_tensors='pt',
        )

        return dict(
            _id=_id,
            comment_text=comment_text,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.FloatTensor(labels)
        )
