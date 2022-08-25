import os
import pandas as pd
import torch
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import spacy
from config import get_config
from logger import get_logger, close_logger
from train import train_bert, train_logreg
from utils import tf_idf, get_dataloaders, set_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.path.dirname(os.path.realpath(__file__))
path_to_config = os.path.join(path, 'config.yaml')
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words


def main():
    """Main function to train baseline model and Bert

    :returns: exit code
    :rtype: int
    """
    config = get_config(path_to_config)
    logger = get_logger(config['data']['log_file_path'])

    model_type = config['model_type']

    set_seed(config['seed'])

    data = pd.read_csv(config['data']['train_data_path'])
    if config['test_run']:
        data = data[:1000]

    if model_type == 'baseline':
        logger.info('Baseline model')
        text, targets = data['comment_text'], data.iloc[:, 2:8]

        train_text, val_text, train_targets, val_targets = train_test_split(
            text,
            targets,
            test_size=config['data']['test_size'],
            random_state=config["seed"]
        )

        logger.info('Getting  tf-idf embeddings...')
        train_features, val_features = tf_idf(train_text, val_text, config, logger)

        logger.info('Training Logistic Regression model...')
        train_logreg(train_features, val_features, train_targets, val_targets, config, logger)
    elif model_type == 'bert':
        logger.info('Bert model')

        logger.info('Downloading pretrained bert tokenizer...')
        bert_tokenizer = BertTokenizerFast.from_pretrained(config['bert']['bert_model_name'])

        train_dataloader, val_dataloader = get_dataloaders(data, bert_tokenizer, config)

        logger.info('Training bert model...')
        best_valid_loss = train_bert(data, train_dataloader, val_dataloader, config, logger, device)
        logger.info(f'Best val loss: {best_valid_loss}')

    logger.info('End\n')
    close_logger(logger)

    return 0


if __name__ == '__main__':
    main()
