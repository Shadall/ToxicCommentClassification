import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup
from Toxic_comment_classification.model import ToxicClassifier
from utils import get_warmup_steps, set_seed
from joblib import dump, load


def train_bert(data, train_dataloader, val_dataloader, config, logger, device):
    """Function to train and evaluate BertModel

    Args:
        data: data to train on
        train_dataloader: train dataloader
        val_dataloader: validation dataloader
        config: config
        logger: logger
        device: device (cuda/cpu)

    :returns: best loss on validation step
    :rtype: float
    """
    config['data']['checkpoint_path'] = os.path.join(config['data']['output_path'],
                                                     'bert_model_checkpoint.pt')
    config['data']['best_model_path'] = os.path.join(config['data']['output_path'], 'bert_best_model.pt')
    checkpoint = os.path.isfile(config['data']['checkpoint_path'])

    set_seed(config['seed'])

    epochs = config['bert']['epochs']
    target_columns = config['data']['target_columns']
    criterion = nn.BCELoss()

    logger.info('  Downloading pretrained bert embeddings...')
    bert_model = ToxicClassifier(len(target_columns), bert_model_name=config['bert']['bert_model_name']).to(device)
    optimizer = AdamW(bert_model.parameters(), lr=config['bert']['learning_rate'])

    if not checkpoint:
        start_epoch = -1
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
    else:
        logger.info('  Initializing bert embeddings from checkpoint...')

        checkpoint = torch.load(config['data']['checkpoint_path'])
        bert_model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['valid_losses']
        best_val_loss = checkpoint['best_val_loss']

    warmup_steps, total_training_steps = get_warmup_steps(data, config['bert']['warmup_frac'], config)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps, last_epoch=start_epoch)

    for epoch in range(start_epoch + 1, epochs):
        logger.info(f'  Epoch {epoch + 1} / {epochs}')

        train_loss = train_step(bert_model, train_dataloader, optimizer, scheduler, criterion, device)
        val_loss = evaluate_step(bert_model, val_dataloader, target_columns, criterion, logger, device)

        train_losses.append(round(train_loss, 3))
        val_losses.append(round(val_loss, 3))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config['save_model']:
                logger.info(f'  Found a better model with val loss {best_val_loss:.3}, saving model in output...')
                torch.save(bert_model.state_dict(), config["data"]["best_model_path"])

        torch.save({
            'epoch': epoch,
            'model_state_dict': bert_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': val_losses,
            'best_val_loss': best_val_loss
        }, config['data']['checkpoint_path'])

    logger.info(f'    Train Epoch Losses: {train_losses}')
    logger.info(f'      Val Epoch Losses: {val_losses}')

    return best_val_loss


def train_step(bert_model, train_dataloader, optimizer, scheduler, criterion, device):
    """Function for training step of BertModel

    Args:
        bert_model: BertModel
        train_dataloader: train dataloader
        optimizer: optimizer for training
        scheduler: scheduler for training
        criterion: criterion
        device: device (cuda/cpu)

    :returns: train loss
    :rtype: float
    """
    bert_model.train()

    total_loss, total_accuracy = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader)):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        bert_model.zero_grad()
        outputs = bert_model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_dataloader)


def evaluate_step(bert_model, val_dataloader, target_columns, criterion, logger, device):
    """Function for evaluation step of BertModel

    Args:
        bert_model: BertModel
        val_dataloader: validation dataloader
        target_columns: types of toxicity (we need to predict a probability of each type)
        criterion: criterion
        logger: logger
        device: device (cuda/cpu)

    :returns: validation loss
    :rtype: float
    """
    bert_model.eval()

    total_loss, total_accuracy = 0, 0
    total_preds, total_labels = [], []
    roc_auc_scores, precision_scores, recall_scores = [], [], []

    for step, batch in enumerate(tqdm(val_dataloader)):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            total_preds.append(outputs)
            total_labels.append(labels)

    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    true = np.array(total_labels)
    pred_prob = np.array(total_preds)
    pred = np.array(total_preds > 0.5)

    for i, target_column in enumerate(target_columns):
        logger.info(f'      Class {target_column}:')
        try:
            roc_auc = roc_auc_score(true[:, i], pred_prob[:, i])
            precision = precision_score(true[:, i], pred[:, i], zero_division=0)
            recall = recall_score(true[:, i], pred[:, i], zero_division=0)

            roc_auc_scores.append(roc_auc)
            precision_scores.append(precision)
            recall_scores.append(recall)

            logger.info(f'        ROC_AUC:   {roc_auc:.3}')
            logger.info(f'        Precision: {precision:.3}')
            logger.info(f'        Recall:    {recall:.3}')
        except Exception as e:
            logger.info(f'        Exception: {e}')

    logger.info(f'      Average:')
    logger.info(f'        ROC_AUC:   {np.mean(roc_auc_scores):.3}')
    logger.info(f'        Precision: {np.mean(precision_scores):.3}')
    logger.info(f'        Recall:    {np.mean(recall_scores):.3}')

    return total_loss / len(val_dataloader)


def train_logreg(train_features, val_features, train_targets, val_targets, config, logger):
    """Function to train and evaluate LogisticRegression

    Args:
        train_features: matrix of tf-idf train features
        val_features: matrix of tf-idf validation features
        train_targets: targets from train data
        val_targets: targets from validation data
        config: config
        logger: logger

    :returns: best loss on validation step
    :rtype: float
    """
    config['data']['model_path'] = os.path.join(config['data']['output_path'],
                                                'logreg_model.joblib')
    checkpoint = os.path.isfile(config['data']['model_path'])

    set_seed(config['seed'])

    model = None
    roc_auc_scores, precision_scores, recall_scores = [], [], []
    target_columns = config['data']['target_columns']
    for target_column in target_columns:
        train_target = train_targets[target_column]
        val_target = val_targets[target_column]

        if not checkpoint:
            model = LogisticRegression(**config['logreg'], random_state=config['seed'])
            model.fit(train_features, train_target)
        else:
            try:
                model = load(config['data']['model_path'])
            except Exception as e:
                logger.info(f'    Exception: {e}')

        pred_prob = model.predict_proba(val_features)[:, 1]
        pred = model.predict(val_features)

        logger.info(f'      Class {target_column}:')
        try:
            roc_auc = roc_auc_score(val_target, pred_prob)
            precision = precision_score(val_target, pred, zero_division=0)
            recall = recall_score(val_target, pred, zero_division=0)

            roc_auc_scores.append(roc_auc)
            precision_scores.append(precision)
            recall_scores.append(recall)

            logger.info(f'        ROC_AUC:   {roc_auc:.3}')
            logger.info(f'        Precision: {precision:.3}')
            logger.info(f'        Recall:    {recall:.3}')
        except Exception as e:
            logger.info(f'        Exception: {e}')

    if config['save_model'] and not checkpoint:
        logger.info('Saving model in output...')
        dump(model, config['data']['model_path'])

    logger.info(f'    Average:')
    logger.info(f'      ROC_AUC:   {np.mean(roc_auc_scores):.3}')
    logger.info(f'      Precision: {np.mean(precision_scores):.3}')
    logger.info(f'      Recall:    {np.mean(recall_scores):.3}')
