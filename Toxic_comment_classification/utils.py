import spacy
import numpy as np
import random
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from Toxic_comment_classification.dataset import ToxicCommentsDataset

nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words


def set_seed(seed):
    """Set seed for reproducibility

    Args:
        seed (int): Seed
    """

    random.seed(seed)
    np.random.seed(seed)


def spacy_tokenizer(sentence):
    """Spacy preprocessor for TfidfVectorizer

    Args:
        sentence: comment

    :returns: joined tokens
    :rtype: str
    """

    tokens = nlp(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    return ' '.join(tokens)


def lemmatize(document):
    """Lemmatizer

    Args:
        document: document to lemmatize

    :returns: list of docement tokens
    :rtype: list
    """

    lemma_list = [str(token.lemma_).lower() for token in document if
                  token.is_alpha and token.text.lower() not in stop_words]

    return lemma_list


def preprocess(text):
    """Sentence preprocessor

    Args:
        text: text to preprocess

    :returns: list of lemmatized documents
    :rtype: list
    """

    preproc_pipe = []

    for document in nlp.pipe(text, batch_size=32, n_process=-1, disable=["parser", "ner"]):
        preproc_pipe.append(' '.join(lemmatize(document)))

    return preproc_pipe


def tf_idf(train_text, val_text, config, logger):
    """Get tf-idf features from text

    Args:
        train_text: text from train data to train TfidfVectorizer on
        val_text: text from validation data to transform TfidfVectorizer on
        config: config
        logger: logger

    :returns: matrix of tf-idf features
    """

    word_vectorizer = TfidfVectorizer(preprocessor=spacy_tokenizer, **config['tf-idf']['word'])
    char_vectorizer = TfidfVectorizer(preprocessor=spacy_tokenizer, **config['tf-idf']['char'])

    logger.info('  Fitting word tf-idf')
    word_vectorizer.fit(train_text)
    word_train_features = word_vectorizer.transform(train_text)

    logger.info('  Fitting char tf-idf')
    char_vectorizer.fit(train_text)
    char_train_features = char_vectorizer.transform(train_text)

    logger.info('  Transform val text')
    word_val_features = word_vectorizer.transform(val_text)
    char_val_features = char_vectorizer.transform(val_text)

    train_features = hstack([word_train_features, char_train_features])
    val_features = hstack([word_val_features, char_val_features])

    return train_features, val_features


def get_dataloaders(data, bert_tokenizer, config):
    """Form dataset and get DataLoaders

    Args:
        data: data to form dataset on
        bert: BertTokenizer
        config: config

    :returns: train dataloader, validation dataloader
    :rtype: (torch.DataLoader, torch.DataLoader)
    """

    train_df, val_df = train_test_split(data, test_size=config['data']['test_size'], random_state=config['seed'])
    target_columns = config['data']['target_columns']

    train_dataset = ToxicCommentsDataset(
        data=train_df,
        tokenizer=bert_tokenizer,
        target_columns=target_columns,
        max_token_len=config['bert']['max_token_len']
    )
    val_dataset = ToxicCommentsDataset(
        data=val_df,
        tokenizer=bert_tokenizer,
        target_columns=target_columns,
        max_token_len=config['bert']['max_token_len']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config['bert']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['bert']['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader


def get_warmup_steps(data, warmup_frac, config):
    """Count how many warm-up steps needed and total training steps

    Args:
        data: data
        warmup_frac: proportion of training steps to warm-up
        config: config

    :returns: warmup steps and total training steps counts
    :rtype: (int, int)
    """
    epochs = config['bert']['epochs']
    steps_per_epoch = len(data) // config['bert']['batch_size']
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_frac * total_training_steps)

    return warmup_steps, total_training_steps
