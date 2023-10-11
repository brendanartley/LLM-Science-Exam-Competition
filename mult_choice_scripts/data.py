from functools import partial
import torch
import pandas as pd
import datasets
from datasets import Dataset
import numpy as np

# Mutes TQDM for datasets
datasets.disable_progress_bar()
    
class DataCollatorForMultipleChoice:
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None, option_to_index=None, config=None, valid=False):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.option_to_index = option_to_index
        self.config = config
        self.valid = valid

    def preprocess(self, example):
        """
        Preprocess w/ out answers.
        """
        first_sentence = [ "[CLS] " + example['context']] * 5
        second_sentences = [" #### {} [SEP] {} [SEP]".format(example['prompt'], example[option]) for option in 'ABCDE']
        tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation='only_first', max_length=self.config.max_tokens, add_special_tokens=False)
        if self.config.kaggle == True:
            tokenized_example["id"] = [example["id"]]*5
        tokenized_example['label'] = self.option_to_index[example['answer']]
        return tokenized_example
    
    def preprocess_with_answers(self, example):
        """
        Preprocess w/ answers.
        """
        first_sentence = [example['context']] * 5
        options = [example[x] for x in "ABCDE"]

        # Add answers in random order
        for i in range(5):
            other_options = options[:i] + options[i+1:]
            # Shuffle answer options
            if self.valid == False: 
                np.random.shuffle(other_options)
            first_sentence[i] =  "[CLS] " + "(o) ".join(other_options) + " ### " + first_sentence[i]
        second_sentences = [" ### {} [SEP] {} [SEP]".format(example['prompt'], example[option]) for option in 'ABCDE']

        # Tokenize texts
        tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation='only_first', max_length=self.config.max_tokens, add_special_tokens=False)
        if self.config.kaggle == True:
            tokenized_example["id"] = [example["id"]]*5
        tokenized_example['label'] = self.option_to_index[example['answer']]
        return tokenized_example

    def __call__(self, examples):
        if self.config.with_answers == True:
            tokenized_examples = [self.preprocess_with_answers(example) for example in examples]
        else:
            tokenized_examples= [self.preprocess(example) for example in examples]

        label_name = 'label' if 'label' in tokenized_examples[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in tokenized_examples]
        batch_size = len(tokenized_examples)
        num_choices = len(tokenized_examples[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in tokenized_examples
        ]
        flattened_features = sum(flattened_features, [])  # flattens array
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
    

def load_data(config, valid_only=False):
    
    # Loading CSV Data
    if valid_only == False:
        df_train = pd.read_csv(config.data_dir + "{}/{}".format(config.q_data_version, config.file_name)) \
                     .sample(frac=1, random_state=0)
        
        # These IDXs make token length fail for 512 + new preprocess funnction
        df_train = df_train[~df_train.index.isin([21577, 27521, 9242, 4028, 8593, 19977, 26948, 17389])]
    else:
        df_train = None

    # All validation
    test_df = pd.read_csv(config.data_dir + "{}/{}".format(config.q_data_version, config.file_name.replace("all_data", "train")))

    # Training on all the data
    if config.train_all == True:
        df_train = pd.concat([df_train, test_df], axis=0, ignore_index=True).sample(frac=1, random_state=config.seed)
        test_df = None

    # Testing run
    if config.fast_dev_run == True:
        if valid_only == False:
            df_train = df_train.head(100)
        if config.train_all == False:
            test_df = test_df.head(100)

    # Fill nan values
    for col in "ABCDE":
        if df_train is not None:
            df_train[col] = df_train[col].fillna("None")
        if test_df is not None:
            test_df[col] = test_df[col].fillna("None")

    # Log shapes
    print(df_train.shape if df_train is not None else None, test_df.shape if test_df is not None else None)

    # Create train + valid dataset
    if df_train is not None:
        dataset = Dataset.from_pandas(df_train)
    else:
        dataset = None

    if test_df is not None:
        valid_dataset = Dataset.from_pandas(test_df)
    else:
        valid_dataset = None

    return dataset, valid_dataset