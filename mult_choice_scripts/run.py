from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoModelForMultipleChoice
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import wandb
import numpy as np
import pandas as pd

from mult_choice_scripts.metrics import compute_map_at_3
from mult_choice_scripts.data import DataCollatorForMultipleChoice, load_data
from mult_choice_scripts.model import get_model, get_tokenizer

def train(config):

    # Logging config
    if config.no_wandb == True or config.fast_dev_run == True:
        report_to = 'none'
    else:
        report_to = "wandb"

    # Optional: Early stopping
    if config.no_early_stop == True:
        load_best_model_at_end = False
        callbacks = []
    else:
        load_best_model_at_end = True
        callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]

    # BF16 over FP16
    if config.fp16 and config.bf16:
        config.fp16 = False

    # Tokenizer + Data
    model = get_model(config)
    tokenizer = get_tokenizer(config=config)

    train_dataset, valid_dataset = load_data(config=config)

    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir + "{}_{}_{}".format(config.run_name, config.seed, config.file_name.split(".")[0]),
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        # per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        report_to=report_to,
        save_total_limit=2,
        disable_tqdm=config.disable_tqdm,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_accumulation_steps=config.grad_acc_steps,
        # eval_accumulation_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        # evaluation_strategy="steps",
        logging_steps=config.logging_steps,
        save_steps=config.logging_steps,
        # eval_steps=config.logging_steps,
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False, # allows new data collator to work
        dataloader_num_workers=config.num_workers,
    )

    data_collator = DataCollatorForMultipleChoice(
        tokenizer=tokenizer,
        option_to_index={option: idx for idx, option in enumerate('ABCDE')},
        config=config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=valid_dataset,
        compute_metrics=compute_map_at_3,
        callbacks=callbacks,
    )

    # Set up LR scheduler
    trainer.create_optimizer()
    num_steps = (len(train_dataset)*config.epochs) // (config.grad_acc_steps*config.batch_size)

    trainer.lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=trainer.optimizer,
        num_warmup_steps=int(num_steps*config.warmup_ratio),
    )

    # Training
    print("-"*25 + " TRAINING " + "-"*25 + "\n")
    trainer.train(
        resume_from_checkpoint=config.resume,
        )
    trainer.save_model(config.model_save_dir + "{}_{}".format(config.run_name, config.seed))

    # log wandb config (hacky fix..)
    if config.no_wandb == False and config.fast_dev_run == False:
        wandb.config.update(config.__dict__, allow_val_change=True)

    # Predicting
    if valid_dataset is not None:
        print("-"*25 + " PREDICTING " + "-"*25 + "\n")
        result = trainer.predict(
            test_dataset=valid_dataset,
        )
        print(result.metrics)

def valid(config):

    # Tokenizer + Data
    # model = get_model(config)
    model = AutoModelForMultipleChoice.from_pretrained(config.model_name)
    tokenizer = get_tokenizer(config=config)

    train_dataset, valid_dataset = load_data(config=config, valid_only=True)

    training_args = TrainingArguments(
        output_dir = "./tmp/",
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        report_to='none',
        bf16=config.bf16,
        fp16=config.fp16,
        logging_first_step=True,
        evaluation_strategy="steps",
        save_strategy="epoch",
        logging_steps=config.logging_steps,
        save_steps=config.logging_steps,
        eval_steps=config.logging_steps,
        seed=config.seed,
        data_seed=config.seed,
        # disable_tqdm=True,
        remove_unused_columns=False, # allows new data collator to work
    )

    data_collator = DataCollatorForMultipleChoice(
        tokenizer=tokenizer,
        option_to_index={option: idx for idx, option in enumerate('ABCDE')},
        config=config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=valid_dataset,
        compute_metrics=compute_map_at_3,
    )

    # Predicting
    print("-"*25 + " PREDICTING " + "-"*25 + "\n")
    result = trainer.predict(
        test_dataset=valid_dataset,
    )
    print(result.metrics)

    # Creating prediction DF
    df = pd.DataFrame(result.predictions, columns=list("ABCDE"))
    df["pred"] = df.apply(lambda row: "ABCDE"[row.argmax()], axis=1)
    df["label"] = ["ABCDE"[x] for x in result.label_ids]
    df = df.drop(columns=list("ABCDE"))
    df.to_csv("./tmp/preds.csv", index=False)