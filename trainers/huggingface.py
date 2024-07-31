### imports
import torch
from typing import Optional
from transformers import Trainer, TrainingArguments
from transformers.trainer import has_length
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
from torch.utils.data import Dataset as TorchDataset


class SortedTrainer(Trainer):
    """
    A custom Trainer class inheriting from HuggingFace's Trainer, created to prioritize loading of
    larger batches first. This approach ensures that if there are out-of-memory errors due to
    VRAM/RAM limitations, they occur early on. 
    If `group_by_length` is not enabled, it defaults to a random sampling strategy. TODO: Evaluate

    """
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, TorchDataset):
                lengths = self.train_dataset.lengths # this requires your dataset has a self.lengths with the lengths in it
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(self.train_dataset)


def hf_trainer_from_kwargs(model,
               train_dataset,
               valid_dataset=None,
               compute_metrics=None,
               data_collator=None,
               callbacks=None,
               token=None,
               **kwargs):
    """
    Builds huggingface trainer given settings
    """
    
    if 'output_dir' not in kwargs:
        kwargs['output_dir'] = './output'  # default output directory
    
    training_args = TrainingArguments(load_best_model_at_end=True if callbacks != None else False,
                                      hub_token=token,
                                      **kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )
    return trainer


def hf_trainer_from_args(args,
                         model,
                         train_dataset,
                         valid_dataset=None,
                         compute_metrics=None,
                         data_collator=None,
                         callbacks=None,
                         **kwargs):
    """
    args should contain batch_size, lr, grad_accum, save_path, num_epochs, effective_batch_size (optional)
    """
    try:
        effective_batch_size = args.effective_batch_size
    except:
        effective_batch_size = None
    if effective_batch_size != None:
        try:
            from ..utils.batching import calculate_batching_parameters
            args = calculate_batching_parameters(args, train_dataset)
        except:
            print('There was either an issue with your train_dataset.__avg__() or importing from batching')

    training_args = TrainingArguments(
        output_dir=args.save_path.split('/')[-1],
        logging_dir="./logs",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps",
        logging_steps=100,
        disable_tqdm=False,
        load_best_model_at_end=True if callbacks != None else False,
        hub_token=args.token,
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )
    return trainer


"""
TODO

Add trainer to work with freeze() and unfreeze()
train parts of a model and then the whole thing for a percentage of the total batches

"""