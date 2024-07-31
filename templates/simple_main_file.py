import argparse
import torch
from torchinfo import summary
from transformers import AutoTokenizer
from datasets import load_dataset
### Search through files to find the functions / classes you need
from udev.models import *
from udev.data.nlp_dataset_classes import *
from udev.metrics import *
#from udev.data.image_dataset_classes import *
from udev.data.data_collators import *
from udev.trainers.huggingface import *


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--data_path', type=str, default='lhallee/annotations_uniref90')
    return parser.parse_args()


def main():
    args = get_args()
    args.num_epochs, args.grad_accum = 1, 1

    ### Assign from the correct import
    DATASET = # the dataset class you need
    DATA_COLLATOR = # the collator you need
    GET_TRAINER = # the trainer you need
    MODEL = # the model you need
    CONFIG = # the config you need
    COMPUTE_METRICS = # the metrics you need

    data = load_dataset(args.data_path, token=args.token)
    train = data['train'].shuffle(seed=42)
    valid = data['valid']
    test = data['test']

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.token)
    train_dataset = DATASET(train, tokenizer=tokenizer)
    valid_dataset = DATASET(valid, tokenizer=tokenizer)
    test_dataset = DATASET(test, tokenizer=tokenizer)

    ### Process a list into tensors, or other data processing
    """
    from functools import partial
    from udev.data.data_utils import process_column
    col_name = 'combined'
    process = partial(process_column, col_name=nlp_col)
    data = data.map(process)
    """

    config = CONFIG()
    model = MODEL(config)
    summary(model)

    save_path = f'' # make a descriptive path based on settings and your HF username
    args.save_path = save_path

    ### Get trainer from args or kwargs
    kwargs = {
        'eval_strategy': 'steps',
        'eval_steps':100,
        'save_strategy': 'steps',
        'save_steps':1000
    }

    ### Get trainer from args or kwargs
    trainer = GET_TRAINER(args,
                          model,
                          train_dataset,
                          valid_dataset,
                          data_collator=DATA_COLLATOR,
                          compute_metrics=COMPUTE_METRICS,
                          **kwargs)
    metrics = trainer.evaluate(eval_dataset=test_dataset) # evaluate with random weights
    print(f'Random weight metrics: \n{metrics}')
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=test_dataset) # evaluate with trained weights
    print(f'Trained weight metrics: \n{metrics}')

    trainer.model.push_to_hub(save_path, token=args.token, private=True)
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
