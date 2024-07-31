import torch


def calculate_batching_parameters(args, train_dataset):
    if args.effective_batch_size > args.batch_size:
        num_devices = torch.cuda.device_count() if torch.cuda.device_count() > 1 else 1  # for CPU
        avg_length = train_dataset.avg()  # Assuming this is a method to get the average length
        args.grad_accum = int((args.effective_batch_size / avg_length) / (args.batch_size * num_devices))
        args.grad_accum = max(args.grad_accum, 1)  # Ensure grad_accum is at least 1
        if args.grad_accum == 1:
            args.effective_batch_size = avg_length * args.batch_size * num_devices
        print('\n-----Batching Summary-----\n')
        print(f'Number of devices: {num_devices}')
        print(f'Average sequence length: {avg_length}')
        print(f'Local batch size: {args.batch_size} seqs')
        print(f'Gradient accumulation: {args.grad_accum}')
        print(f'Effective batch size: {int(args.effective_batch_size)} tokens')
    else:
        args.grad_accum = 1

    return args
