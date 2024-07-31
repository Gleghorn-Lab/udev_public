import pandas as pd
from functools import partial
from datasets import Dataset
from tqdm import tqdm
import time


def pd_to_hf_in_sections(path, section_size, hf_path, delimiter='\t', max_retries=3, retry_delay=5, skip_sections=0):
    # path to delimited file: str
    # number of rows per section: int
    # hub path for huggingface: str
    # text file delimiter: str
    # max_retries: int (maximum number of retry attempts)
    # retry_delay: int (delay in seconds between retries)
    
    read_file = partial(pd.read_csv, delimiter=delimiter, low_memory=False)
    header = list(read_file(path, nrows=2).columns) # get column names
    print('Counting lines')
    total_rows = sum(1 for _ in open(path))
    
    for i, current in tqdm(enumerate(range(0, total_rows, section_size)), desc='Processing', total=total_rows//section_size + 1 - skip_sections):
        retries = 0
        while retries < max_retries:
            try:
                data = Dataset.from_pandas(read_file(path, skiprows=range(0, current+1+(skip_sections*section_size)), nrows=section_size, names=header))
                split = f'section{i+skip_sections}'
                data.push_to_hub(hf_path, split=split)
                break  # If successful, break out of the retry loop
            except:
                retries += 1
                if retries < max_retries:
                    print(f"Error occurred. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to push section {i+skip_sections} after {max_retries} attempts. Moving to next section.")
        
        if retries == max_retries:
            print(f"Warning: Section {i} was not successfully pushed to the hub.")
