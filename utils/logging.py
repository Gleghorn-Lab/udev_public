import csv
import re


def process_results_files(file_list, title_list):
    # Define the regex patterns to match each metric
    loss_pattern = r'test_loss: (\d+\.\d+)'
    f1_pattern = r'test_f1: (\d+\.\d+)'
    precision_pattern = r'test_precision: (\d+\.\d+)'
    recall_pattern = r'test_recall: (\d+\.\d+)'
    accuracy_pattern = r'test_accuracy: (\d+\.\d+)'

    domains = ['EC', 'CC', 'MF', 'BP', 'MI', 'DL2', 'DL10']

    # Open a new CSV file for writing
    with open('results_summary.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header row
        writer.writerow(['Title', 'Loss', 'F1', 'Precision', 'Recall', 'Accuracy'])
        
        # Process each file in the file list
        for file_name, title in zip(file_list, title_list):
            # Open the text file and read its contents
            with open(file_name, 'r') as file:
                content = file.read()

            # Find all matches for each metric in the content
            loss_matches = re.findall(loss_pattern, content)
            f1_matches = re.findall(f1_pattern, content)
            precision_matches = re.findall(precision_pattern, content)
            recall_matches = re.findall(recall_pattern, content)
            accuracy_matches = re.findall(accuracy_pattern, content)

            # Convert the matches to floats
            loss_values = [float(match) for match in loss_matches]
            f1_values = [float(match) for match in f1_matches]
            precision_values = [float(match) for match in precision_matches]
            recall_values = [float(match) for match in recall_matches]
            accuracy_values = [float(match) for match in accuracy_matches]

            # Write each set of values as a row in the CSV file
            for i in range(len(loss_values)):
                writer.writerow([
                    f'{title}_run{domains[i]}',
                    loss_values[i],
                    f1_values[i],
                    precision_values[i],
                    recall_values[i],
                    accuracy_values[i]
                ])

            # Calculate the averages for each metric
            avg_loss = sum(loss_values) / len(loss_values)
            avg_f1 = sum(f1_values) / len(f1_values)
            avg_precision = sum(precision_values) / len(precision_values)
            avg_recall = sum(recall_values) / len(recall_values)
            avg_accuracy = sum(accuracy_values) / len(accuracy_values)

            # Write the averages as a row in the CSV file
            writer.writerow([f'{title}_average', avg_loss, avg_f1, avg_precision, avg_recall, avg_accuracy])


def log_results(cfg, results, data_path='path'):
    keys_to_exclude = {'test_runtime', 'test_samples_per_second', 'test_steps_per_second'}
    filtered_results = {k: v for k, v in results.items() if k not in keys_to_exclude}
    for key, value in filtered_results.items():
        if isinstance(value, (float, int)):
            filtered_results[key] = round(value, 5)

    with open(cfg.log_path, 'a') as file:
        file.write('Data Path: ' + data_path + '\n')
        file.write('Results:\n')  # Section title
        for key, value in filtered_results.items():
            file.write(f'{key}: {value}\n')
        file.write('\n\n')