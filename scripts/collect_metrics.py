import os
import re
import csv

main_dir = 'scripts/LRU'

scripts_dir = 'scripts'
os.makedirs(scripts_dir, exist_ok=True)

cache_sizes = []
thresholds = []
policies = []
vector_db_latencies = []
retrieve_latencies = []
hit_rates = []
accuracies = []

retrieve_pattern = r'database\.MyScaleSearcher\.retrieve\.profile: (\d+\.\d+)'
search_pattern = r'database\.MyScaleSearcherWithCache\.search\.profile: (\d+\.\d+)'
hits_pattern = r'hits: (\d+\.\d+)'
accuracy_pattern = r'Accuracy: (\d+\.\d+)%'

cache_size_pattern = r'cache_size_(\d+)'
threshold_pattern = r'threshold_(\d+)'
policy_pattern = r'policy_(\w+)'

for subdir in os.listdir(main_dir):
    if subdir.startswith('cache_size'):
        folder_path = os.path.join(main_dir, subdir, 'mmlu_prehistory')
        
        cache_size_match = re.search(cache_size_pattern, subdir)
        threshold_match = re.search(threshold_pattern, subdir)
        policy_match = re.search(policy_pattern, subdir)
        
        if cache_size_match:
            cache_sizes.append(int(cache_size_match.group(1)))
        else:
            cache_sizes.append(None)
            
        if threshold_match:
            thresholds.append(int(threshold_match.group(1)))
        else:
            thresholds.append(None)
            
        if policy_match:
            policies.append(policy_match.group(1))
        else:
            policies.append(None)
        
        analysis_file = os.path.join(folder_path, '5_m100_p40_gpt35_analysis.txt')
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                content = f.read()
                
                retrieve_match = re.search(retrieve_pattern, content)
                if retrieve_match:
                    vector_db_latencies.append(float(retrieve_match.group(1)) * 1000)
                else:
                    vector_db_latencies.append(None)
                
                search_match = re.search(search_pattern, content)
                if search_match:
                    retrieve_latencies.append(float(search_match.group(1)) * 1000)
                else:
                    retrieve_latencies.append(None)
                
                hits_match = re.search(hits_pattern, content)
                if hits_match:
                    hit_rates.append(float(hits_match.group(1)) * 100)
                else:
                    hit_rates.append(None)
        else:
            vector_db_latencies.append(None)
            retrieve_latencies.append(None)
            hit_rates.append(None)
        
        accuracy_file = os.path.join(folder_path, '5_m100_p40_gpt35.jsonl')
        if os.path.exists(accuracy_file):
            with open(accuracy_file, 'r') as f:
                first_line = f.readline()
                accuracy_match = re.search(accuracy_pattern, first_line)
                if accuracy_match:
                    accuracies.append(float(accuracy_match.group(1)))
                else:
                    accuracies.append(None)
        else:
            accuracies.append(None)

with open(os.path.join(scripts_dir, 'metrics.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Cache Size', 'Threshold', 'Policy', 'VectorDB Latency (ms)', 
                     'Retrieve Latency (ms)', 'Accuracy', 'Hit Rate (%)'])
    
    for i in range(len(cache_sizes)):
        writer.writerow([
            cache_sizes[i],
            thresholds[i],
            policies[i],
            vector_db_latencies[i],
            retrieve_latencies[i],
            accuracies[i],
            hit_rates[i]
        ])

print(f"Combined metrics CSV file has been generated successfully in the '{scripts_dir}' folder.")