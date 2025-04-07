import json
import sys
from collections import defaultdict

def process_file(file_path):
    totals = defaultdict(float)
    counts = defaultdict(int)
    
    try:
        with open(file_path, 'r') as file:
            # Skip the first three lines
            for _ in range(3):
                next(file, None)
            
            # Process each line
            line_number = 4  # Start at line 4 (after skipping 3 lines)
            for line in file:
                try:
                    # Try to parse JSON from the line
                    data = json.loads(line.strip())
                    
                    # Check if profile_time exists in the data
                    if 'profile_time' in data:
                        # Sum up values for each key in profile_time
                        for key, value in data['profile_time'].items():
                            totals[key] += value
                            counts[key] += 1
                except json.JSONDecodeError:
                    # Skip lines that aren't valid JSON
                    continue
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")
                
                line_number += 1
        
        # Calculate and print averages
        print("\nAverage values in profile_time:")
        print("-" * 40)
        
        for key in sorted(totals.keys()):
            if counts[key] > 0:
                average = totals[key] / counts[key]
                print(f"{key}: {average:.10f}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        return
    
    file_path = sys.argv[1]
    process_file(file_path)

if __name__ == "__main__":
    main()