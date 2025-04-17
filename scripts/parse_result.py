import json
import sys
import os
from collections import defaultdict

def write_output(file_or_stdout, message):
    if file_or_stdout == 'stdout':
        print(message, end='')
    else:
        file_or_stdout.write(message)

def process_file(input_file_path, output_file_path):
    totals = defaultdict(float)
    counts = defaultdict(int)
    
    # Check if input file exists, create it if not
    if not os.path.exists(input_file_path):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
            # Create the empty file
            with open(input_file_path, 'w') as f:
                pass
            print(f"Created empty file: {input_file_path}")
        except Exception as e:
            print(f"Error creating input file: {e}")
            return
    
    try:
        with open(input_file_path, 'r') as file:
            # Skip the first three lines (if they exist)
            for _ in range(3):
                next(file, None)
            
            # Process each line
            line_number = 4  # Start at line 4 (after skipping 3 lines)
            
            # Open output file or use stdout
            if output_file_path == 'stdout':
                output = 'stdout'
            else:
                # Create directory for output file if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                output = open(output_file_path, 'w')
            
            try:
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
                        write_output(output, f"Error processing line {line_number}: {e}\n")
                    
                    line_number += 1
                
                # Write averages
                write_output(output, "\nAverage values in profile_time:\n")
                write_output(output, "-" * 40 + "\n")
                
                for key in sorted(totals.keys()):
                    if counts[key] > 0:
                        average = totals[key] / counts[key]
                        write_output(output, f"{key}: {average:.10f}\n")
            
            finally:
                # Close the output file if it's not stdout
                if output_file_path != 'stdout':
                    output.close()
        
    except Exception as e:
        # Handle any other errors
        if output_file_path == 'stdout':
            print(f"Error processing file: {e}")
        else:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as output:
                output.write(f"Error processing file: {e}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file_path> <output_file_path>")
        return
    
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    process_file(input_file_path, output_file_path)

if __name__ == "__main__":
    main()