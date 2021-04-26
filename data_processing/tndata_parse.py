# Author: Yuexin Li
# Purpose: 
#         (1) Parse (.wav, label_info) from this dataset (https://www.openslr.org/18/)
#         (2) Get IPA form of .wav files 

# ---------------------- Helper Functions -------------------------
def process_f(root, filename):
    res = []
    filepath = root + filename

    # get phonemic transcription
    command = 'python3 -m allosaurus.run -i ' + filepath
    result = subprocess.check_output(command, shell=True)
    result_list = result.split()
    phones_list = [phone.decode('utf-8') for phone in result_list]
    
    # format res (csv column order)
    res = [filename, phones_list]
    return res

# output data to csv file
def save_file(rows, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "phones"])
        writer.writerows(rows)

if __name__ == "__main__":
    import argparse, os, subprocess, csv

    parser = argparse.ArgumentParser(description='null')

    # hard-coded multi-processing
    parser.add_argument('partition_prefix', metavar='p', type=str, help='partition_id: letter A, B, C, D')
    parser.add_argument('worker_id', metavar='i', type=int, help='worker_id')
    parser.add_argument('output_path', metavar='o', type=str, help='output_path')
    args = parser.parse_args()

    root = "./data_thchs30/train/"
    rows = []
    i = 0

    for filename in os.listdir(root):
        if filename.startswith(args.partition_prefix) and filename.endswith('.wav'):
            print(filename)
            i += 1

            res = process_f(root, filename)
            rows.append(res)
            
            if i % 10 == 0:
                print("i=%d" % i)

    print(len(rows))
    save_file(rows, args.output_path)

