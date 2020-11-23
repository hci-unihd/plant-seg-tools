import argparse
import glob
import re
import os


def main():
    parser = argparse.ArgumentParser(description='Rename files in a  time series by a specific offset.')
    parser.add_argument('--source', type=str, help='Path to the source files', required=True)
    parser.add_argument('--offset', type=int, help='Offset to apply to the timestamps (can be negative)', required=True)

    args = parser.parse_args()
    infiles = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]

    for file in infiles:
        old_name = os.path.split(file)[1]
        if re.search('[tT](\d{1,})', old_name):
            tp = int(re.search('[tT](\d{1,})', old_name).group(1))
            new_name = re.sub('[tT]\d{1,}', f't{str(tp + args.offset)}', old_name)
            os.rename(os.path.join(args.source, old_name), os.path.join(args.source, new_name))



if __name__ == '__main__':
    print("Starting Offset Time Series")
    main()
    print("Done!")
