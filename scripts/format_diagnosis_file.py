import pandas as pd
import sys
import os

def main(infile, outfile):
    # Read all lines from file
    with open(infile, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Create DataFrame with single column
    df = pd.DataFrame({'full_text': lines})

    # Split at first space
    df[['icd10_code', 'description']] = df['full_text'].str.split(' ', n=1, expand=True)

    # Drop the original column
    df = df[['icd10_code', 'description']]

    print(df[0:5])
    df.to_csv(outfile, index=False)
    return()

if __name__ == '__main__':
    ipath = "../data/raw/"
    opath = '../data/processed/'
    print('parameters length',len(sys.argv))
    if len(sys.argv) == 3:
        # Run with command-line arguments
        print("got here")
        infile = os.path.join(ipath,sys.argv[1])
        outfile = os.path.join(opath,sys.argv[2])
    else:
        # Default files
        print("got here 2")
        infile = os.path.join(ipath,'icd10_codes.txt')
        outfile = os.path.join(opath, 'icd10_descriptions.csv')
    
    main(infile, outfile)