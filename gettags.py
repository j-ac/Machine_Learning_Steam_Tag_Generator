#!/usr/bin/env python3

import sys
import re
import csv


def associate_id_to_tag(i_file, t_file):
    associations = {}
    with open(i_file, 'r') as icsv:
        iReader = csv.reader(icsv)
        next(iReader) #skip over column identifier
        with open(t_file, 'r') as tcsv:
            tReader = csv.reader(tcsv)
            next(tReader) #skip cols again

            # This part uneccesarially iterates the entire file but it'll still be fast
            for trow in tReader:
                tags = re.sub('[^A-Za-z0-9 ]+', '', trow[0]).split()
                ids = re.sub('[^0-9 ]+', '', next(iReader)[0]).split() 
                for i, t in enumerate(ids):
                    associations[t] = tags[i]
                    
    return associations


def main():

    if (len(sys.argv) != 5):
        print("Usage: gettags [input] [tag file] [id file] [output file]")
        sys.exit()

    inpt = sys.argv[1] # That is, the output of the Naive-Bayes program
    tags = sys.argv[2] # Csv containing the tags, as text. Generated from prepro.py
    ids = sys.argv[3]  # Same as above but the one with numeric IDs
    output_file = sys.argv[4] 

    associations = associate_id_to_tag(ids, tags)

    with open(inpt, 'r') as csvfile:
        with open(output_file, 'w') as out:

            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvfile:
                output_row = []
                binary_array = re.sub('[^0-1,]+', '', row).split(',')
                for i, binary_val in enumerate(binary_array):
                    if binary_val == '1':
                        output_row.append(associations[str(i)])

                print(str(output_row), file=out)





if __name__ == "__main__":
    main()
