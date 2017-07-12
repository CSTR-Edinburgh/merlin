import os
import sys
import numpy as np

if __name__ == "__main__":

    if len(sys.argv)!=3:
        print('Usage: python src/prepare_txt_done_data_file.py <txt_dir> <txt.done.data>\n')
        sys.exit(0)

    txt_dir  = sys.argv[1]
    out_file = sys.argv[2]

    out_f = open(out_file,'w')

    for txtfile in os.listdir(txt_dir):
        if txtfile is not None:
            file_id = os.path.basename(txtfile).split(".")[0]
            txtfile = os.path.join(txt_dir, txtfile)
            with open(txtfile, 'r') as myfile:
                data = myfile.read().replace('\n', '')
            data = data.replace('"', '\\"')
            out_f.write("( "+file_id+" \" "+data+" \")\n")

    out_f.close()
