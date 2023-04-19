# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fname = 'input.txt'
    f2='out.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        s = [i[:-1].split(',') for i in f.readlines()]
        print(s)

        with open(f2, 'r+', encoding='utf-8') as f:
            s2 = [i[:-1].split(',') for i in f.readlines()]
            print(s2)
    input = np.array(s,dtype=float)
    out = np.array(s2,dtype=float)

    np.savez_compressed("yuang.npz",input=input,out=out)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
