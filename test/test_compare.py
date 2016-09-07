import sys
import argparse

import numpy as np

def similar_reals(ref, test, tol, colnames=None):
    '''
    Compare vector test against vector ref with a tolerance of tol (common scalar or vector)
    '''

    ref = np.array(ref)
    test = np.array(test)
    tol = np.atleast_1d(tol)

    if len(ref)!=len(test):
        raise ValueError('Cannot compare arrays of different size')

    if len(tol)==1:
        tol = tol*np.ones(len(ref))


    row_format ="{:>10}" * len(ref)

    if colnames:
        print('           '+row_format.format(*colnames))
    print('Reference: '+row_format.format(*ref))
    print('Test:      '+row_format.format(*test))
    print('Diff:      '+row_format.format(*(ref-test)))
    print('Tolerance: '+row_format.format(*tol))
    if any(abs(ref-test)>tol):
        print('FAILED')
        return False

    return True

if __name__ == '__main__':
    argpar = argparse.ArgumentParser()
    argpar.add_argument("--ref", nargs='+', type=float, default=None, help="Reference values.")
    argpar.add_argument("--test", nargs='+', type=float, help="Values to test against the references.")
    argpar.add_argument("--tol", nargs='+', type=float, default=0.1, help="Accepted tolerance (if a single value is provided, it is used for all compared pairs")
    argpar.add_argument("--colnames", nargs='+', default=None, help="Names for each column")
    args = argpar.parse_args()

    if not similar_reals(args.ref, args.test, args.tol, args.colnames):
        sys.exit(1)
