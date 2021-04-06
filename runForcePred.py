#!/usr/bin/env python

__author__ = 'Jas Kalayan'
__credits__ = ['Jas Kalayan', 'Neil Burton']
__license__ = 'GPL'
__maintainer__ = 'Jas Kalayan'
__email__ = 'jkalayan@gmail.com'
__status__ = 'Development'

import sys
sys.path.append('..') #package is in parent directory
import ForcePred
from ForcePred import OPTParser
from datetime import datetime
import argparse

def force_pred(input_files='input_files'):


    startTime = datetime.now()

    print(startTime)


    molecule = OPTParser(input_files)
    #print(molecule.get_3D_array(molecule.forces))
    #print(molecule.__dict__)
    print(molecule)
    print(molecule.energies)



    print(datetime.now() - startTime)


def main():

    try:
        usage = 'runForcePred.py [-h]'
        parser = argparse.ArgumentParser(description='Program for reading '\
                'in molecule forces, coordinates and energies for '\
                'force prediction.', usage=usage, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group('Options')
        group = parser.add_argument('-i', '--input_files', nargs='+', 
                metavar='file', help='name of file/s containing forces '\
                'coordinates and energies.')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    force_pred(input_files=op.input_files)

if __name__ == '__main__':
    main()


