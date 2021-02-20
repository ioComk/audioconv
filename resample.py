#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:51:18 2018

@author: makishima
"""

# -*- coding: utf-8 -*-
import sys
import os
import click
import logging
#from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

import numpy as np
# import h5py
import glob
# import re
# from scipy.io import wavfile
import soundfile as sf
from scipy.signal import resample, resample_poly


def resample_fs(signal, fs_input=44100, fs_output=8000, poly=True):
    """
    scipy.signal.resample[_poly] を用い，サンプリング周波数指定でリサンプリング
    """
    len_input = signal.size
    len_output = (len_input * fs_output) // fs_input
    if poly:
        res = resample_poly(signal, len_output, len_input)
    else:
        res = resample(signal, num=len_output)
    # return res.astype(np.int16)
    return res


@click.command()
@click.option('--input_folderpath', '-i', type=click.Path(exists=True),
            default='data/raw/DSD100/Sources',
            help='Folder path of source wav files.')
@click.option('--output_folderpath', '-o', type=click.Path(),
            default='data/interim/8k',
            help='Folder path of output wav files.')
@click.option('--sampling_freq', '-fs', type=int, default=8000)
@click.option('--poly', '-p', type=bool, default=True)
def main(input_folderpath, output_folderpath, sampling_freq, poly):
    """ Runs data processing scripts to turn raw data from (../raw) into
        resampled data ready to be trained and analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making resampled data set from raw data')

    # import pudb; pudb.set_trace()

    # maker = DatasetMaker(c=4)
    files = glob.glob(os.path.join(input_folderpath, '**/*.wav'), recursive=True)
    logger.info(str(len(files)) + ' files detected.')
    for f in tqdm(files, leave=False):
        f_relpath = os.path.relpath(f, start=input_folderpath)
        signal_raw, fs_raw = sf.read(f)
        signal_resampled = resample_fs(
            signal_raw, fs_raw, sampling_freq, poly)
        outpath = os.path.join(output_folderpath, f_relpath)
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        sf.write(outpath, signal_resampled, sampling_freq)
        logger.info('Created wav file: {} -> {}'.format(f, outpath))
        # import pudb; pudb.set_trace()


if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()