import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(sys.argv[0])))

import numpy as np
from pathlib import Path
import soundfile as sf
from resample import resample_fs
import logging
from scipy.signal import convolve
import librosa

pardir = str(Path(__file__).absolute().parents[2])
sys.path.append(pardir)

logger = logging.getLogger(__name__)

def create_srcimg(name_drysrc, insts, angle_src, pos_mic,
                fs_src=8000, fs_imp=48000,
                kind='RWCP_E2A',
                ):
    '''
    [input]
        name_drysrc : path to dry source directory
        insts       : name list of dry source
        pos_mic     : list of mic (ch.)
        angle_src   : list of pos of src (src.)

    [output]
        src image   : (n × m × time)(n=0:vocal)
    '''

    drysrc, _ = load_drysrc(name_drysrc, insts) # drysrc: n ×　time (n=0:vocal)

    imps = load_impulse(angle_src, pos_mic, kind, fs_src, fs_raw=fs_imp) #angle x pos x time

    list_all = list()
    for n in range(np.shape(drysrc)[0]):
        list_pos = list()
        for m in range(len(pos_mic)):
            seq = convolve(drysrc[n,:], imps[n, m, :])
            list_pos.append(seq)
        list_all.append(list_pos) #n × m × time

    return np.array(list_all)


# create_srcimg の下請け
def load_drysrc(name_drysrc, insts):
    assert isinstance(insts, list), 'insts must be list.'
    filename = '{name_drysrc}/{inst}.wav'
    filepath = str(Path(filename))
    list_src = list()
    for inst in insts:
        fname = filepath.format(name_drysrc=name_drysrc, inst=inst)
        sig, fs = sf.read(fname)
        sig_avg = np.mean(sig, axis=1)  # channel ごとの平均（ドライだからね）
        list_src.append(sig_avg)
    return np.array(list_src), fs


# create_srcimg の下請け
def load_impulse(angle_src, pos_mic, kind,
                fs_out, fs_raw=48000,
                folderpath='impulse'):
    '''
    output: impulse response (src. * ch. * time)
    '''
    if not isinstance(pos_mic, list):
        pos_mic = [pos_mic]
    if not isinstance(angle_src, list):
        angle_src = [angle_src]
    filename = '{kind}/imp{angle_src:03}.{pos_mic:02}'
    filepath = str(Path(folderpath) / Path(filename))

    list_all = list()
    for ang in angle_src:
        list_pos = list()
        for pos in pos_mic:
            fname = filepath.format(pos_mic=pos, angle_src=ang, kind=kind)
            with open(fname, 'rb') as f:
                seq = np.fromfile(f, np.float32, -1)
            seq_res = resample_fs(seq, fs_input=fs_raw, fs_output=fs_out)
            list_pos.append(seq_res)
        list_all.append(list_pos)

    return np.array(list_all)


# save
def savesrcimg(name_drysrc, inst, angle_src, pos_mic,
                savestyle='wav',
                output_folder='output',
                fs_src=8000, fs_imp=48000, fs_output=16000,
                kind='E2A',):
    '''
    [input]
        name_drysrc : path to dry source directory
        inst        : name list of dry source 
        pos_mic     : list of mic position(ch.)
        angle_src   : list of source position (src.)

    [output]
        src image   : vocals for left mic, other for right mic
    '''

    #img : src x mic x time
    img = create_srcimg(name_drysrc, inst, angle_src, pos_mic,
                        fs_src=fs_src, fs_imp=fs_imp, kind=kind)

    folder = Path(output_folder) / Path(name_drysrc+'_'+kind)

    if not folder.exists():
        folder.mkdir(parents=True)

    # Save format
    if savestyle == 'wav':
        filename = 'srcimg_s{n}_m{m}.wav'
        filepath = str(folder / Path(filename))
        for n in range(len(angle_src)):
            for m in range(len(pos_mic)):
                fname = filepath.format(n=n, m=m)
                res = librosa.resample(img[n, m, :], fs_src, fs_output)
                sf.write(fname, res.T, fs_output)
        logger.info('Successfully saved: ' +
                    filepath.format(n='*', m='*'))

    elif savestyle == 'stereowav':
        assert len(pos_mic) == 2, "number of mics has to be 2 in 'stereowav'."
        filename = '{inst}_{kind}_pos{ang}_mic{pos}.wav'
        filepath = str(folder / Path(filename))
        for n in range(len(inst)):
            fname = filepath.format(inst=inst[n],kind=kind,ang=angle_src[n],
                                        pos=str(pos_mic[0])+str(pos_mic[1]))
            res = librosa.resample(img[n, :, :], fs_src, fs_output)
            sf.write(fname, res.T, fs_output)

    # リサンプルして書き出しは未実装
    elif savestyle == 'triwav':
        assert len(pos_mic) == 3, "number of mics has to be 3 in 'triwav'."
        filename = '{kind}_{inst}_pos{ang}_mic{pos}.wav'
        filepath = str(folder / Path(filename))
        for n in range(len(inst)):
            fname = filepath.format(inst=inst[n],kind=kind,ang=angle_src[0],
                                        pos=str(pos_mic[0])+str(pos_mic[1])+str(pos_mic[2]))
            sf.write(fname, img[n, :, :].T, fs_src)

    elif savestyle == 'ndarray':
        filename = 'srcimg.npy'
        fname = str(folder / Path(filename))
        np.save(fname, img)
    else:
        raise ValueError("savestyle must be ['wav', 'stereowav', 'triwav' 'ndarray'].")

if __name__ == '__main__':

    fs_output = 16000

    mic_angle = [
        50,
        110,
    ]

    mic_pos = [
        21,
        23,
    ]

    # Song ID 1 ----------------------------------------------
    folder_path = 'drySources/music/bearlin-roads'
    src = [
        'bearlin-roads__snip_85_99__vocals',
        'bearlin-roads__snip_85_99__drums'
    ]
    # --------------------------------------------------------

    # Song ID 2 ----------------------------------------------
    # folder_path = 'drySources/music/another_dreamer-the_ones_we_love'
    # src = [
    #     'another_dreamer-the_ones_we_love__snip_69_94__vocals',
    #     'another_dreamer-the_ones_we_love__snip_69_94__drums',
    # ]
    # --------------------------------------------------------

    # Song ID 3 ----------------------------------------------
    # folder_path = 'drySources/music/fort_minor-remember_the_name'
    # src = [
    #     'fort_minor-remember_the_name__snip_54_78__vocals',
    #     'fort_minor-remember_the_name__snip_54_78__drums',
    # ]
    # --------------------------------------------------------

    # Song ID 4 ----------------------------------------------
    # folder_path = 'drySources/music/ultimate_nz_tour'
    # src = [
    #     'ultimate_nz_tour__snip_43_61__vocals',
    #     'ultimate_nz_tour__snip_43_61__drums',
    # ]
    # --------------------------------------------------------
    

    savesrcimg(folder_path, src, mic_angle, mic_pos,
            savestyle='stereowav', fs_src=44100, fs_imp=48000, fs_output=fs_output,
            kind='E2A')

    print('finished!')
