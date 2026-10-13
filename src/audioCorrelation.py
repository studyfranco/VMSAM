'''
Created on 1 May 2022

@author: francois
'''

import tools
import gc
import json
import time
from threading import RLock
import sys

'''
1 May 2022
Based on https://raw.githubusercontent.com/kdave/audio-compare/master/correlation.py 
'''
import numpy

# Step size (in points) of the cross correlation.
step = 1
# Minimum number of points that must overlap in a cross correlation.
min_overlap = 32

def calculate_fingerprints(filename,length=1):
    '''Return the raw Chromaprint fingerprint (fpcalc -raw) of the first `length` seconds.'''
    cmd = [tools.software["fpcalc"], '-raw', '-length', str(length), filename]
    stdout, stderror, exitCode = tools.launch_cmdExt_no_test(cmd)
    if exitCode != 0:
        from re import search,MULTILINE
        if exitCode == 3 and search(r'.*ERROR. Error decoding audio frame .End of file.',stderror.decode("utf-8"), MULTILINE) != None:
            pass
        else:
            raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    
    fpcalc_out = stdout.decode("utf-8").strip().replace('\\n', '').replace("'", "")
    fingerprint_index = fpcalc_out.find('FINGERPRINT=') + 12
    fingerprints = list(map(int, fpcalc_out[fingerprint_index:].split(',')))
    
    return fingerprints
  
def correlation(listx, listy):
    '''Return the bitwise similarity (0..1) of two fingerprint lists, truncated to equal length.'''
    if len(listx) == 0 or len(listy) == 0:
        raise Exception('Empty lists cannot be correlated.')
    if len(listx) > len(listy):
        listx = listx[:len(listy)]
    elif len(listx) < len(listy):
        listy = listy[:len(listx)]
    
    covariance = 0
    for i in range(len(listx)):
        covariance += 32 - bin(listx[i] ^ listy[i]).count("1")
    covariance = covariance / float(len(listx))
    
    return covariance/32
  
def cross_correlation(listx, listy, offset):
    '''Return the correlation at a shift of `offset` points, or None if the overlap is too small.'''
    if offset > 0:
        listx = listx[offset:]
        listy = listy[:len(listx)]
    elif offset < 0:
        offset = -offset
        listy = listy[offset:]
        listx = listx[:len(listy)]
    if min(len(listx), len(listy)) < min_overlap:
        return 
    return correlation(listx, listy)
  
def compare(listx, listy, span, step):
    '''Return the cross correlations for every offset from -span to span.'''
    if span > min(len(listx), len(listy)):
        raise Exception('span >= sample size: %i >= %i\n'
                        % (span, min(len(listx), len(listy)))
                        + 'Reduce span, reduce crop or increase sample_time.')
    corr_xy = []
    for offset in numpy.arange(-span, span + 1, step):
        corr_xy.append(cross_correlation(listx, listy, offset))
    return corr_xy
  
def max_index(listx):
    max_index = 0
    max_value = listx[0]
    for i, value in enumerate(listx):
        if value > max_value:
            max_value = value
            max_index = i
    return max_index
  
def get_max_corr(corr, source, target, span, sizePoint):
    '''Return (best correlation, offset in points, offset in ms) from a correlation list.'''
    max_corr_index = max_index(corr)
    max_corr_offset = -span + max_corr_index * step
    return corr[max_corr_index],max_corr_offset,-max_corr_offset*sizePoint

def correlate(source, target, lengthFile):
    '''Correlate the fingerprints of two audio files and return get_max_corr's result.'''
    fingerprint_source = calculate_fingerprints(source,length=lengthFile)
    fingerprint_target = calculate_fingerprints(target,length=lengthFile)
    
    if len(fingerprint_source) != len(fingerprint_target):
        if len(fingerprint_target) < len(fingerprint_source):
            span = len(fingerprint_target) - min_overlap
        else:
            span = len(fingerprint_source) - min_overlap
    else:
        span = len(fingerprint_target) - min_overlap
    corr = compare(fingerprint_source, fingerprint_target, span, step)
    return get_max_corr(corr, source, target, span, int(lengthFile/len(fingerprint_source)*1000))

'''
End Copy
'''

def test_calcul_can_be(filename,length):
    '''Return True when a fingerprint can be computed for the file.'''
    try:
        calculate_fingerprints(filename,length)
        return True
    except:
        import traceback
        traceback.print_exc()
        return False

'''
4 Jan 2023
Based on https://github.com/rpuntaie/syncstart/blob/main/syncstart.py
'''
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from scipy.io import wavfile
from os import path,remove

ax = None
normalize = False
denoise = False
lowpass = 0

def get_files_metrics(outfile):
    '''Return (sample rate, first-channel samples) of a WAV file.'''
    r,s = wavfile.read(outfile)
    if len(s.shape)>1: #stereo
        s = s[:,0]
    return r,s

def generate_norm_cmd(in_file,out_file):
    '''Return the ffmpeg command that loudness-normalises a file to 16-bit PCM.'''
    return [tools.software["ffmpeg"], "-y", "-threads", str(2), "-nostdin", "-i", in_file, "-filter_complex",
            "[0:0]loudnorm=i=-23.0:lra=7.0:tp=-2.0:offset=4.45:linear=true:print_format=json[norm0]",
            "-map_metadata", "0", "-map_metadata:s:a:0", "0:s:a:0", "-map_chapters", "0", "-c:v", "copy", "-map", "[norm0]",
            "-c:a:0", "pcm_s16le", "-c:s", "copy", out_file]

def read_normalized(in1,in2):
    '''Read two WAV files at a common sample rate, normalising (then denoising) if rates differ.

    Returns:
        (sample rate, samples1, samples2).
    '''
    from video import ffmpeg_pool_audio_convert,wait_end_big_job

    r1,s1 = get_files_metrics(in1)
    r2,s2 = get_files_metrics(in2)
    if r1 != r2:
        base_namme_in1 = path.splitext(path.basename(in1))[0]
        base_namme_in2 = path.splitext(path.basename(in2))[0]
        wait_end_big_job()
        out_in1_norm = path.join(tools.tmpFolder,base_namme_in1+"_norm.wav")
        job_in1 = ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, (generate_norm_cmd(in1,out_in1_norm),) )
        out_in2_norm = path.join(tools.tmpFolder,base_namme_in2+"_norm.wav")
        job_in2 = ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, (generate_norm_cmd(in2,out_in2_norm),) )
            
        job_in1.get()
        r1,s1 = get_files_metrics(out_in1_norm)
        job_in2.get()
        r2,s2 = get_files_metrics(out_in2_norm)
        if r1 != r2:
            wait_end_big_job()
            out_in1_norm_denoise = path.join(tools.tmpFolder,base_namme_in1+"_norm_denoise.wav")
            job_in1 = ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, ([tools.software["ffmpeg"], "-y", "-threads", str(2), "-i", out_in1_norm, "-af", "'afftdn=nf=-25'", out_in1_norm_denoise],) )
            out_in2_norm_denoise = path.join(tools.tmpFolder,base_namme_in2+"_norm_denoise.wav")
            job_in2 = ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, ([tools.software["ffmpeg"], "-y", "-threads", str(2), "-i", out_in2_norm, "-af", "'afftdn=nf=-25'", out_in2_norm_denoise],) )
            
            job_in1.get()
            r1,s1 = get_files_metrics(out_in1_norm_denoise)
            remove(out_in1_norm)
            remove(out_in1_norm_denoise)
            job_in2.get()
            r2,s2 = get_files_metrics(out_in2_norm_denoise)
            remove(out_in2_norm)
            remove(out_in2_norm_denoise)
        else:
            remove(out_in1_norm)
            remove(out_in2_norm)

    assert r1 == r2, "not same sample rate"
    fs = r1
    return fs,s1,s2

def corrabs(s1,s2):
    '''Return the FFT cross correlation magnitude of two signals and its peak index.

    Returns:
        (len1, len2, padded size, peak index, |correlation|).
    '''
    ls1 = len(s1)
    ls2 = len(s2)
    padsize = ls1+ls2+1
    padsize = 2**(int(np.log(padsize)/np.log(2))+1)
    s1pad = np.zeros(padsize)
    s1pad[:ls1] = s1
    s2pad = np.zeros(padsize)
    s2pad[:ls2] = s2
    corr = fft.ifft(fft.fft(s1pad)*np.conj(fft.fft(s2pad)))
    ca = np.absolute(corr)
    xmax = np.argmax(ca)
    return ls1,ls2,padsize,xmax,ca

"""
Visualisation
"""
def fig1(title=None):
    fig = plt.figure(1)
    plt.margins(0, 0.1)
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    plt.title(title or 'Signal')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Amplitude')
    axs = fig.get_axes()
    global ax
    ax = axs[0]

def show1(fs, s, color=None, title=None, v=None):
    if not color: fig1(title)
    if ax and v: ax.axvline(x=v,color='green')
    plt.plot(np.arange(len(s))/fs, s, color or 'black')
    if not color: plt.show()

def show2(fs,s1,s2,title=None):
    fig1(title)
    show1(fs,s1,'blue')
    show1(fs,s2,'red')
    plt.show()

lock_fallback = RLock()
def second_correlation(in1,in2):
    '''Return (file to cut, offset in seconds) aligning two audio files.

    Uses the audio_sync tool, falling back to an in-process FFT correlation.
    '''
    try:
        begin = time.time()
        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["audio_sync"],in1,in2],3,28800)
        data = json.loads(stdout.decode("utf-8").strip())
        file = data['file']
        offset = data['offset_seconds']
        if tools.dev:
            tools.logs.append(f"\t\tSecond correlation in new function took {time.time()-begin:.2f} seconds\n\t\tand we obtain: {data}\n")
    except Exception as e:
        # Fall back to the in-process FFT correlation when audio_sync fails.
        sys.stderr.write(f"\t\taudio_sync not working: {e}\n")
        tools.logs.append(f"\t\taudio_sync not working: {e}\n")
        
        with lock_fallback:
            begin = time.time()
            fs,s1,s2 = read_normalized(in1,in2)
            ls1,ls2,padsize,xmax,ca = corrabs(s1,s2)
            ls1 = None
            ls2 = None
            ca = None
            s1 = None
            s2 = None
            if xmax > padsize // 2:
                file,offset = in2,(padsize-xmax)/fs
            else:
                file,offset = in1,xmax/fs
            padsize = None
            xmax = None
            fs = None
            gc.collect()
        
        if tools.dev:
            tools.logs.append(f"\t\tSecond correlation in old function took {time.time()-begin:.2f} seconds\n\t\tand we obtain: {file} in offset {offset}\n")
    
    return file,offset

'''
End Copy
'''