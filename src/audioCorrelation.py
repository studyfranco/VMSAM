'''
Created on 1 May 2022

@author: francois
'''

import tools
import gc
import json
import time

import sys

'''
1 May 2022
Based on https://raw.githubusercontent.com/kdave/audio-compare/master/correlation.py 
'''
# correlation.py
import numpy

# seconds to sample audio file for
#sample_time = 500
# number of points to scan cross correlation over
#span = 300
# step size (in points) of cross correlation
step = 1
# minimum number of points that must overlap in cross correlation
# exception is raised if this cannot be met
min_overlap = 32
# report match when cross correlation has a peak exceeding threshold
#threshold = 0.5

# calculate fingerprint
# Generate file.mp3.fpcalc by "fpcalc -raw -length 500 file.mp3"
def calculate_fingerprints(filename,length=1):
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
    # convert fingerprint to list of integers
    fingerprints = list(map(int, fpcalc_out[fingerprint_index:].split(',')))
    
    return fingerprints
  
# returns correlation between lists
def correlation(listx, listy):
    if len(listx) == 0 or len(listy) == 0:
        # Error checking in main program should prevent us from ever being
        # able to get here.
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
  
# return cross correlation, with listy offset from listx
def cross_correlation(listx, listy, offset):
    if offset > 0:
        listx = listx[offset:]
        listy = listy[:len(listx)]
    elif offset < 0:
        offset = -offset
        listy = listy[offset:]
        listx = listx[:len(listy)]
    if min(len(listx), len(listy)) < min_overlap:
        # Error checking in main program should prevent us from ever being
        # able to get here.
        return 
    #raise Exception('Overlap too small: %i' % min(len(listx), len(listy)))
    return correlation(listx, listy)
  
# cross correlate listx and listy with offsets from -span to span
def compare(listx, listy, span, step):
    if span > min(len(listx), len(listy)):
        # Error checking in main program should prevent us from ever being
        # able to get here.
        raise Exception('span >= sample size: %i >= %i\n'
                        % (span, min(len(listx), len(listy)))
                        + 'Reduce span, reduce crop or increase sample_time.')
    corr_xy = []
    for offset in numpy.arange(-span, span + 1, step):
        corr_xy.append(cross_correlation(listx, listy, offset))
    return corr_xy
  
# return index of maximum value in list
def max_index(listx):
    max_index = 0
    max_value = listx[0]
    for i, value in enumerate(listx):
        if value > max_value:
            max_value = value
            max_index = i
    return max_index
  
def get_max_corr(corr, source, target, span, sizePoint):
    max_corr_index = max_index(corr)
    max_corr_offset = -span + max_corr_index * step
    #print("max_corr_index = ", max_corr_index, "max_corr_offset = ", max_corr_offset)
    # report matches
    #print("File A: %s" % (source))
    #print("File B: %s" % (target))
    #print('Match with correlation of %.2f%% at offset %i'
    #     % (corr[max_corr_index] * 100.0, max_corr_offset))
    return corr[max_corr_index],max_corr_offset,-max_corr_offset*sizePoint

def correlate(source, target, lengthFile):
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

#ffmpeglow = [tools.software["ffmpeg"], "-y", "-threads", str(tools.core_to_use), "-i", '"{}"', "-af", f"'lowpass=f={lowpass}'", '"{}"']

def get_files_metrics(outfile):
    r,s = wavfile.read(outfile)
    if len(s.shape)>1: #stereo
        s = s[:,0]
    return r,s

def generate_norm_cmd(in_file,out_file):
    return [tools.software["ffmpeg"], "-y", "-threads", str(2), "-nostdin", "-i", in_file, "-filter_complex",
            "[0:0]loudnorm=i=-23.0:lra=7.0:tp=-2.0:offset=4.45:linear=true:print_format=json[norm0]",
            "-map_metadata", "0", "-map_metadata:s:a:0", "0:s:a:0", "-map_chapters", "0", "-c:v", "copy", "-map", "[norm0]",
            "-c:a:0", "pcm_s16le", "-c:s", "copy", out_file]

def read_normalized(in1,in2):
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

def second_correlation(in1,in2):
    begin = time.time()
    fs,s1,s2 = read_normalized(in1,in2)
    ls1,ls2,padsize,xmax,ca = corrabs(s1,s2)
    ls1 = None
    ls2 = None
    ca = None
    s1 = None
    s2 = None
    # if show: show1(fs,ca,title='Correlation',v=xmax/fs) Change if we want reports
    #sync_text = """
    #==============================================================================
    #%s needs 'ffmpeg -ss %s' cut to get in sync
    #==============================================================================
    #"""
    if xmax > padsize // 2:
        # if show: show2(fs,s1,s2[padsize-xmax:],title='1st=blue;2nd=red=cut(%s;%s)'%(in1,in2))
        file,offset = in2,(padsize-xmax)/fs
    else:
        # if show: show2(fs,s1[xmax:],s2,title='1st=blue=cut;2nd=red (%s;%s)'%(in1,in2))
        file,offset = in1,xmax/fs
    #print(sync_text%(file,offset))
    padsize = None
    xmax = None
    fs = None
    gc.collect()
    
    sys.stderr.write(f"\t\tSecond correlation in old function took {time.time()-begin:.2f} seconds\n\t\tand we obtain: {file} in offset {offset}\n")
    try:
        begin = time.time()
        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["audio_sync"],in1,in2],5,30)
        data = json.loads(stdout.decode("utf-8").strip())
        sys.stderr.write(f"\t\tSecond correlation in new function took {time.time()-begin:.2f} seconds\n\t\tand we obtain: {data}\n")
    except Exception as e:
        # If audio_sync is not installed, we return the file and offset
        sys.stderr.write(f"\t\taudio_sync not working: {e}\n")
    
    return file,offset

'''
End Copy
'''