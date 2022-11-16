'''
Created on 1 May 2022

@author: francois
'''

import tools

'''
1 May 2022
Based on https://raw.githubusercontent.com/kdave/audio-compare/master/correlation.py 
'''
# correlation.py
import subprocess
import numpy

# seconds to sample audio file for
#sample_time = 500
# number of points to scan cross correlation over
#span = 300
# step size (in points) of cross correlation
step = 1
# minimum number of points that must overlap in cross correlation
# exception is raised if this cannot be met
min_overlap = 30
# report match when cross correlation has a peak exceeding threshold
#threshold = 0.5

# calculate fingerprint
# Generate file.mp3.fpcalc by "fpcalc -raw -length 500 file.mp3"
def calculate_fingerprints(filename,length=0):
    fpcalc_out = str(subprocess.check_output([tools.software["fpcalc"], '-raw', '-length', str(length), filename])).strip().replace('\\n', '').replace("'", "") # '-length', str(sample_time),

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

def test_calcul_can_be(filename):
    try:
        calculate_fingerprints(filename)
        return True
    except Exception as e:
        print(e)
        return False