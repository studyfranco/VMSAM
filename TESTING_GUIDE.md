# VMSAM Enhanced Testing Guide

## Testing Strategy

### 1. Unit Tests

#### ML Scene Detection
```bash
# Test scene detection on sample video
python3 -c "
from ml_scene_detection import SceneDetector
detector = SceneDetector()
scenes = detector.detect_scenes_from_video('/path/to/test.mkv', duration=60)
print(f'Detected {len(scenes)} scenes')
"
```

#### Delay Uncertainty Analysis
```bash
# Test uncertainty analyzer
python3 -c "
from delay_uncertainty_analyzer import DelayUncertaintyAnalyzer
analyzer = DelayUncertaintyAnalyzer()
print('Uncertainty analyzer ready')
"
```

### 2. Integration Tests

#### Enhanced Merge Workflow
```bash
# Test full enhanced merge
python3 main.py --input /config/input --output /config/output --sync
```

#### Performance Comparison
```bash
# Compare original vs enhanced
export VMSAM_ML_SCENE_DETECTION=false  # Original mode
time python3 main.py --input test_videos --output output1

export VMSAM_ML_SCENE_DETECTION=true   # Enhanced mode  
time python3 main.py --input test_videos --output output2
```

### 3. Docker Environment Tests

#### Build Enhanced Container
```bash
docker build -t vmsam-enhanced .
```

#### Run Test Suite
```bash
docker run -v /path/to/test/videos:/config/input vmsam-enhanced python3 test_vmsam_enhanced.py
```

### 4. Edge Case Testing

#### Low Quality Videos
- Test with heavily compressed videos
- Test with different framerates
- Test with audio sync issues

#### Large Files
- Test memory usage with large video files
- Monitor CPU usage during ML processing
- Validate temp file cleanup

#### Error Scenarios
- Test with corrupted video files
- Test with missing audio tracks
- Test ML component failures

### 5. Performance Benchmarks

#### Metrics to Monitor
- Processing time per video minute
- Memory usage peak and average
- Accuracy of delay detection
- Success rate of merges

#### Benchmark Commands
```bash
# Memory usage monitoring
/usr/bin/time -v python3 main.py --input test --output out

# CPU usage monitoring
top -p $(pgrep -f main.py) -b -n 1

# Accuracy testing
# (requires manual validation of output sync quality)
```

### 6. Regression Testing

#### Ensure Compatibility
- Test original merge functionality still works
- Verify all configuration options preserved
- Check Docker environment variables
- Validate output file formats

## Expected Results

### Success Criteria
- All unit tests pass
- Integration tests complete successfully
- Memory usage remains reasonable (<2GB for typical videos)
- Processing time increase <20%
- Delay accuracy improves by >20%

### Warning Signs
- Memory usage spikes >4GB
- Processing time increases >50%
- ML components fail frequently
- Accuracy degrades vs original

## Troubleshooting

### Common Issues

1. **ML Components Not Loading**
   ```bash
   pip install opencv-python scenedetect[opencv]
   ```

2. **High Memory Usage**
   ```bash
   export VMSAM_ANALYSIS_WINDOW=15  # Reduce analysis window
   ```

3. **Scene Detection Too Slow**
   ```bash
   export VMSAM_ML_SCENE_DETECTION=false  # Disable ML
   ```

### Debug Mode

```bash
export dev=true  # Enable detailed logging
export VMSAM_DEBUG_SCENES=true  # Scene detection debug
```

## Validation Checklist

- [ ] Dependencies install correctly
- [ ] ML components initialize without errors  
- [ ] Scene detection works on test videos
- [ ] Delay uncertainty analysis functions
- [ ] Enhanced merge produces valid output
- [ ] Performance remains acceptable
- [ ] Memory usage stays reasonable
- [ ] Error handling works correctly
- [ ] Docker container builds successfully
- [ ] Backward compatibility maintained

## Reporting Issues

When reporting issues, include:
- Test video characteristics (resolution, duration, codec)
- System specifications (CPU, RAM)
- Docker/environment details
- Full error logs with debug mode enabled
- Performance measurements (time, memory)

## Next Steps

After successful testing:
1. Tag stable release
2. Update production deployment
3. Monitor performance metrics
4. Gather user feedback
5. Plan future enhancements