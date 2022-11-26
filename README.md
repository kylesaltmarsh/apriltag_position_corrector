# apriltag_marker_locator

Generates pose for Apriltags in csv format for SLAM landmarking

## Data Input

Marker location data must be input in csv form as seen in the TargetCoordinates.csv sample file

## Requirements

```bash
pip install -r Requirements.txt
```

## Usage

Python script takes exactly one system argument which is the csv file name without the extension

```bash
python AprilTag_Locator.py "TargetCoordinates"
```

## Output

The script will produce the following output

```bash
AprilTag_Locator.py
├── _TargetCoordinates
|   ├── 001.png
|   └── 00x.png
|   └── All_AprilTags.png
|   └── AprilTag_Poses.csv
```

## References

https://en.wikipedia.org/wiki/Quaternion

http://www.ros.org/reps/rep-0105.html