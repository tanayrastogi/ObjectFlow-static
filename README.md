# ObjectFlow-static
Tracking objects in video taken from static camera.
There are two python scripts here,
- *polygon_points.py* is for getting the points for the polygon within the space in the video. This is to limit the object flow detection to a fixed space. 
- *objectflow-static.py* is to get the flow of objects in the input video. 

Uses the object detection from [Github](https://github.com/tanayrastogi/ObjectDetection) and a modified centroied tracking from [Ardian Pyimagesearch](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)

### Usage
- **Polygon**
    python polygon_points.py --input [path to video]
- **Object Flow**
    python objectflow-static.py --input [path to video] --polygon 0 356 607 83 1279 159 1216 617

You will need the model files for the object detection and is to be stored in "ObjectDetection/models/"
