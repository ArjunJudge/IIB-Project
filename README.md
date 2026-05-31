# IIB-Project

## Requirements

### Visualization Core
`vtk`

### Trame Web Framework
`trame`
`trame-vuetify`
`trame-vtk`
`trame-plotly`

### Data Processing
`numpy`
`PyYAML`

## Installation:
`pip install vtk trame trame-vuetify trame-vtk trame-plotly numpy PyYAML`

## How to Run:
`local_python_path path_to_script.py path_to_config_file.yaml`


## Merged.py

The 2 previous scripts - vtklocalview.py and vtkremote_view.py - have been merged to get a best of both.

This script uses VTKLocalView. Clicking vessel surface (with left mouse button) performs the BFS + Slice calculations. Clicking centerlines in plotting mode fills a scatter plot showing pressure distribution along clicked branch. Centerline file needs to provided in input file for slicing and plotting to work.

Capabilities:
* Mesh rendering (vtu/vtk)
* Colouring by array
* Streamline rendering
* Image rendering (point cloud)
* Animating (to see moving boundary). This requires an array to contour by at each time step e.g. signed distance field. The format must by 'array_name_tN' where N is an integer. e.g. 'sdf_t0', 'sdf_t1', ...
* Slicing - must provide path to a vtp containing skeleton of aorta - CURRENTLY WORKING ON THIS
* Point picking (click scrollwheel). If clicking a face of aorta, face is shaded and volume flow rate calculated as well as average quantities. If clicking along aorta boundary, and slicing is enabled, aorta is sliced and shown inside aorta
