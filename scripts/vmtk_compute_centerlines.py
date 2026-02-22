import vmtk
import vtk
import numpy as np
from vmtk import vmtkscripts

# vmtk centerlines
def compute_centerlines(surface):
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    centerlines.AppendEndPoints = 1
    centerlines.Execute()
    attributes = vmtkscripts.vmtkCenterlineAttributes()
    attributes.Centerlines = centerlines.Centerlines
    attributes.Execute()
    return attributes.Centerlines

surface = "../path_to_vtp_file.vtp"

surface_reader = vmtkscripts.vmtkSurfaceReader()
surface_reader.InputFileName = surface
surface_reader.Execute()
surface = surface_reader.Surface

centerlines = compute_centerlines(surface)
# Save centerlines to file
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("output_centerlines.vtp")
writer.SetInputData(centerlines)
writer.Write()

print("Centerlines computed and saved to file.")