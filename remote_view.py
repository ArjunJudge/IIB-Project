import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, vtk as trame_vtk, html
import os
import sys
from collections import deque
import numpy as np
import builtins
from vtkmodules.util import numpy_support
import yaml

class VTUViewer:
    def __init__(self, state, controller, mesh_path, skeleton_path,
                 velocity_array_name, pressure_array_name,
                 clip_array_name, streamline_seed_pt_idx, find_gradients_of=None):  # #TODO: Add streamline tracing option
        """
        Initialize the VTU Viewer with VTK rendering setup.
        """

        """ INTERACTOR AND RENDERER SETUP """
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(True)
        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.AddObserver("MiddleButtonPressEvent", self.on_middle_click)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        """ OBJECT AND SURFACE ACTORS, FILTERS AND MAPPERS SETUP """
        self.objectGrid = vtk.vtkUnstructuredGrid()
        self.cellPicker = vtk.vtkCellPicker()
        self.cellPicker.SetTolerance(0.0005)  # small tolerance for picking
        self.objectSurface = vtk.vtkPolyData()
        self.surfaceMapper = vtk.vtkDataSetMapper()
        self.surfaceFilter = vtk.vtkDataSetSurfaceFilter()
        self.surfaceActor = vtk.vtkLODActor()
        self.surface_cell_areas_np = None
        self.surface_pt_norms_np = None
        self.surface_cell_norms_np = None
        # moved here to ensure we only add the surface actor ONCE - before
        # I was adding it multiple times causing duplicates in the renderer - slows down rendering
        self.cellPicker.AddPickList(self.surfaceActor)  # allows picking ONLY from points/cells on the surface actor
        self.cellPicker.PickFromListOn()
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        colour_tf = vtk.vtkColorTransferFunction()
        colour_tf.SetColorSpaceToRGB()  # ensures smooth RGB interpolation
        # Dark Blue → Light Blue → Soft Cyan → Light Orange → Orange → Red
        colour_tf.AddRGBPoint(0.00, 0.00, 0.00, 1.00)  # Dark Blue
        colour_tf.AddRGBPoint(0.15, 0.20, 0.45, 0.90)  # Blue
        colour_tf.AddRGBPoint(0.30, 0.55, 0.75, 1.00)  # Light Blue
        colour_tf.AddRGBPoint(0.45, 0.80, 0.85, 0.80)  # Very light blue (almost white)
        colour_tf.AddRGBPoint(0.60, 1.00, 0.85, 0.60)  # Light yellow-orange
        colour_tf.AddRGBPoint(0.75, 1.00, 0.65, 0.35)  # Orange
        colour_tf.AddRGBPoint(0.90, 1.00, 0.35, 0.20)  # Dark Orange
        colour_tf.AddRGBPoint(1.00, 1.00, 0.00, 0.00)  # Red
        # Convert to a LUT for mapper
        for i in range(256):
            rgb = list(colour_tf.GetColor(i / 255.0)) + [1.0]  # [R, G, B, A]
            self.lut.SetTableValue(i, *rgb)  # Set RGBA values. note the * unpacks the list
        # Attach to mapper
        self.surfaceMapper.SetLookupTable(self.lut)
        # setup scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)

        """ SLICE ACTOR, FILTERS AND MAPPERS SETUP """
        self.slice = vtk.vtkPolyData()
        self.sliceMapper = vtk.vtkDataSetMapper()
        self.sliceActor = vtk.vtkLODActor()
        self.sliceActor.SetMapper(self.sliceMapper)
        self.sliceActor.GetProperty().SetColor(1.0, 1.0, 0.0)  # yellow
        self.sliceActor.SetVisibility(False)
        self.slice_cell_areas_np = None
        self.slice_pt_norms_np = None
        self.slice_cell_norms_np = None
        self.sliceMapper.SetLookupTable(self.lut)
        self.plane = vtk.vtkPlane()
        self.cutter = vtk.vtkCutter()
        self.cutter.SetCutFunction(self.plane)
        self.connectivity = vtk.vtkConnectivityFilter()
        self.connectivity.SetInputConnection(self.cutter.GetOutputPort())
        self.connectivity.SetExtractionModeToClosestPointRegion()
        """ STREAMLINE ACTOR, FILTERS AND MAPPERS SETUP """
        self.streamTracer = vtk.vtkStreamTracer()
        self.seedSource = vtk.vtkPointSource()
        self.streamlineSeedPointIndex = streamline_seed_pt_idx
        self.streamlineActor = vtk.vtkLODActor()
        self.streamlineActor.SetVisibility(False)

        """ SKELETON DATASET SETUP """
        self.skeleton = vtk.vtkPolyData()
        
        """ EXTRA SETTINGS AND VARIABLES """
        self.state = state
        self.controller = controller
        self.array_names = {}
        self.doVelocityCalcs, self.doPressureCalcs = False, False
        self.velocityName = velocity_array_name
        self.pressureName = pressure_array_name
        self.clippingName = clip_array_name
        self.Q_running_total = 0  # accumulated volume flow rate through picked faces
        self.find_gradients_of = find_gradients_of

        """ RENDERING SETTINGS AND FILE READING """
        self.renderingObject, self.renderingStreamlines, self.doSlicing = False, False, False
        if not mesh_path:
            raise ValueError("No mesh file path provided")
        self.file_reader(mesh_path)
        self.renderingObject = True
        if streamline_seed_pt_idx is not None:
            self.add_streamlines()
            self.streamlineActor.SetVisibility(True)
            self.renderingStreamlines = True
        if skeleton_path:
            self.file_reader(skeleton_path)
            self.skeleton_points = numpy_support.vtk_to_numpy(self.skeleton.GetPoints().GetData())
            self.skeleton_to_pt_distances= np.zeros(self.skeleton.GetNumberOfPoints())
            self.doSlicing = True
        else:
            print("No skeleton file provided - slicing operations disabled.")
            pass
        print(f"Rendering Object: {self.renderingObject}, Rendering Streamlines: {self.renderingStreamlines}, Slicing Enabled: {self.doSlicing}")
        
        self.setup_grids_arrays_actors()

    def setup_arrays_in_dataset(self, dataset):
        """
        Setup necessary arrays in the dataset for visualisation and calculations.
        """
        # iterate through all arrays in point data
        point_data = dataset.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()
        for i in range(num_arrays):
            array_name = point_data.GetArrayName(i)
            # if array has 3 components, generate x, y, z and magnitude datasets
            array = point_data.GetArray(array_name)
            if array.GetNumberOfComponents() == 3:
                #print("3-component array found:", array_name)
                self.generate_xyz_datasets(dataset, array_name)
                self.array_names[array_name] = "Vector"
            elif array.GetNumberOfComponents() == 1:
                self.array_names[array_name] = "Scalar"              

    def set_colouring_by_dataset(self, data_set_array_name, data_set, mapper):
        """
        Set the colouring of the mapper based on the specified dataset array name.
        """
        # no data set array specified - disable scalar bar and colouring
        if data_set_array_name is None or data_set_array_name == "No Colouring":
            self.scalar_bar.SetVisibility(0)
            self.state.selected_colour = "No Colouring"
            self.state.flush()
            mapper.ScalarVisibilityOff()
            return
        min, max = data_set.GetPointData().GetArray(data_set_array_name).GetRange()
        self.lut.SetRange(min, max)
        self.lut.Build()
        mapper.SetScalarRange(min, max)
        mapper.SetColorModeToMapScalars()  # essential to enable colour mapping
        mapper.SetScalarModeToUsePointFieldData()  # use point data for colouring
        mapper.SelectColorArray(data_set_array_name)
        mapper.ScalarVisibilityOn()
        self.scalar_bar.SetTitle(data_set_array_name)
        self.scalar_bar.SetVisibility(1)
    
    def set_representation(self, repr):
        """
        Set the representation of the surface actor based on the specified dataset array name.
        """
        property = self.surfaceActor.GetProperty()
        if repr == "Surface":
            property.SetRepresentationToSurface()
        elif repr == "Wireframe":
            property.SetRepresentationToWireframe()
        else:
            property.SetRepresentationToPoints()

    def add_streamlines(self):
        """
        Add streamlines to the object grid based on the velocity field.
        """
        self.streamTracer.SetInputData(self.objectGrid)
        self.streamTracer.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, self.velocityName
        )
        self.streamTracer.SetIntegrator(vtk.vtkRungeKutta4())
        self.streamTracer.SetIntegrationDirectionToBoth()
        self.streamTracer.SetMaximumError(1e-6)
        self.streamTracer.SetComputeVorticity(True)
        self.streamTracer.SetMaximumPropagation(100)
        streamline_seed_point = self.objectGrid.GetPoint(self.streamlineSeedPointIndex)
        self.seedSource.SetCenter(*streamline_seed_point)
        self.seedSource.SetNumberOfPoints(500)
        self.seedSource.SetRadius(1.0)
        self.streamTracer.SetSourceConnection(self.seedSource.GetOutputPort())
        streamline_mapper = vtk.vtkPolyDataMapper()
        streamline_mapper.SetInputConnection(self.streamTracer.GetOutputPort())
        self.streamlineActor.SetMapper(streamline_mapper)
        self.streamlineActor.GetProperty().SetColor(1.0, 1.0, 1.0)  # white

    def clip_grid(self):  # OPTIMISED
        """
        Clip the object grid using the specified scalar field.
        """        
        clipper = vtk.vtkClipDataSet()
        clipper.SetInputData(self.objectGrid)
        # invert clipping to keep inside region
        clipper.InsideOutOn()
        clipper.SetValue(0.0)
        clipper.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, self.clippingName)
        clipper.Update()
        self.objectGrid = clipper.GetOutput()  # clipped grid
    
    def generate_xyz_datasets(self, dataset, name):  # OPTIMISED
        """
        Generate separate datasets for x, y, z components of a 3D vector field.
        """
        point_data = dataset.GetPointData()
        three_d_array = numpy_support.vtk_to_numpy(point_data.GetArray(name))
        #print("Generating separate component datasets for:", name)
        for i, component_name in enumerate([f"{name}_x", f"{name}_y", f"{name}_z"]):
            component_array = numpy_support.numpy_to_vtk(three_d_array[:, i], deep=True)
            component_array.SetName(component_name)
            point_data.AddArray(component_array)
        magnitude = np.sqrt(three_d_array[:,0]**2 + three_d_array[:,1]**2 + three_d_array[:,2]**2)
        magnitude_array = numpy_support.numpy_to_vtk(magnitude, deep=True)
        magnitude_array.SetName(f"{name}_mag")
        point_data.AddArray(magnitude_array)        
    
    def generate_magnitude_dataset(self, dataset, name):  # OPTIMISED
        """
        Generate magnitude dataset from separate x, y, z component datasets.
        """
        point_data = dataset.GetPointData()
        qty_x = numpy_support.vtk_to_numpy(point_data.GetArray(f"{name}_x"))
        qty_y = numpy_support.vtk_to_numpy(point_data.GetArray(f"{name}_y"))
        qty_z = numpy_support.vtk_to_numpy(point_data.GetArray(f"{name}_z"))
        mag_values = np.sqrt(qty_x**2 + qty_y**2 + qty_z**2)
        # stack arrays to create the 3D vec
        three_d_values = np.column_stack((qty_x, qty_y, qty_z))
        magnitude_array = numpy_support.numpy_to_vtk(mag_values, deep=True)
        magnitude_array.SetName(f"{name}_mag")
        qty_array = numpy_support.numpy_to_vtk(three_d_values, deep=True)
        qty_array.SetName(f"{name}_3D")
        point_data.AddArray(magnitude_array)
        point_data.AddArray(qty_array)

    def on_middle_click(self, caller, event):
        """
        Handle right click event to pick a cell.
        """
        click_pos = self.interactor.GetEventPosition()
        self.cellPicker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_cell_id = self.cellPicker.GetCellId()
        if picked_cell_id >= 0:
            pickedCell = self.objectSurface.GetCell(picked_cell_id)
            cellNormal = self.surface_cell_norms_np[picked_cell_id]
            picked_point_id = -1
            pickedPointNormal = None
            track_minimum_dot_product = sys.float_info.max
            # the below method finds the cellpoint with normal most closely matching the normal of the cell
            for i in range(pickedCell.GetNumberOfPoints()):
                point_id = pickedCell.GetPointId(i)
                pointNormal = self.surface_pt_norms_np[point_id]
                dot_product = np.dot(cellNormal, pointNormal)
                if dot_product < track_minimum_dot_product:
                    track_minimum_dot_product = dot_product
                    picked_point_id = point_id
                    pickedPointNormal = pointNormal
            if self.doSlicing is False:
                print("No skeleton loaded for slicing operations - performing flow calculations through face.")
                self.BFS_planar_constraint(picked_point_id)
                return
            axis = self.find_local_axis(picked_point_id)
            # check if axis is approximately aligned with point normal
            # this indicates whether we are at the end of a branch i.e. at a face of the aorta
            axis_normalized = axis / np.linalg.norm(axis)
            point_normal_normalized = np.array(pickedPointNormal) / np.linalg.norm(pickedPointNormal)
            alignment = np.abs(np.dot(axis_normalized, point_normal_normalized))
            if np.isclose(alignment, 1.0, atol=0.2):
                print(f"Clicked near end face of branch - performing volume flow rate and average\\"
                      f"pressure calculations across face.")
                self.BFS_planar_constraint(picked_point_id)
            else:
                print(f"Clicked along branch - performing local slice visualisation.")
                self.show_local_slice(np.array(self.objectSurface.GetPoint(picked_point_id)), axis)
        else:
            print("No cell picked")
            pass

    def find_local_axis(self, pt_id):
        """
        Find the closest point on the skeleton to the clicked position.
        """
        clicked_pt_coords = np.array(self.objectSurface.GetPoint(pt_id))
        self.skeleton_to_pt_distances = np.linalg.norm(self.skeleton_points - clicked_pt_coords, axis=1)
        # need to find axis of the skeleton at the closest point
        # get closest point id and find nearest neighbours to define axis
        closest_point_id = np.argmin(self.skeleton_to_pt_distances)
        # get cells connected to the closest point if the type of the vtkcells is VTK_LINE
        tangents = []
        if self.skeleton.GetCellType(0) == vtk.VTK_LINE:
            print("Using connected cells to define tangents along VTK_LINE skeleton")
            connected_closest_cells = vtk.vtkIdList()
            closest_point_coords = self.skeleton_points[closest_point_id]
            self.skeleton.GetPointCells(closest_point_id, connected_closest_cells)
            for i in range(connected_closest_cells.GetNumberOfIds()):
                cell_id = connected_closest_cells.GetId(i)
                cell_point_ids = vtk.vtkIdList()
                self.skeleton.GetCellPoints(cell_id, cell_point_ids)
                for j in range(cell_point_ids.GetNumberOfIds()):
                    if cell_point_ids.GetId(j) != closest_point_id:
                        tangent = np.array(self.skeleton.GetPoint(cell_point_ids.GetId(j))) - closest_point_coords
                        break  # only need one tangent point from one cell
                tangents.append(tangent)
        else:
            # use 3 closest points to define tangents
            print("Using 3 closest points to define tangents")
            closest_point_ids = np.argsort(self.skeleton_to_pt_distances)[:3]
            tangent1 = self.skeleton_points[closest_point_ids[1]] - self.skeleton_points[closest_point_ids[0]]
            tangent2 = self.skeleton_points[closest_point_ids[2]] - self.skeleton_points[closest_point_ids[0]]
            if np.dot(tangent1, tangent2) > 0:
                print("Both tangents in same direction - likely end of branch")
                tangents.append(tangent1)  # only append one tangent
            else:
                tangents.append(tangent1)
                tangents.append(tangent2)            
        if len(tangents) < 2:
            # end of branch case - only one tangent available
            skeleton_axis = tangents[0]
        else:
            # weighted tangents
            skeleton_axis = tangents[0] - tangents[1]
        return skeleton_axis

    def show_local_slice(self, clicked_pt_coords, skeleton_axis):
        """
        Show a local slice of the object grid at the clicked position.
        """
        self.create_local_slice(click_coords=clicked_pt_coords,
                                                     skeleton_pt=clicked_pt_coords,
                                                     axis_vec=skeleton_axis)
        self.setup_arrays_in_dataset(self.slice)
        #TODO: re-enable calculations on slice
        #self.compute_normals_arrays(self.slice)
        #self.compute_cell_area_array(self.slice)
        #if self.doVelocityCalcs:
        #    Q=self.calculate_flow_rate(self.velocityName, self.slice, None)
        #    print(f"Volume Flow Rate Through Slice: {np.round(Q,2)}cm^3/s \nAccumulated Volume Flow Rate: {self.Q_running_total:.2e}cm^3/s")
        #for array_name in self.array_names.keys():
        #    # calculate average and total values on the slice for each array
        #    avg = self.calculate_value(array_name, self.slice, "average", None)
        #    total = self.calculate_value(array_name, self.slice, "total", None)
        #    print(f"Slice - Average {array_name}: {np.round(avg,2)}, Total {array_name}: {np.round(total,2)}")
        average_pressure = self.calculate_value(self.pressureName, self.slice, "average", None)
        print(f"Average Pressure on Slice: {np.round(average_pressure,2)}Pa")       
        self.sliceMapper.SetInputData(self.slice)
        self.sliceActor.SetVisibility(True)
        self.surfaceActor.SetVisibility(True)
        self.objectSurface.GetPointData().SetActiveScalars(None)
        self.surfaceMapper.SetScalarVisibility(False)
        # Make the aorta semi-transparent so you can see the slice inside
        self.surfaceActor.GetProperty().SetOpacity(0.2)
        # if selected colouring is a vector field, append component suffix
        print("Selected colour:", self.state.selected_colour)
        print("Selected component:", self.state.selected_component)
        print(self.array_names.get(self.state.selected_colour))
        if self.array_names.get(self.state.selected_colour) == "Vector":
            array_to_colour_by = f"{self.state.selected_colour}_{self.state.selected_component}"
        else:
            array_to_colour_by = self.state.selected_colour
        print("Colouring slice by:", array_to_colour_by)
        self.set_colouring_by_dataset(array_to_colour_by, self.slice, self.sliceMapper)
        self.controller.view_update()
    
    def create_local_slice(self, click_coords, skeleton_pt, axis_vec):
        """Create a local slice of the object grid."""
        self.plane.SetOrigin(skeleton_pt)
        self.plane.SetNormal(axis_vec)
        self.cutter.SetInputData(self.objectGrid) # Ensure input is set
        self.connectivity.SetClosestPoint(click_coords)
        self.connectivity.Update()
        self.slice = self.connectivity.GetOutput()

    def BFS_planar_constraint(self, seed_point_id):
        """
        Search neighbouring cells from a seed point within a distance threshold.
        """
        print("Performing BFS with planar constraint from seed point ID:", seed_point_id)
        visitedCells = set()
        visitedPoints = set([seed_point_id])
        pointsToVisit = deque([seed_point_id])

        seed_point_normal = self.surface_pt_norms_np[seed_point_id]
        seed_point_coords = self.surface_points_coords[seed_point_id]

        point_cell_ids = vtk.vtkIdList()
        neighbour_points = vtk.vtkIdList()

        # BFS to find neighbouring cells within distance threshold
        while pointsToVisit:
            current_point_id = pointsToVisit.popleft()

            point_cell_ids.Initialize()  # clear previous IDs
            self.objectSurface.GetPointCells(current_point_id, point_cell_ids)

            for i in range(point_cell_ids.GetNumberOfIds()):
                cell_id = point_cell_ids.GetId(i)
                visitedCells.add(cell_id)
                # clear previous neighbour points
                neighbour_points.Initialize()
                self.objectSurface.GetCellPoints(cell_id, neighbour_points)
                valid_cell = True
                distance_threshold = 0.01*self.calculate_average_cell_side_length(cell_id)

                for j in range(neighbour_points.GetNumberOfIds()):
                    neighbour_point_id = neighbour_points.GetId(j)
                    if neighbour_point_id in visitedPoints:
                        continue
                    # store neighbour point coordinates as numpy array
                    neighbour_point_coords = self.surface_points_coords[neighbour_point_id]
                    # Compute perpendicular distance to the seed plane using dot product
                    dist = np.abs(np.dot((neighbour_point_coords - seed_point_coords), seed_point_normal))
                    if dist <= distance_threshold:
                        visitedPoints.add(neighbour_point_id)
                        pointsToVisit.append(neighbour_point_id)
                    else:
                        valid_cell = False
                if valid_cell is False:
                    visitedCells.remove(cell_id) 

        for array_name in self.array_names.keys():
            # calculate average and total values on the surface for each array
            avg = self.calculate_value(array_name, self.objectSurface, "average", visitedPoints)
            total = self.calculate_value(array_name, self.objectSurface, "total", visitedPoints)
            print(f"Surface - Average {array_name}: {np.round(avg,2)}, Total {array_name}: {np.round(total,2)}") 
        if self.doPressureCalcs:
            print(f"Average Pressure: {np.round(self.calculate_value(self.pressureName, self.objectSurface, 'average', visitedPoints),2)}Pa")
            pass
        if self.doVelocityCalcs:
            print(f"Volume Flow Rate Through Face: {np.round(self.calculate_flow_rate(self.velocityName, self.objectSurface, visitedCells),2)}cm^3/s \
                  \nAccumulated Volume Flow Rate: {self.Q_running_total:.2e}cm^3/s")
            pass

        self.shade_surface_points(visitedPoints)
    
    def calculate_average_cell_side_length(self, cell_id):
        """
        Calculate average side length of a cell.
        """
        cell = self.objectSurface.GetCell(cell_id)
        # use cell edges to calculate average side length
        num_edges = cell.GetNumberOfEdges()
        if num_edges == 0:  # degenerate cell
            return 0.0
        total_length = 0.0
        for i in range(num_edges):
            edge = cell.GetEdge(i)
            p1_id = edge.GetPointId(0)
            p2_id = edge.GetPointId(1)
            p1 = self.surface_points_coords[p1_id]
            p2 = self.surface_points_coords[p2_id]
            total_length += np.linalg.norm(p1 - p2)
        return total_length / num_edges

    def calculate_value(self, array_name, dataset, calc_type, point_ids=None):
        """
        Compute average or total value over a set of points.
        calc_type: "average" or "total"
        """
        point_data = dataset.GetPointData()
        data_vtk_array = point_data.GetArray(array_name)
        if not data_vtk_array:
            raise ValueError(f"Array {array_name} not found in point data")
        if calc_type not in ["average", "total"]:
            raise ValueError("Invalid calc_type. Use 'average' or 'total'.")
        data_np_array = numpy_support.vtk_to_numpy(data_vtk_array)  # convert to numpy array for easier indexing
        if point_ids is not None:
            target_vals = data_np_array[list(point_ids)]  # extract values at specified point IDs
        else:
            target_vals = data_np_array
        # if size of target_vals is zero, return 0 to avoid division by zero
        if target_vals.size == 0:
            return 0.0
        # compute average or total using numpy
        return np.mean(target_vals) if calc_type == "average" else np.sum(target_vals)
    
    def calculate_flow_rate(self, array_name, dataset, cell_ids=None):
        """
        Compute flow rate through a set of cells along the seed normal direction.
        """
        point_data = dataset.GetPointData()
        vel_vtk_array = point_data.GetArray(array_name)
        if not vel_vtk_array:
            raise ValueError(f"Array {array_name} not found in point data")
        vel_np_array = numpy_support.vtk_to_numpy(vel_vtk_array)  # convert to numpy array for easier indexing
        Q = 0.0
        if cell_ids is None:
            # use all cell ids if none specified
            cell_ids = range(dataset.GetNumberOfCells())
        print("Calculating flow rate through", len(cell_ids), "cells")
        for cell_id in cell_ids:
            # get cell normal - cannot use self.surface_cell_norms_np as this is for the entire surface
            if dataset == self.objectSurface:
                normal = self.surface_cell_norms_np[cell_id]
                area = self.surface_cell_areas_np[cell_id]
            elif dataset == self.slice:
                normal = self.slice_cell_norms_np[cell_id]
                area = self.slice_cell_areas_np[cell_id]
            else:
                raise ValueError("Dataset must be either mesh surface or slice")
            cell_point_ids = vtk.vtkIdList()
            dataset.GetCellPoints(cell_id, cell_point_ids)
            cell_pt_ids_np = np.array([cell_point_ids.GetId(i) for i in range(cell_point_ids.GetNumberOfIds())])
            # get dataset 
            v_dot_n = np.sum(vel_np_array[cell_pt_ids_np] * normal, axis=1) 
            Q += np.mean(v_dot_n) * area
        self.Q_running_total += Q
        return Q

    def shade_surface_points(self, point_ids):
        """
        Shade selected surface points in red.
        """
        self.sliceActor.SetVisibility(False)
        self.surfaceActor.GetProperty().SetOpacity(1.0)
        self.surfaceMapper.SetScalarVisibility(False)
        num_pts = self.objectSurface.GetNumberOfPoints()
        colors = np.full((num_pts, 3), 255, dtype=np.uint8)  # default white - uint8 for vtk RGB colors is essential
        if point_ids:
            colors[list(point_ids)] = [255, 0, 0]  # selected points in red
        color_array = numpy_support.numpy_to_vtk(colors, deep=True)
        color_array.SetName("CellSelectionColors")
        point_data = self.objectSurface.GetPointData()
        point_data.AddArray(color_array)
        point_data.SetActiveScalars("CellSelectionColors")
        self.surfaceMapper.ScalarVisibilityOn()
        self.surfaceMapper.SetColorModeToDirectScalars()  # 
        # Ensure it's looking at Point Data
        self.surfaceMapper.SetScalarModeToUsePointData()
        # Force the selection of the new array
        self.surfaceMapper.SelectColorArray("CellSelectionColors")
        # disable scalar bar and revert to default colouring
        self.scalar_bar.SetVisibility(0)
        self.controller.view_update()

    def file_reader(self, path):
        """
        Read file based on its extension and load into appropriate VTK data structure.
        """
        if not path:
            raise ValueError("No file path provided")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        ext = path[-4:]
        if ext == ".vtu":
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif ext == ".vtk":
            reader = vtk.vtkUnstructuredGridReader()
        elif ext == ".vti":
            reader = vtk.vtkXMLImageDataReader()
        elif ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        reader.SetFileName(path)
        reader.Update()
        if ext in [".vtu", ".vtk"]:
            self.objectGrid = reader.GetOutput()
        elif ext == ".vti":
            self.image = reader.GetOutput()
        elif ext == ".vtp":
            self.skeleton = reader.GetOutput()
        
    #def find_gradients(self):
    #    """
    #    Compute gradients of velocity and pressure fields in the object grid.
    #    """
    #    if self.find_gradients_of is None:
    #        print("No fields specified for gradient calculation.")
    #        return
    #    point_data = self.objectGrid.GetPointData()
    #    gradientFilter = vtk.vtkGradientFilter()
    #    gradientFilter.SetInputData(self.objectGrid)
    #    for array_name in self.find_gradients_of:
    #        if point_data.GetArray(array_name) is None:
    #            print(f"Array '{array_name}' not found in object grid - skipping gradient calculation.")
    #            continue
    #        gradientFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, array_name)
    #        gradientFilter.SetResultArrayName(f"{array_name}_gradient")
    #        gradientFilter.Update()
    #        result_array = gradientFilter.GetOutput().GetPointData().GetArray(f"{array_name}_gradient")
    #        print(f"number of components in gradient array for '{array_name}': {result_array.GetNumberOfComponents()}")
    #        self.objectGrid.GetPointData().AddArray(result_array)
    #        print(f"Computed gradient for '{array_name}'.")
    
    def compute_normals_arrays(self, dataset):
        """
        Set up normals arrays for surface or slice.
        """
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(dataset)
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.ComputeCellNormalsOn()
        normalsFilter.Update()
        np_pt_norms_arr = numpy_support.vtk_to_numpy(normalsFilter.GetOutput().GetPointData().GetNormals())
        np_cell_norms_arr = numpy_support.vtk_to_numpy(normalsFilter.GetOutput().GetCellData().GetNormals())
        if dataset == self.objectSurface:
            self.surface_pt_norms_np = np_pt_norms_arr
            self.surface_cell_norms_np = np_cell_norms_arr
        elif dataset == self.slice:
            self.slice_pt_norms_np = np_pt_norms_arr
            self.slice_cell_norms_np = np_cell_norms_arr
        else:
            raise ValueError("Dataset must be either mesh surface or slice")
    
    def compute_cell_area_array(self, dataset):
        """
        Compute cell area array for surface or slice.
        """
        # Calculate area of all 2D cells
        cellSizeFilter = vtk.vtkCellSizeFilter()
        cellSizeFilter.SetInputData(dataset)
        cellSizeFilter.SetComputeArea(True)
        cellSizeFilter.Update()
        # SafeDownCast checks type compativility and returns ptr to derived class safely
        np_cell_areas_arr = numpy_support.vtk_to_numpy(vtk.vtkDataSet.SafeDownCast(cellSizeFilter.GetOutput()).GetCellData().GetArray("Area"))
        if dataset == self.objectSurface:
            self.surface_cell_areas_np = np_cell_areas_arr
        elif dataset == self.slice:
            self.slice_cell_areas_np = np_cell_areas_arr
        else:
            raise ValueError("Dataset must be either mesh surface or slice")

    def setup_grids_arrays_actors(self):
        """
        Set up grids, arrays, and actors for rendering.
        """
        if self.renderingObject is True:
            # if clipping array specified, clip the object grid
            if self.clippingName:
                self.clip_grid()
            # compute gradients if specified
            #self.find_gradients()
            self.surfaceFilter.SetInputData(self.objectGrid)
            self.surfaceFilter.Update()
            self.objectSurface = self.surfaceFilter.GetOutput()
            self.compute_normals_arrays(self.objectSurface)
            self.compute_cell_area_array(self.objectSurface)
            self.surfaceMapper.SetInputData(self.objectSurface)
            self.surfaceMapper.SetScalarModeToUsePointFieldData()
            self.surfaceActor.SetMapper(self.surfaceMapper)
            # store all coords as numpy array for faster access
            self.surface_points_coords = numpy_support.vtk_to_numpy(self.objectSurface.GetPoints().GetData())
            # find ALL arrays in the object grid
            self.setup_arrays_in_dataset(self.objectSurface)
            if self.velocityName:
                if not (self.velocityName in self.array_names.keys() and self.array_names[self.velocityName] == "Vector"):
                    raise ValueError(f"Velocity array '{self.velocityName}' not found in object grid.")
                else:
                    self.doVelocityCalcs = True
            if self.doPressureCalcs is True:
                if self.pressureName not in self.array_names.keys():
                    raise ValueError(f"Pressure array '{self.pressureName}' not found in object grid.")
                else:
                    self.doPressureCalcs = True
            self.setup_scalar_bar()
            self.renderer.AddActor(self.surfaceActor)

        if self.renderingStreamlines is True:
            self.renderer.AddActor(self.streamlineActor)
        
        if self.doSlicing is True:
            self.renderer.AddActor(self.sliceActor)

        self.renderer.ResetCamera()
    
    def setup_scalar_bar(self):
        """
        Setup the scalar bar actor for the renderer.
        """
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetMaximumWidthInPixels(100)
        self.scalar_bar.SetMaximumHeightInPixels(200)
        self.scalar_bar.SetPosition(0.85, 0.1)
        self.scalar_bar.SetOrientationToVertical()
        self.scalar_bar.SetVisibility(0)  # initially hidden
        # Create a property for both Title and Labels
        text = self.scalar_bar.GetLabelTextProperty()
        text.SetFontFamilyToArial()
        text.SetFontSize(14)
        text.BoldOn()
        text.ItalicOff()
        text.ShadowOff()
        text.SetColor(1, 1, 1)
        self.scalar_bar.GetTitleTextProperty().ShallowCopy(text)  # Title uses same text property
        self.scalar_bar.SetLabelFormat("%.2f")
        self.renderer.AddActor(self.scalar_bar)
    
    def update_scalar_bar(self):
        """
        Update the scalar bar to reflect current colouring.
        """
        if self.state.selected_colour == "No Colouring":
            self.scalar_bar.SetVisibility(0)
            self.scalar_bar.Modified()
            return
        self.scalar_bar.SetVisibility(1)
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.Modified()
    
    def clear_slice_and_selection(self):
        """
        Clear any existing slice and selection colouring, resetting to default view.
        """
        print("Clearing slice and selection colouring.")
        # reset opacity and hide slice
        self.sliceActor.SetVisibility(False)
        self.surfaceActor.GetProperty().SetOpacity(1.0)
        if self.array_names.get(self.state.selected_colour) == "Vector":
            array_to_colour_by = f"{self.state.selected_colour}_{self.state.selected_component}"
        else:
            array_to_colour_by = self.state.selected_colour
        self.set_colouring_by_dataset(array_to_colour_by, self.objectSurface, self.surfaceMapper)
        self.update_scalar_bar()
        self.controller.view_update()

    def render(self):
        """Return a Trame RemoteView for the render window."""
        view = trame_vtk.VtkRemoteLocalView(self.render_window)
        view.enable_interaction = True  # enable rotation, pan, zoom
        self.controller.view_update = view.update
        return view

def main():
    if len(sys.argv) != 2:
        print("Usage: python gradient.py <config.yaml>")
        sys.exit(1)
    
    config = load_config(sys.argv[1])

    mesh_path = config.get("mesh_path")
    skeleton_path = config.get("skeleton_path")
    velocity_array_name = config.get("velocity_array_name")
    pressure_array_name = config.get("pressure_array_name")
    clip_array_name = config.get("clip_array_name")
    streamline_seed_pt_idx = config.get("streamline_seed_point_index")
    #find_gradients_of = config.get("find_gradients_of", [])

    print("Mesh Path:", mesh_path)
    print("Skeleton Path:", skeleton_path)
    print("Velocity Array Name:", velocity_array_name)
    print("Pressure Array Name:", pressure_array_name)
    print("Clip Array Name:", clip_array_name)
    print("Streamline Seed Point Index:", streamline_seed_pt_idx)
    #print("Find Gradients Of:", find_gradients_of)

    server = get_server()
    state = server.state
    controller = server.controller

    viewer = VTUViewer(state, controller, mesh_path, skeleton_path,
                       velocity_array_name, pressure_array_name,
                       clip_array_name, streamline_seed_pt_idx)

    #state.log_text = ""
    #def gui_print(*args, **kwargs):
    #    text = " ".join(str(a) for a in args)
    #    builtins._original_print(text)       # keep console printing
    #    state.log_text = text + "\n"        # update GUI
    #    state.flush()
    #if not hasattr(builtins, "_original_print"):
    #    builtins._original_print = builtins.print
    #builtins.print = gui_print  # override print function

    state.colour_options = ["No Colouring"] 
    state.selected_colour = "No Colouring"

    state.selected_component = "mag"
    state.component_options = ["x", "y", "z", "mag"]

    state.representation_options = ["Surface", "Wireframe", "Points"]  # Surface, Wireframe, Points
    state.representation = "Surface"  # default to Surface

    for name in viewer.array_names.keys():
        state.colour_options.append(name)
    
    @state.change("selected_colour")
    def update_colour(selected_colour, **kwargs):
        if viewer.array_names.get(state.selected_colour) == "Vector":
            array_to_colour_by = f"{state.selected_colour}_{state.selected_component}"
            state.component_options = ["x", "y", "z", "mag"]
            state.flush()  # ensure component selection updates in GUI
        else:
            array_to_colour_by = state.selected_colour 
            state.component_options = ["mag"]
            state.selected_component = "mag"
            state.flush()  # ensure component selection updates in GUI
        # if slice is visible, colour by slice instead
        if viewer.sliceActor.GetVisibility() == 1:
            dataset, mapper = viewer.slice, viewer.sliceMapper
        else:
            dataset, mapper = viewer.objectSurface, viewer.surfaceMapper
        viewer.set_colouring_by_dataset(
            array_to_colour_by,
            dataset, 
            mapper
        )
        viewer.update_scalar_bar()
        controller.view_update()

    @state.change("selected_component")
    def update_component(selected_component, **kwargs):
        # if selected colouring is a vector field, append component suffix
        if viewer.array_names.get(state.selected_colour) == "Vector":
            array_to_colour_by = f"{state.selected_colour}_{selected_component}"
        else:
            array_to_colour_by = state.selected_colour        
        # if slice is visible, colour by slice instead
        if viewer.sliceActor.GetVisibility() == 1:
            dataset, mapper = viewer.slice, viewer.sliceMapper
        else:
            dataset, mapper = viewer.objectSurface, viewer.surfaceMapper
        viewer.set_colouring_by_dataset(
            array_to_colour_by,
            dataset, 
            mapper
        )
        viewer.update_scalar_bar()
        controller.view_update()    

    @state.change("representation")
    def update_repr(representation, **kwargs):
        viewer.set_representation(representation)
        controller.view_update()

    #@state.change("timestep")
    #def update_timestep(timestep, **kwargs):
    #    viewer.velocityName = "velocity_0" + str(7040+timestep)
    #    viewer.pressureName = "pressure_0" + str(7040+timestep)
    #    print("New velocity dataset:", viewer.velocityName)
    #    print("New pressure dataset:", viewer.pressureName)
    #    viewer.generate_xyz_datasets(viewer.objectSurface)
    #    viewer.set_colouring_by_dataset(valToColourDataSet[state.colour_value], viewer.objectSurface, viewer.surfaceMapper)
    #    viewer.update_scalar_bar()
    #    viewer.Q_running_total = 0  # reset accumulated flow rate
    #    controller.view_update()

    with SinglePageLayout(server) as layout:
        # Toolbar
        with layout.toolbar:
            v3.VSpacer()
            v3.VSelect(
                label="Representation",
                v_model="representation",
                items=("representation_options",), # Pass the list directly
                dense=True,
            )
        with layout.toolbar:
            v3.VSpacer()
            v3.VSelect(
                label="Colouring",
                v_model="selected_colour",
                items=("colour_options",), # Pass the list directly
                dense=True,
            )
        with layout.toolbar:
            v3.VSpacer()
            v3.VSelect(
                label="Component",
                v_model="selected_component",
                items=("component_options",), # Pass the list directly
                dense=True,
            )
        # button to clear slice and selection
        with layout.toolbar:
            v3.VBtn(
                "Clear Slice/Selection",
                dense=True,
                click=viewer.clear_slice_and_selection,
            )
        #with layout.toolbar:
        #    v3.VSpacer()
        #    v3.VSlider(
        #        v_model="timestep",
        #        min=0,
        #        max=150,
        #        step=5,
        #        hide_details=True,
        #        dense=True,
        #        style="max-width: 300px",
        #        label="Timestep"
        #    )

        # VTK view content
        with layout.content:
            view = viewer.render()

        #with layout.footer:
        #    html.Pre(
        #        "{{ log_text }}",
        #        style=(
        #            "height: 200px; overflow-y: auto; width: 100%; "
        #            "background:#000; color:#0f0; padding:10px; border:1px solid #333; "
        #        ),
        #    )

    # Start server
    server.start(host="0.0.0.0", port=1234)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    main()