import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, vtk as trame_vtk
import os
import numpy as np
import asyncio
from vtkmodules.util import numpy_support
import yaml
import sys
from collections import deque
import plotly.graph_objects as go
from trame_plotly.widgets import plotly

class Scatter3DPlot:
    def __init__(self, mesh, state):
        self.state = state
        self.arrays, self.array_names = self.extract_arrays(mesh)
        self.x, self.y, self.z = self.get_xyz_coordinates(mesh)
        self.fig = self.create_figure()
    
    def get_xyz_coordinates(self, mesh):
        points = mesh.GetPoints()
        x = numpy_support.vtk_to_numpy(points.GetData())[:, 0]
        y = numpy_support.vtk_to_numpy(points.GetData())[:, 1]
        z = numpy_support.vtk_to_numpy(points.GetData())[:, 2]
        return x, y, z
    
    def extract_arrays(self, mesh):
        arrays, array_names = [], []
        point_data = mesh.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()
        for i in range(num_arrays):
            array_name = point_data.GetArrayName(i)
            print(f"Extracting array: {array_name}")
            array = point_data.GetArray(array_name)
            if array.GetNumberOfComponents() != 1:
                continue
            array_names.append(array_name)
            arrays.append(numpy_support.vtk_to_numpy(array))
        return arrays, array_names

    def create_figure(self):
        # Prepare the matrix for customdata
        print("Arrays: ", self.arrays)
        data = np.stack(self.arrays, axis=-1)
    
        hover_lines = []
        for i, name in enumerate(self.array_names):
            hover_lines.append(f"{name}: %{{customdata[{i}]:.1f}}")
        scatter1 = go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode='markers',
            marker=dict(size=3, color='white', opacity=0.8),
            customdata=data, 
            hovertemplate=hover_lines[0],  # default to first array
            name="Data"
        )
        fig = go.Figure(
            data=[scatter1]
        )
        fig.update_layout(
            template="plotly_dark",
            scene=dict(aspectmode='data',
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", showbackground=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", showbackground=False),
                        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", showbackground=False),),
            margin=dict(l=0, r=0, b=0, t=30),
            updatemenus=[{
            "buttons": [
                {
                    "label": str(self.array_names[i]),
                    "method": "update",
                    "args": [{"hovertemplate": [hover_lines[i]]}]
                }
                for i in range(len(self.array_names))
            ],
            "direction": "down",
            "showactive": True,
            "type": "dropdown",
            "x": 0.05,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
            "pad": {"r": 10, "t": 10},
            "bgcolor": "#333333",
            "bordercolor": "#555555",
            "font": {"color": "white", "size": 12}
            }],
            )
        
        return fig

class SphereSourceCustom:
    def __init__(self):
        super().__init__()
        skeletonSphereSource = vtk.vtkSphereSource()
        skeletonSphereMapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        skeletonSphereSource.SetRadius(0.05)
        skeletonSphereMapper.SetInputConnection(skeletonSphereSource.GetOutputPort())
        self.actor.SetMapper(skeletonSphereMapper)
        self.actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # cyan
        self.actor.SetVisibility(False)  # initially hidden

class VTUViewer:
    def __init__(self, state, controller, mesh_path,
                        image_path,
                        mesh_extraction_array_name,
                        mesh_extraction_value,
                        image_scalar_name,
                        mesh_target_num_points,
                        streamline_seed_pt_idx,
                        animation_enabled,
                        initial_timestep,
                        final_timestep,
                        update_time,
                        velocity_array_name,
                        pressure_array_name,
                        skeleton_path,
                        mu):
        """
        Initialize the VTUViewer with VTK rendering components.
        """

        """ INTERACTOR AND RENDERER SETUP """
        self.view=None
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(True)
        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(300,300)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        """ EXTRA VARIABLES AND SCALAR BAR SETUP """
        self.imageScalarName = image_scalar_name
        self.velocityName = velocity_array_name
        self.pressureName = pressure_array_name
        self.doVelocityCalcs, self.doPressureCalcs = False, False
        if image_path and not self.imageScalarName:
            raise ValueError("No image scalar name specified.")
        self.state = state
        self.controller = controller
        self.mu = mu
        if not self.mu:
            print("No dynamic viscosity specified, defaulting to 0.004")
            self.mu = 0.004
        self.current_picked_skeleton_line_id = None

        """ OBJECT ACTORS, FILTERS AND MAPPERS SETUP """
        if mesh_path:
            self.objectGrid = vtk.vtkUnstructuredGrid()
            self.decimateFilter = vtk.vtkDecimatePro()
            self.surfaceMapper = vtk.vtkDataSetMapper()
            self.surfaceActor = vtk.vtkActor()
            self.surfaceMapper.SetScalarModeToUsePointFieldData()
            self.surfaceMapper.SetColorModeToMapScalars()
            self.surfaceActor.SetMapper(self.surfaceMapper)
            self.surfaceActor.GetProperty().SetColor(0.0, 1.0, 0.0)  # green
            self.surfaceActor.SetVisibility(True)
            self.contourFilter = vtk.vtkContourFilter()
            self.clipFilter = vtk.vtkClipDataSet()
            self.surfaceExtractionFilter = vtk.vtkDataSetSurfaceFilter()
            self.meshExtractionArrayName = mesh_extraction_array_name
            self.meshTargetNumPoints = mesh_target_num_points if mesh_target_num_points else 25000
            self.meshExtractionValue = mesh_extraction_value
            self.cellPicker = vtk.vtkCellPicker()
            self.cellPicker.SetTolerance(0.0005)  # small tolerance for picking
            self.cellPicker.AddPickList(self.surfaceActor)  # allows picking ONLY from points/cells on the surface actor
            self.cellPicker.PickFromListOn()
            if self.meshExtractionValue is None:
                print("No mesh extraction value specified, defaulting to 0.0")
                self.meshExtractionValue = 0.0
            self.surface_pt_norms_np = None
            self.surface_cell_norms_np = None
            self.lut = vtk.vtkLookupTable()
            self.lut.SetNumberOfTableValues(256)
            colour_tf = vtk.vtkColorTransferFunction()
            colour_tf.SetColorSpaceToRGB()  # ensures smooth RGB interpolation
            # Add colors at key points (0=blue, 0.33=light blue, 0.66=light orange, 1=red)
            colour_tf.AddRGBPoint(0.0, 0.0, 0.0, 1.0)   # Blue
            colour_tf.AddRGBPoint(0.33, 0.6, 0.8, 1.0)   # Light Blue
            colour_tf.AddRGBPoint(0.66, 1.0, 0.7, 0.4)   # Light Orange
            colour_tf.AddRGBPoint(1.0, 1.0, 0.0, 0.0)   # Red
            # Convert to a LUT for mapper
            for i in range(256):
                rgb = list(colour_tf.GetColor(i / 255.0)) + [1.0]  # [R, G, B, A]
                self.lut.SetTableValue(i, *rgb)  # Set RGBA values. note the * unpacks the list
            # Attach to mapper
            self.surfaceMapper.SetLookupTable(self.lut)
            self.colour_ranges = {}
            self.scalar_bar = vtk.vtkScalarBarActor()
            self.scalar_bar.SetLookupTable(self.lut)
            self.scalar_bar.SetVisibility(True)

        """ IMAGE DATA SETUP """
        if image_path:
            self.pointCloudLayers = {}
            self.numPoints = {}  # to store number of points in each layer
            self.totalPointCloudPoints = 1e6 if animation_enabled is True else 2e6  # limit total points based on animation

        """ SLICE ACTOR, FILTERS AND MAPPERS SETUP """
        self.slice = vtk.vtkPolyData()
        self.sliceMapper = vtk.vtkPolyDataMapper()
        self.sliceActor = vtk.vtkActor()
        self.sliceActor.SetMapper(self.sliceMapper)
        self.sliceActor.GetProperty().SetColor(1.0, 1.0, 0.0)  # yellow
        self.sliceActor.SetVisibility(False)
        self.slice_cell_areas_np = None
        self.slice_pt_norms_np = None
        self.slice_cell_norms_np = None
        self.sliceMapper.SetLookupTable(self.lut)
        self.plane = vtk.vtkPlane()
        self.planeCutter = vtk.vtkPlaneCutter()
        cleaner = vtk.vtkCleanPolyData()  # NEEDED FOR VTK 9.2.6 CONNECTIVITY FILTER BUG
        cleaner.SetInputConnection(self.planeCutter.GetOutputPort())
        cleaner.SetTolerance(0.0)
        self.connectivity = vtk.vtkConnectivityFilter()
        self.connectivity.SetInputConnection(cleaner.GetOutputPort())
        self.connectivity.SetExtractionModeToClosestPointRegion()
        self.sliceMapper.SetInputConnection(self.connectivity.GetOutputPort())
        self.sliceMapper.StaticOn()  # prevents automatic recalculation of scalar range on each update, which causes flickering in animation. we will manage scalar range manually in get_grid_surface()
        self.sliceActor.GetProperty().BackfaceCullingOff()
        self.sliceActor.GetProperty().LightingOff()

        """ STREAMLINE SETUP """
        if streamline_seed_pt_idx is not None:
            self.streamTracer = vtk.vtkStreamTracer()
            self.seedSource = vtk.vtkPointSource()
            self.streamlineSeedPointIndex = streamline_seed_pt_idx
            self.streamlineActor = vtk.vtkActor()

        """ TIMESTEP AND ANIMATION SETTINGS """
        self.animating = False
        if animation_enabled is True:
            self.animating = True
            self.update_ts = update_time if update_time else 0.15  # seconds
            if self.update_ts < 0.01 or self.update_ts > 0.3:
                print("Update time out of bounds (0.01 to 0.3s). Defaulting to 0.15s.")
                self.update_ts = 0.15
            if initial_timestep is None:
                print("Initial timestep not specified, defaulting to 1.")
                self.timestep = 1
            else:
                self.timestep = initial_timestep
            if not final_timestep:
                raise ValueError("Final timestep must be specified for animation.")
            self.max_timestep = final_timestep
            self.cached_contours = []  # to store precomputed contours

        """ SKELETON DATASET SETUP """
        self.skeleton = vtk.vtkPolyData()
        self.skeleton_with_cells = vtk.vtkPolyData()  # new dataset that will include line cells connecting skeleton points
        self.skeletonActor = vtk.vtkActor()

        """ GLYPH SETUP FOR vWSS VISUALISATION """
        self.glyph = vtk.vtkGlyph3D()  # glyph is customisable shape at each point - in this case arrows
        self.glyphMapper = vtk.vtkPolyDataMapper()
        self.glyphActor = vtk.vtkActor()
        self.glyphs_rendering = False

        self.setting_centerline_seeds = True
        self.centerline_seed_points = []  # to store seed points for centerline computation, set by clicking on surface
        self.setting_centerline_targets = False
        self.centerline_target_points = []  # to store target points for centerline computation, set by clicking on surface

        """ RENDERING SETTINGS AND FILE READING """
        self.renderingObject, self.renderingPointCloud, self.renderingStreamlines, self.doSlicing, self.doProbing = False, False, False, False, False
        if mesh_path:
            self.renderingObject = True
            self.file_reader(mesh_path)
        if image_path:
            self.renderingPointCloud = True
            self.file_reader(image_path)
        if streamline_seed_pt_idx is not None and self.velocityName:
            self.renderingStreamlines = True

        self.fig = go.FigureWidget()
        self.fig.add_scatter(x=[], y=[], mode='lines+markers', name=f"Branch {i+1}")
        self.fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
                               xaxis_title="Arc Length Along Skeleton",
                               yaxis_title="")
        self.state.plot = self.fig.to_dict()

        self.compute_centerline = True
        if skeleton_path:
            self.doProbing = True
            self.doSlicing = True
            self.file_reader(skeleton_path)
            self.setup_skeleton(skeleton_path_provided=True)
            self.compute_centerline = False

        self.setup_grids_arrays_actors()
    
    def setup_skeleton(self, skeleton_path_provided=False):
        """
        """
        self.setup_skeleton_with_cells()
        self.doProbing = True
        self.skeleton_mapper = vtk.vtkPolyDataMapper()
        self.skeleton_mapper.SetInputData(self.skeleton)
        self.skeletonActor.SetMapper(self.skeleton_mapper)
        self.skeletonActor.SetVisibility(False)
        self.skeleton_points = numpy_support.vtk_to_numpy(self.skeleton_with_cells.GetPoints().GetData())
        self.skeleton_to_pt_distances= np.zeros(self.skeleton_with_cells.GetNumberOfPoints())
        self.skeletonProbe = vtk.vtkProbeFilter()
        print("Probing along skeleton...")
        self.renderer.AddActor(self.skeletonActor)
        self.sphereSources = []
        self.skeletonCellLocator = vtk.vtkStaticCellLocator()
        print("Building cell locator for skeleton...")
        self.skeletonCellLocator.SetDataSet(self.skeleton)
        self.skeletonCellLocator.BuildLocator()

    def setup_arrays_in_dataset(self, dataset):
        """
        Setup necessary arrays in the dataset for visualisation and calculations.
        """
        # iterate through all arrays in point data
        point_data = dataset.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()
        self.array_names = {}
        for i in range(num_arrays):
            array_name = point_data.GetArrayName(i)
            # if array has 3 components, generate x, y, z and magnitude datasets
            array = point_data.GetArray(array_name)
            if array.GetNumberOfComponents() == 3:
                self.generate_xyz_datasets(dataset, array_name)
                self.array_names[array_name] = "Vector"
            elif array.GetNumberOfComponents() == 1:
                self.array_names[array_name] = "Scalar"
            else:
                print(f"Array '{array_name}' has, as of now, unsupported number of components ({array.GetNumberOfComponents()}). Skipping.")

    def add_streamlines(self):
        """
        Add streamlines to the object grid based on the velocity field.
        """
        if self.streamlineSeedPointIndex < 0 or self.streamlineSeedPointIndex >= self.objectGrid.GetNumberOfPoints():
            print("Adding streamlines failed: Seed point index out of bounds.")
            self.renderingStreamlines = False
            return
        self.streamTracer.SetInputData(self.clipFilter.GetOutput() if self.animating is False and self.meshExtractionArrayName is not None else self.objectGrid)  # use clipped mesh if not animating
        self.streamTracer.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, self.velocityName
        )
        self.streamTracer.SetIntegratorTypeToRungeKutta45()
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
        streamline_mapper.ScalarVisibilityOff()
        self.streamlineActor.SetMapper(streamline_mapper)
        self.streamlineActor.GetProperty().SetColor(1.0, 1.0, 1.0)  # white
    
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
        mapper.SetInterpolateScalarsBeforeMapping(0)
        mapper.StaticOn()
        mapper.SetScalarRange(min, max)
        mapper.SetColorModeToMapScalars()  # essential to enable colour mapping
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(data_set_array_name)
        mapper.ScalarVisibilityOn()  # controls whether to colour by scalars
        self.scalar_bar.SetTitle(data_set_array_name)
        self.scalar_bar.SetVisibility(1)
    
    async def animate(self):
        """Animate the render window by updating it periodically."""
        while self.state.animation:
            self.surfaceMapper.SetInputData(self.cached_contours[self.timestep - 1])
            self.controller.view_update()
            self.timestep = self.timestep + 1 if self.timestep < self.max_timestep else 1
            await asyncio.sleep(self.update_ts)

    def get_grid_surface(self):
        """
        Create a surface representation of the object grid.
        This is either a contour or the full surface.
        """
        array_name = f"{self.meshExtractionArrayName}_t{self.timestep}" if self.animating else self.meshExtractionArrayName
        if self.animating is True:
            self.contourFilter.SetInputData(self.objectGrid)
            self.contourFilter.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, array_name
            )
            self.contourFilter.SetValue(0, self.meshExtractionValue)
            self.contourFilter.Update()
            num_points = self.contourFilter.GetOutput().GetNumberOfPoints()
            print("Number of points in contour: ", num_points)
            self.decimateFilter.SetInputConnection(self.contourFilter.GetOutputPort())
            reduction = 1.0 - (self.meshTargetNumPoints / num_points) if num_points > self.meshTargetNumPoints else 0.0
            if reduction > 0.0:
                print("Reducing from ", num_points, " to approx ", self.meshTargetNumPoints, " points.")
                print("Reduction factor: ", reduction)
            print("Decimation reduction: ", reduction)
            self.decimateFilter.SetTargetReduction(reduction)
            self.decimateFilter.PreserveTopologyOn()
            self.decimateFilter.SetFeatureAngle(15.0)
            self.decimateFilter.Update()
            self.surfaceMapper.SetInputConnection(self.decimateFilter.GetOutputPort())
            self.surfaceMapper.Update()
        else:  # not animating
            if self.meshExtractionArrayName is not None:  # clip to get surface if extraction array given
                self.clipFilter.SetInputData(self.objectGrid)
                self.clipFilter.SetInputArrayToProcess(
                    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, array_name
                )
                self.clipFilter.SetValue(self.meshExtractionValue)
                self.clipFilter.InsideOutOn()
                self.clipFilter.Update()
                self.objectGrid = self.clipFilter.GetOutput()
                self.surfaceExtractionFilter.SetInputData(self.clipFilter.GetOutput())
            else:  # otherwise get full surface
                self.surfaceExtractionFilter.SetInputData(self.objectGrid)
            self.surfaceExtractionFilter.Update()
            self.surfaceMapper.SetInputConnection(self.surfaceExtractionFilter.GetOutputPort())
            self.surfaceMapper.Update()
        self.objectSurface = self.surfaceMapper.GetInput()
    
    def precompute_animation_surfaces(self):
        """
        Precompute animation surfaces for all timesteps and store them.
        Also calculate global colour ranges for ALL datasets.
        """
        for t in range(self.timestep, self.max_timestep+1):
            # Update filter to current timestep
            print(f"Precomputing mesh for timestep {t}...")
            print(f"Using array: {self.meshExtractionArrayName}_t{t}")
            self.contourFilter.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, f'{self.meshExtractionArrayName}_t{t}'
            )
            self.decimateFilter.Update()
            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(self.decimateFilter.GetOutput())
            self.cached_contours.append(polydata)
            # calculate global colour range for this dataset and update if needed
            self.populate_colour_ranges(polydata)
    
    def populate_colour_ranges(self, dataset):
        """
        Populate the colour ranges for velocity and pressure datasets in the given dataset.
        """
        for arr_name in self.colour_ranges.keys():
            array = dataset.GetPointData().GetArray(arr_name)
            if array:
                array_range = array.GetRange()
                self.colour_ranges[arr_name][0] = min(self.colour_ranges[arr_name][0], array_range[0])
                self.colour_ranges[arr_name][1] = max(self.colour_ranges[arr_name][1], array_range[1])
    
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
    
    def generate_xyz_datasets(self, dataset, name):  # OPTIMISED
        """
        Generate separate datasets for x, y, z components of a 3D vector field.
        """
        point_data = dataset.GetPointData()

        three_d_array = numpy_support.vtk_to_numpy(point_data.GetArray(name))
        print("Generating separate component datasets for ", name)

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
    
    def probe_along_skeleton(self):
        """
        Probe data along the skeleton points and store distances.
        """
        # Initialize the probe filter
        self.skeletonProbe.SetInputData(self.skeleton)
        self.skeletonProbe.SetSourceData(self.objectGrid)
        self.skeletonProbe.Update()
        self.scatter3D = Scatter3DPlot(
            self.skeletonProbe.GetOutput(),
            self.state
        )
        self.probe_ids = np.array([], dtype=int)

    def plot_probed_data(self):
        """
        Plot the probed data along the skeleton.
        """
        if self.current_picked_skeleton_line_id is None:
            print("No skeleton line picked for plotting.")
            return
        #array_name = self.state.selected_colour
        #if self.array_names.get(self.state.selected_colour) == "Vector":
        #    array_name = f"{self.state.selected_colour}_{self.state.selected_component}"
        #else:
        #    array_name = self.state.selected_colour
        #if array_name is None or array_name == "No Colouring":
        array_name = self.pressureName
        cell = self.skeletonProbe.GetOutput().GetCell(self.current_picked_skeleton_line_id)
        cell_point_ids = cell.GetPointIds()
        point_indices_list = [cell_point_ids.GetId(i) for i in range(cell_point_ids.GetNumberOfIds())]
        numpy_ids = np.array(point_indices_list)
        arc_length_array = numpy_support.vtk_to_numpy(self.skeleton.GetPointData().GetArray("Abscissas"))  # Abscissas is the arc length array name when skeleton is generated with VMTK
        arc_length_array = list(arc_length_array[numpy_ids])
        val_array = list(numpy_support.vtk_to_numpy(self.skeletonProbe.GetOutput().GetPointData().GetArray(array_name))[numpy_ids])
        with self.fig.batch_update():
            self.fig.data[0].x = arc_length_array
            self.fig.data[0].y = val_array
            self.fig.update_layout(title=f"{array_name} Along Branch {self.current_picked_skeleton_line_id}",
                                   yaxis_title=array_name)
        self.state.plot = self.fig.to_dict()
        self.state.flush()

    def on_mesh_click(self, cellId):
        """
        Handle right click event to pick a cell.
        """
        picked_cell_id = cellId
        print(f"Picked cell id: {picked_cell_id}")
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
            print(f"Local axis at picked point: {axis}")
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
        if self.skeleton_with_cells.GetCellType(0) == vtk.VTK_LINE:
            print("Using connected cells to define tangents along VTK_LINE skeleton")
            connected_closest_cells = vtk.vtkIdList()
            closest_point_coords = self.skeleton_points[closest_point_id]
            self.skeleton_with_cells.GetPointCells(closest_point_id, connected_closest_cells)
            for i in range(connected_closest_cells.GetNumberOfIds()):
                cell_id = connected_closest_cells.GetId(i)
                cell_point_ids = vtk.vtkIdList()
                self.skeleton_with_cells.GetCellPoints(cell_id, cell_point_ids)
                for j in range(cell_point_ids.GetNumberOfIds()):
                    if cell_point_ids.GetId(j) != closest_point_id:
                        tangent = np.array(self.skeleton_with_cells.GetPoint(cell_point_ids.GetId(j))) - closest_point_coords
                        break  # only need one tangent point from one cell
                tangents.append(tangent)
        else:
            # use 3 closest points to define tangents
            print("Using 3 closest points to define tangents")
            closest_point_ids = np.argsort(self.skeleton_to_pt_distances)[:3]
            print("Closest skeleton point IDs: ", closest_point_ids)
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
            skeleton_axis = tangents[0] - tangents[1]
        return skeleton_axis

    def show_local_slice(self, clicked_pt_coords, skeleton_axis):
        """
        Show a local slice of the object grid at the clicked position.
        """
        self.create_local_slice(click_coords=clicked_pt_coords,
                                                     skeleton_pt=clicked_pt_coords,
                                                     axis_vec=skeleton_axis)
        self.connectivity.Update()
        self.slice = self.connectivity.GetOutput()
        self.compute_normals_arrays(self.slice)
        self.compute_cell_area_array(self.slice)
        if self.doVelocityCalcs:
            Q=self.calculate_flow_rate(self.velocityName, self.slice, None)
            print(f"Volume Flow Rate Through Slice: {np.round(Q,2)}cm^3/s")
        for array_name in self.array_names.keys():
            # calculate average and total values on the slice for each array
            if array_name == "vWSS[dyn/cm^2]":
                continue  # skip WSS calculations on slice
            avg = self.calculate_value(array_name, self.slice, "average", None)
            total = self.calculate_value(array_name, self.slice, "total", None)
            print(f"Slice - Average {array_name}: {np.round(avg,2)}, Total {array_name}: {np.round(total,2)}")
        if self.doPressureCalcs:
            average_pressure = self.calculate_value(self.pressureName, self.slice, "average", None)
            print(f"Average Pressure on Slice: {np.round(average_pressure,2)}Pa")       
        self.surfaceActor.SetVisibility(True)
        # Make the aorta semi-transparent so you can see the slice inside
        self.sliceActor.SetVisibility(True)
        self.surfaceActor.GetProperty().SetOpacity(0.2)
        if self.state.selected_colour == "vWSS[dyn/cm^2]":
            self.state.selected_colour = "No Colouring"
            self.glyphActor.SetVisibility(False)
            self.state.flush()
        #if self.array_names.get(self.state.selected_colour) == "Vector":
        #    array_to_colour_by = f"{self.state.selected_colour}_{self.state.selected_component}"
        #else:
        #    array_to_colour_by = self.state.selected_colour
        #print("Colouring slice by:", array_to_colour_by)
        #self.set_colouring_by_dataset(array_to_colour_by, self.slice, self.sliceMapper)
        #self.slice.GetPointData().SetActiveScalars(array_to_colour_by)
        self.controller.view_update()
    
    def create_local_slice(self, click_coords, skeleton_pt, axis_vec):
        """Create a local slice of the object grid."""
        self.sliceMapper.StaticOff()  # allow scalar range to update for new slice
        self.plane.SetOrigin(skeleton_pt)
        self.plane.SetNormal(axis_vec)
        self.planeCutter.SetInputData(self.objectGrid)
        self.planeCutter.SetPlane(self.plane)
        self.connectivity.SetClosestPoint(click_coords)

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
        if self.doVelocityCalcs:
            print(f"Volume Flow Rate Through Face: {np.round(self.calculate_flow_rate(self.velocityName, self.objectSurface, visitedCells),2)}cm^3/s")

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
            else:
                normal = self.slice_cell_norms_np[cell_id]
                area = self.slice_cell_areas_np[cell_id]
            cell_point_ids = vtk.vtkIdList()
            dataset.GetCellPoints(cell_id, cell_point_ids)
            cell_pt_ids_np = np.array([cell_point_ids.GetId(i) for i in range(cell_point_ids.GetNumberOfIds())])
            v_dot_n = np.sum(vel_np_array[cell_pt_ids_np] * normal, axis=1) 
            Q += np.mean(v_dot_n) * area
        return Q

    def shade_surface_points(self, point_ids):
        """
        Shade selected surface points in red.
        """
        if self.glyphs_rendering is True:
            self.glyphActor.SetVisibility(False)
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
        self.surfaceMapper.SetScalarModeToUsePointData()
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

    def compute_cell_area_array(self, dataset=None):
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
        else:
            self.slice_cell_areas_np = np_cell_areas_arr

    def compute_normals_arrays(self, dataset=None):
        """
        Set up normals arrays for surface or slice.
        """
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(dataset)
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.ComputeCellNormalsOn()
        normalsFilter.SplittingOff()  # prevent normal splitting at sharp edges - keeps number of points consistent
        normalsFilter.Update()
        np_pt_norms_arr = numpy_support.vtk_to_numpy(normalsFilter.GetOutput().GetPointData().GetNormals())
        np_cell_norms_arr = numpy_support.vtk_to_numpy(normalsFilter.GetOutput().GetCellData().GetNormals())
        if dataset == self.objectSurface:
            self.surface_pt_norms_np = np_pt_norms_arr
            self.surface_cell_norms_np = np_cell_norms_arr
        else:
            self.slice_pt_norms_np = np_pt_norms_arr
            self.slice_cell_norms_np = np_cell_norms_arr

    def generate_gradient(self, quantity_name):
        """
        
        """
        gradientFilter = vtk.vtkGradientFilter()
        gradientFilter.SetInputData(self.objectGrid)
        gradientFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, quantity_name)
        gradientFilter.SetResultArrayName(f"{quantity_name}_gradient")
        gradientFilter.Update()
        # output of filter will be 3x3 gradient tensor, expressed as 1x9 vector
        self.objectGrid = gradientFilter.GetOutput()
        # print the length of the first point's gradient array to verify
        grad_array = self.objectGrid.GetPointData().GetArray(f"{quantity_name}_gradient")
        print(f"Generated {quantity_name} gradient array with {grad_array.GetNumberOfComponents()} components per point.")
    
    def find_wall_shear_stress(self, dataset):
        """
        Compute wall shear stress (WSS) on the object surface.
        """
        # Get velocity gradient array from object surface point data
        v_grad_vtk = dataset.GetPointData().GetArray(f"{self.velocityName}_gradient")
        v_grad_np = numpy_support.vtk_to_numpy(v_grad_vtk)
        # at each point, v_grad is expressed as 1x9 vector - reshape to 3x3 tensor
        v_grad_tensor = v_grad_np.reshape((-1, 3, 3))  # -1 infers number of points
        # compute matmul of tensor with normal vector - using einstein summation
        f = np.einsum('nij,nj->ni', v_grad_tensor, self.surface_pt_norms_np)
        f_dot_n = np.einsum('ni,ni->n', f, self.surface_pt_norms_np)
        # subtract normal component to get the tangential component only
        f_tangential = f - (f_dot_n[:, np.newaxis] * self.surface_pt_norms_np)
        wss_vectors = self.mu * f_tangential * 10  # factor of 10 converts to dyn/cm^2  # 1 dyn/cm^2 = 0.1 Pa
        # add WSS vector array to point data
        wss_vtk_array = numpy_support.numpy_to_vtk(wss_vectors, deep=True)
        wss_vtk_array.SetName("vWSS[dyn/cm^2]")
        dataset.GetPointData().AddArray(wss_vtk_array)
    
    def setup_shear_stress_glyphs(self):
        """
        Setup glyphs to visualise wall shear stress vectors.
        """
        arrow_source = vtk.vtkArrowSource()
        masker = vtk.vtkMaskPoints()  # for filtering points selectively
        masker.SetInputData(self.objectSurface)
        masker.SetOnRatio(100) # show every 100th point
        masker.RandomModeOn()
        masker.SingleVertexPerCellOn() # prevent multiple arrows per cell

        self.glyph.SetSourceConnection(arrow_source.GetOutputPort())
        self.glyph.SetInputConnection(masker.GetOutputPort())
        self.glyph.SetVectorModeToUseVector()  # use vector data for orientation and scaling
        self.glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vWSS[dyn/cm^2]")
        self.glyph.SetScaleModeToScaleByVector()  # scaling by tau magnitude
        self.glyph.OrientOn()  # make sure arrows are oriented along vector direction

        self.glyphMapper.SetInputConnection(self.glyph.GetOutputPort())

        self.glyphActor.SetMapper(self.glyphMapper)
        self.glyphMapper.ScalarVisibilityOff()  # disconnect from scalar colouring
        self.glyphActor.GetProperty().SetColor(1.0, 1.0, 1.0)
        # make glyph actor invisible by default
        self.glyphActor.SetVisibility(False)
       
    def setup_grids_arrays_actors(self):
        """
        Setup the grids, arrays, and actors for rendering.
        """
        if self.renderingObject:
            print("Setting up object surface and contours...")
            if self.velocityName:
                self.generate_gradient(self.velocityName)
                self.doVelocityCalcs = True
            if self.pressureName:
                self.generate_gradient(self.pressureName)
                self.doPressureCalcs = True
            self.setup_arrays_in_dataset(self.objectGrid)
            self.get_grid_surface()
            self.compute_normals_arrays(self.objectSurface)
            self.compute_cell_area_array(self.objectSurface)
            if self.velocityName:
                self.find_wall_shear_stress(self.objectSurface)
            self.setup_arrays_in_dataset(self.objectSurface)  # ensure WSS arrays are included
            for arr_name, arr_type in self.array_names.items():  # setup colour ranges
                if arr_type == "Vector":
                    self.colour_ranges[arr_name + "_x"] = [float('inf'), float('-inf')]
                    self.colour_ranges[arr_name + "_y"] = [float('inf'), float('-inf')]
                    self.colour_ranges[arr_name + "_z"] = [float('inf'), float('-inf')]
                    self.colour_ranges[arr_name + "_mag"] = [float('inf'), float('-inf')]
                else:
                    self.colour_ranges[arr_name] = [float('inf'), float('-inf')]
            if self.animating:  # also implies contouring
                print("Precomputing contours for all timesteps...")
                self.precompute_animation_surfaces()
            else:
                self.populate_colour_ranges(self.objectSurface)
            self.surface_points_coords = numpy_support.vtk_to_numpy(self.objectSurface.GetPoints().GetData())
            self.setup_scalar_bar()
            self.renderer.AddActor(self.surfaceActor)
            
            if self.velocityName:
                self.setup_shear_stress_glyphs()
                self.glyphs_rendering = True
                self.renderer.AddActor(self.glyphActor)
            
            # locator for locating cells
            self.surfaceCellLocator = vtk.vtkStaticCellLocator()
            print("Building cell locator for object surface...")
            self.surfaceCellLocator.SetDataSet(self.objectSurface)
            self.surfaceCellLocator.BuildLocator()

            if self.doProbing:
                print("Setting up probing along skeleton...")
                self.probe_along_skeleton()

        if self.doSlicing is True:
            print("Adding slice actor...")
            self.renderer.AddActor(self.sliceActor)

        if self.renderingPointCloud:
            print("Setting up point cloud layers...")
            self.setup_point_cloud_layers()

        if self.renderingStreamlines:
            print("Setting up streamlines...")
            self.add_streamlines()
            self.renderer.AddActor(self.streamlineActor)

        self.renderer.ResetCamera()
    
    def setup_skeleton_with_cells(self):
        """
        Setup a new vtkPolyData object for the skeleton that includes line cells connecting the points.
        """
        # iterate over each cell (polyline) and iterate over points in each cell starting from the maximum point index in the skeleton
        # for each consecutive point, create a vtkline cell connecting the two points and add it to a new vtkPolyData object representing the skeleton branches
        self.skeleton_with_cells.SetPoints(self.skeleton.GetPoints())
        new_lines = vtk.vtkCellArray()
        for i in range(self.skeleton.GetNumberOfCells()):
            cell = self.skeleton.GetCell(i)
            pids = cell.GetPointIds()
            for j in range(pids.GetNumberOfIds() - 1):
                line = vtk.vtkLine()
                # These IDs now correctly point to the original points array
                line.GetPointIds().SetId(0, pids.GetId(j))
                line.GetPointIds().SetId(1, pids.GetId(j+1))
                new_lines.InsertNextCell(line)
        self.skeleton_with_cells.SetLines(new_lines)
        print(f"Created new skeleton with {self.skeleton_with_cells.GetNumberOfCells()} line cells connecting the points.")
        print("Finished writing new skeleton with cells.")
    
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
        text = self.scalar_bar.GetLabelTextProperty()
        text.SetFontFamilyToArial()
        text.SetFontSize(14)
        text.BoldOn()
        text.ItalicOff()
        text.ShadowOff()
        text.SetColor(1, 1, 1)
        self.scalar_bar.GetTitleTextProperty().ShallowCopy(text)
        self.scalar_bar.SetLabelFormat("%.2f")
        self.renderer.AddActor(self.scalar_bar)
    
    def update_scalar_bar(self):
        """
        Update the scalar bar to reflect current colouring.
        """
        if self.state.selected_colour == "No Colouring":
            self.scalar_bar.SetVisibility(0)
            return
        self.scalar_bar.SetVisibility(1)
        self.scalar_bar.SetLookupTable(self.lut)
        #self.surfaceMapper.ScalarVisibilityOff()
        self.surfaceMapper.ScalarVisibilityOn()
    
    def setup_point_cloud_layers(self):
        """
        Precompute point cloud layers from image data.
        """
        if not self.renderingPointCloud:
            return
        image_range = self.image.GetScalarRange()
        scalarMin, scalarMax = image_range[0], image_range[1]
        jitter_amount = 0.1  # currently hardocded

        # levels
        self.min_level, self.max_level = 0, 100
        self.level_step = 5
        total_points = 0

        for i in range(self.max_level, self.min_level, -self.level_step):
            print(f"Generating point cloud layer for level {i}...")
            # reverse order so that higher intensity layers are on top
            low = scalarMin + ((i - 5) / 100.0) * (scalarMax - scalarMin)
            high = scalarMin + (i / 100.0) * (scalarMax - scalarMin)
            image_threshold = vtk.vtkThreshold()
            image_threshold.SetInputData(self.image)
            image_threshold.SetInputArrayToProcess(0,0,0,0,self.imageScalarName)
            image_threshold.SetLowerThreshold(low)
            image_threshold.SetUpperThreshold(high)
            image_threshold.Update()
            conversionPointCloud = vtk.vtkConvertToPointCloud()
            conversionPointCloud.SetInputConnection(image_threshold.GetOutputPort())
            conversionPointCloud.Update()
            pointCloud = conversionPointCloud.GetOutput()
            num_points = pointCloud.GetNumberOfPoints()
            total_points += num_points
            if total_points > self.totalPointCloudPoints:  # limit total points
                total_points -= num_points
                self.min_level = i
                break
            if num_points == 0:  # skip empty layers
                continue
            pointCloudPoints = numpy_support.vtk_to_numpy(pointCloud.GetPoints().GetData())
            jitters = np.random.normal(scale=jitter_amount, size=pointCloudPoints.shape)
            pointCloudPoints += jitters
            pointCloud.GetPoints().Modified()
            pointCloud.Modified()
            lookupTable = vtk.vtkLookupTable()
            lookupTable.SetNumberOfTableValues(256)
            lookupTable.SetTableRange(low, high)
            lookupTable.Build()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pointCloud)
            mapper.SetScalarVisibility(False)
            mapper.SetStatic(True)
            mapper.Update()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToPoints()
            actor.SetVisibility(0)
            self.renderer.AddActor(actor)
            self.pointCloudLayers[i//5] = actor  # store actor with level as key
            self.numPoints[i//5] = num_points
        sorted_keys = sorted(self.pointCloudLayers.keys(), reverse=True)
        for i, key in enumerate(sorted_keys):
            actor = self.pointCloudLayers[key]
            importance = (1 - self.numPoints[key] / total_points)**2
            # define color/opacity distribution based on importance
            dist = 0.1 + 0.9 * importance
            actor.GetProperty().SetColor(dist, dist, dist)
            actor.GetProperty().SetOpacity(dist)
            actor.GetProperty().SetPointSize(dist)

    def set_image_level(self, level):
        """
        Shows all point cloud layers up to level.
        """
        sorted_keys = sorted(self.pointCloudLayers.keys(), reverse=True)
        for i, key in enumerate(sorted_keys):
            actor = self.pointCloudLayers[key]
            actor.SetVisibility(i < level)
    
    def clear_slice_and_selection(self):
        """
        Clear any existing slice and selection colouring, resetting to default view.
        """
        print("Clearing slice and selection colouring.")
        # reset opacity and hide slice
        self.sliceActor.SetVisibility(False)
        self.surfaceActor.GetProperty().SetOpacity(1.0)
        self.set_colouring_by_dataset("No Colouring", self.objectSurface, self.surfaceMapper)
        self.controller.view_update()
    
    def clear_plot(self):
        """
        Clear any existing plot data.
        """
        print("Clearing plot data.")
        self.probe_ids = np.array([], dtype=int)
        for sphere_source in self.sphereSources:
            sphere_source.actor.SetVisibility(0)
            self.renderer.RemoveActor(sphere_source.actor)
        self.sphereSources = []
        with self.fig.batch_update():
            self.fig.data[0].x = []
            self.fig.data[0].y = []
        self.state.plot = self.fig.to_dict()
        self.state.flush()
        self.controller.view_update()
    
    def on_skeleton_click(self, cell_id):
        """Handler for Skeleton Click"""
        print(f"Clicked on skeleton cell ID: {cell_id}")

    def on_click(self, event):
        """Handler for Click (Button 2)"""
        if not event:
            return
        p1, p2 = event["ray"][0], event["ray"][1]
        if self.state.do_picking:
            t = vtk.mutable(0)
            x = [0.0, 0.0, 0.0]
            pcoords = [0, 0, 0]
            subId = vtk.mutable(0)
            cellId = vtk.mutable(0)
            hit = self.surfaceCellLocator.IntersectWithLine(p1, p2, 0.001, t, x, pcoords, subId, cellId)
            if hit:
                self.on_mesh_click(cellId)
        elif self.state.show_plot:
            t = vtk.mutable(0)
            x = [0.0, 0.0, 0.0]
            pcoords = [0, 0, 0]
            subId = vtk.mutable(0)
            cellId = vtk.mutable(0)
            hit = self.skeletonCellLocator.IntersectWithLine(p1, p2, 0.1, t, x, pcoords, subId, cellId)
            if hit:
                print(f"Clicked on skeleton cell ID: {cellId}")
                self.current_picked_skeleton_line_id = cellId
                self.plot_probed_data()
                #cell = self.skeleton.GetCell(cellId)
                #cell_points = cell.GetPoints()
                ## find average of cell points for sphere position
                #num_cell_points = cell_points.GetNumberOfPoints()
                #avg_point = np.array([0.0, 0.0, 0.0])
                #for i in range(num_cell_points):
                #    pt = cell_points.GetPoint(i)
                #    avg_point += np.array(pt)
                #if len(self.probe_ids) < 2:
                #    self.probe_ids = np.append(self.probe_ids, int(cell.GetPointId(0)))  # store first point id of cell for probing
                #    self.sphereSources.append(SphereSourceCustom())
                #else:
                #    self.probe_ids[-1] = int(cell.GetPointId(0))
                #self.plot_probed_data()
                #sphere_source = self.sphereSources[-1]
                #sphere_source.actor.SetPosition(avg_point/num_cell_points)
                #sphere_source.actor.SetVisibility(1)
                #self.renderer.AddActor(sphere_source.actor)
                self.controller.view_update()
        else:
            return

    def render(self):
        """Return a Trame LocalView for the render window."""
        self.view = trame_vtk.VtkLocalView(self.render_window,
                                  picking_modes=("picking_modes", []),
                                  click=(self.on_click, "[$event]"),
                                  )
        self.controller.view_update = self.view.update

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    
    config = load_config(sys.argv[1])

    mesh_path = config.get("mesh_path")
    image_path = config.get("image_path")
    mesh_extraction_array_name = config.get("mesh_extraction_array_name")
    mesh_extraction_value = config.get("mesh_extraction_value")
    image_scalar_name = config.get("image_scalar_name")
    velocity_array_name = config.get("velocity_array_name")
    pressure_array_name = config.get("pressure_array_name")
    mesh_target_num_points = config.get("mesh_target_num_points")
    streamline_seed_pt_idx = config.get("streamline_seed_pt_idx")
    animation_enabled = config.get("animation_enabled")
    initial_timestep = config.get("initial_timestep")
    final_timestep = config.get("final_timestep")
    update_time = config.get("update_time")
    skeleton_path = config.get("skeleton_path")
    mu = config.get("dynamic_viscosity")

    print("Mesh Path:", mesh_path)
    print("Image Path:", image_path)
    print("Mesh Extraction Array Name:", mesh_extraction_array_name)
    print("Mesh Extraction Value:", mesh_extraction_value)
    print("Image Scalar Name:", image_scalar_name)
    print("Velocity Array Name:", velocity_array_name)
    print("Pressure Array Name:", pressure_array_name)
    print("Mesh Target Number of Points:", mesh_target_num_points)
    print("Streamline Seed Point Index:", streamline_seed_pt_idx)
    print("Animation Enabled:", animation_enabled)
    print("Initial Timestep:", initial_timestep)
    print("Final Timestep:", final_timestep)
    print("Update Time:", update_time)
    print("Skeleton Path:", skeleton_path)
    print("Dynamic Viscosity (mu):", mu)

    server = get_server()
    state = server.state
    controller = server.controller

    viewer = VTUViewer(state, controller, mesh_path,
                        image_path,
                        mesh_extraction_array_name,
                        mesh_extraction_value,
                        image_scalar_name,
                        mesh_target_num_points,
                        streamline_seed_pt_idx,
                        animation_enabled,
                        initial_timestep,
                        final_timestep,
                        update_time,
                        velocity_array_name,
                        pressure_array_name,
                        skeleton_path,
                        mu)

    state.colour_options = ["No Colouring"] 
    state.selected_colour = "No Colouring"
    state.selected_component = "mag"
    state.component_options = ["x", "y", "z", "mag"]
    state.representation_options = ["Surface", "Wireframe", "Points"]  # Surface, Wireframe, Points
    state.representation = "Surface"  # default to Surface
    state.animation = False
    state.show_streamlines = True
    state.show_plot = False
    state.vWSS_glyph_scale = 0.005
    state.picking_modes = ["click"]
    state.do_picking = False
    state.show_plot = False
    state.fig = viewer.fig.to_dict()
    if viewer.renderingPointCloud:
        state.image_level = len(viewer.pointCloudLayers) // 2

    if viewer.animating:
        # Speed settings to perform linear mapping
        MIN_TS = 0.01   # fastest
        MAX_TS = 0.31   # slowest
        state.max_speed = 0.3  # max speed limit  # NOTE: DISTANCE BETWEEN MAX AND MIN SPEED SHOULD BE MAX_TS - MIN_TS
        state.min_speed = 0.0  # min speed limit
        state.speed_step = 0.01  # speed adjustment step
        state.speed = state.max_speed - viewer.update_ts  # inverse mapping

    for name in viewer.array_names.keys():
        state.colour_options.append(name)
    
    @state.change("selected_colour")
    def update_colour(selected_colour, **kwargs):
        if viewer.glyphs_rendering is True and selected_colour == "vWSS[dyn/cm^2]":
            viewer.glyphActor.SetVisibility(1)
        else:
            viewer.glyphActor.SetVisibility(0)
        if viewer.array_names.get(state.selected_colour) == "Vector":
            array_to_colour_by = f"{state.selected_colour}_{state.selected_component}"
            state.component_options = ["x", "y", "z", "mag"]
        else:
            array_to_colour_by = state.selected_colour 
            state.component_options = ["mag"]
            state.selected_component = "mag"
        state.flush()
        #viewer.plot_probed_data()
        #if viewer.sliceActor.GetVisibility() == 1 and selected_colour != "vWSS[dyn/cm^2]":
        #    viewer.set_colouring_by_dataset(array_to_colour_by, viewer.slice, viewer.sliceMapper)
        viewer.set_colouring_by_dataset(array_to_colour_by, viewer.objectSurface, viewer.surfaceMapper)
        viewer.update_scalar_bar()
        controller.view_update()

    @state.change("selected_component")
    def update_component(selected_component, **kwargs):
        # if selected colouring is a vector field, append component suffix
        if viewer.array_names.get(state.selected_colour) == "Vector":
            array_to_colour_by = f"{state.selected_colour}_{selected_component}"
        else:
            array_to_colour_by = state.selected_colour        
        #if viewer.sliceActor.GetVisibility() == 1 and state.selected_colour != "vWSS[dyn/cm^2]":
        #    viewer.set_colouring_by_dataset(array_to_colour_by, viewer.slice, viewer.sliceMapper)
        #viewer.plot_probed_data()
        viewer.set_colouring_by_dataset(array_to_colour_by, viewer.objectSurface, viewer.surfaceMapper)
        viewer.update_scalar_bar()
        controller.view_update()  

    @state.change("representation")
    def update_repr(representation, **kwargs):
        viewer.set_representation(representation)
        controller.view_update()
    
    @state.change("image_level", debounce=250)
    def update_image_level(image_level, **kwargs):
        if viewer.renderingPointCloud:
            viewer.set_image_level(image_level)
            if state.animation is False:
                controller.view_update()

    @state.change("animation", debounce=250)
    def toggle_animation(animation, **kwargs):
        """Called by a button to start/stop"""
        if state.animation is True:
            # Start the background task without blocking Python
            asyncio.create_task(viewer.animate())
    
    @state.change("speed")
    def on_speed_change(speed, **kwargs):
        normalised_ts = (speed - state.min_speed) / (state.max_speed - state.min_speed)  # map speed to 0-1
        ts = MAX_TS - normalised_ts * (MAX_TS - MIN_TS)  # inverse mapping
        viewer.update_ts = np.round(ts, 2)
        print("Updated timestep to: ", viewer.update_ts)
    
    def speed_down():
        state.speed -= state.speed_step
        if state.speed < state.min_speed:
            state.speed = state.min_speed
        state.flush()

    def speed_up():
        state.speed += state.speed_step
        if state.speed > state.max_speed:
            state.speed = state.max_speed
        state.flush()
    
    @state.change("show_streamlines")
    def toggle_streamlines(show_streamlines, **kwargs):
        if viewer.renderingStreamlines is True:
            if show_streamlines:
                viewer.streamlineActor.SetVisibility(1)
            else:
                viewer.streamlineActor.SetVisibility(0)
            controller.view_update()

    @state.change("vWSS_glyph_scale", debounce=250)
    def update_glyph_scale(vWSS_glyph_scale, **kwargs):
        if viewer.glyphs_rendering is False:
            return
        viewer.glyph.SetScaleFactor(vWSS_glyph_scale)
        viewer.glyphMapper.Update()
        controller.view_update()

    @state.change("show_plot")
    def toggle_plot(show_plot, **kwargs):
        viewer.surfaceActor.GetProperty().SetOpacity(
            0.3 if show_plot else 1.0
        )
        viewer.sliceActor.SetVisibility(0)
        viewer.skeletonActor.SetVisibility(1 if show_plot else 0)
        if show_plot:
            state.do_picking = False
            state.flush()  # ensure picking checkbox updates in GUI
            viewer.scalar_bar.SetVisibility(0)
            #if viewer.doProbing:
            #    for sphere in viewer.sphereSources:
            #        sphere.actor.SetVisibility(1)
        #else:
        #    if viewer.doProbing:
        #        for sphere in viewer.sphereSources:
        #            sphere.actor.SetVisibility(0)
        controller.view_update()
    
    @state.change("do_picking")
    def toggle_picking(do_picking, **kwargs):
        if do_picking:
            viewer.surfaceActor.GetProperty().SetOpacity(1.0)
            if state.selected_colour != "No Colouring":
                viewer.scalar_bar.SetVisibility(1)
            state.show_plot = False
            state.flush()  # ensure plot checkbox updates in GUI
        controller.view_update()

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
        if viewer.renderingPointCloud:  # only show if image data is available
            with layout.toolbar:
                v3.VSpacer()
                v3.VSlider(
                    v_model="image_level",
                    min=0,
                    max=len(viewer.pointCloudLayers),
                    step=1,
                    hide_details=True,
                    dense=True,
                    style="max-width: 300px",
                    label="Detail Level"
                )
        if viewer.animating:
            with layout.toolbar:
                v3.VSpacer()
                v3.VCheckbox(
                    v_model="animation",
                    label="Live Update",
                    hide_details=True,
                    dense=True,
                )
                v3.VBtn("-", click=speed_down),
                v3.VSlider(
                    v_model="speed",
                    min=state.min_speed,
                    max=state.max_speed,
                    step=state.speed_step,
                    hide_details=True,
                    style="width: 200px",
                    label="Speed"
                ),
                v3.VBtn("+", click=speed_up),
        if viewer.renderingStreamlines is True:
            with layout.toolbar:
                v3.VSpacer()
                v3.VCheckbox(
                    v_model="show_streamlines",
                    label="Show Streamlines",
                    hide_details=True,
                    dense=True,
                )
        if viewer.velocityName:
            with layout.toolbar:
                v3.VSpacer()
                v3.VSlider(
                    v_model="vWSS_glyph_scale",
                    min=0.0,
                    max=0.2,
                    step=0.005,
                    hide_details=True,
                    dense=True,
                    style="max-width: 300px",
                    label="Glyph Scale"
                )
        with layout.toolbar:
            v3.VSpacer()
            v3.VCheckbox(
                v_model="do_picking",
                label="Enable Picking",
                hide_details=True,
                dense=True,
            )
        with layout.toolbar:
            v3.VSpacer()
            v3.VBtn(
                "Clear",
                dense=True,
                click=viewer.clear_slice_and_selection,
            )
        with layout.toolbar:
            v3.VSpacer()
            v3.VCheckbox(
                v_model="show_plot",
                label="Toggle plot",
                hide_details=True,
                dense=True,
            )
        # VTK view content
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height", style="position: relative;"):
                viewer.render()
                if viewer.doProbing:
                    with v3.VCard(
                        style="""
                            position: fixed;
                            bottom: 20px;
                            right: 20px;
                            width: 25vw;
                            height: 40vh;
                            z-index: 1000;
                        """,
                        elevation=10,
                        v_show="show_plot",
                    ):
                        plotly.Figure(figure=viewer.fig, style="width: 100%; height: 100%;", state_variable_name="plot")
                with v3.VCard(
                    style="""
                        position: fixed;
                        bottom: 20px;
                        left: 20px;
                        width: 25vw;
                        height: 40vh;
                        z-index: 1000;
                    """,
                    elevation=10,
                    v_show="show_plot",
                ):
                    plotly.Figure(figure=viewer.scatter3D.fig, style="width: 100%; height: 100%;")

    # Start server
    server.start(host="127.0.0.1", port=1234)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    main()