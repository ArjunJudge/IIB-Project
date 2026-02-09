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
        if image_path and not self.imageScalarName:
            raise ValueError("No image scalar name specified.")
        self.state = state
        self.controller = controller
        self.mu = mu
        if not self.mu:
            print("No dynamic viscosity specified, defaulting to 0.004")
            self.mu = 0.004

        """ OBJECT ACTORS, FILTERS AND MAPPERS SETUP """
        if mesh_path:
            self.objectGrid = vtk.vtkUnstructuredGrid()
            self.decimateFilter = vtk.vtkDecimatePro()
            self.extractionMapper = vtk.vtkDataSetMapper()
            self.extractionActor = vtk.vtkActor()
            self.extractionMapper.SetScalarModeToUsePointFieldData()
            self.extractionMapper.SetColorModeToMapScalars()
            self.extractionActor.SetMapper(self.extractionMapper)
            self.extractionActor.GetProperty().SetColor(0.0, 1.0, 0.0)  # green
            self.extractionActor.SetVisibility(True)
            self.contourFilter = vtk.vtkContourFilter()
            self.clipFilter = vtk.vtkClipDataSet()
            self.surfaceExtractionFilter = vtk.vtkDataSetSurfaceFilter()
            self.meshExtractionArrayName = mesh_extraction_array_name
            self.meshTargetNumPoints = mesh_target_num_points if mesh_target_num_points else 25000
            self.meshExtractionValue = mesh_extraction_value
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
            self.extractionMapper.SetLookupTable(self.lut)
            self.colour_ranges = {}
            self.scalar_bar = vtk.vtkScalarBarActor()
            self.scalar_bar.SetLookupTable(self.lut)
            self.scalar_bar.SetVisibility(True)

        """ IMAGE DATA SETUP """
        if image_path:
            self.pointCloudLayers = {}
            self.numPoints = {}  # to store number of points in each layer
            self.totalPointCloudPoints = 1e6 if animation_enabled is True else 2e6  # limit total points based on animation

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

        """ RENDERING SETTINGS AND FILE READING """
        self.renderingObject, self.renderingPointCloud, self.renderingStreamlines = False, False, False
        if mesh_path:
            self.renderingObject = True
            self.file_reader(mesh_path)
        if image_path:
            self.renderingPointCloud = True
            self.file_reader(image_path)
        if streamline_seed_pt_idx is not None and self.velocityName:
            self.renderingStreamlines = True
        if skeleton_path:
            self.file_reader(skeleton_path)
            self.skeleton_points = numpy_support.vtk_to_numpy(self.skeleton.GetPoints().GetData())
            self.skeleton_to_pt_distances= np.zeros(self.skeleton.GetNumberOfPoints())
        else:
            print("No skeleton file provided - skipping skeleton probing.")
            pass

        self.setup_grids_arrays_actors()
    
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
    
    def set_colouring_by_dataset(self, data_set_array_name, mapper):
        """
        Set the colouring of the mapper based on the specified dataset array name.
        """
        if data_set_array_name == "No Colouring":
            mapper.ScalarVisibilityOff()
            return
        scalar_range = self.colour_ranges[data_set_array_name] 
        mapper.SetScalarRange(scalar_range)
        self.lut.SetRange(scalar_range)
        self.lut.Build()  # Ensure LUT is built with new range
        mapper.SelectColorArray(data_set_array_name)
        mapper.ScalarVisibilityOn()
        self.scalar_bar.SetTitle(data_set_array_name)
    
    async def animate(self):
        """Animate the render window by updating it periodically."""
        while self.state.animation:
            self.extractionMapper.SetInputData(self.cached_contours[self.timestep - 1])
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
            self.extractionMapper.SetInputConnection(self.decimateFilter.GetOutputPort())
            self.extractionMapper.Update()
        else:  # not animating
            if self.meshExtractionArrayName is not None:  # clip to get surface if extraction array given
                self.clipFilter.SetInputData(self.objectGrid)
                self.clipFilter.SetInputArrayToProcess(
                    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, array_name
                )
                self.clipFilter.SetValue(self.meshExtractionValue)
                self.clipFilter.InsideOutOn()
                self.clipFilter.Update()
                self.surfaceExtractionFilter.SetInputData(self.clipFilter.GetOutput())
            else:  # otherwise get full surface
                self.surfaceExtractionFilter.SetInputData(self.objectGrid)
            self.surfaceExtractionFilter.Update()
            self.extractionMapper.SetInputConnection(self.surfaceExtractionFilter.GetOutputPort())
            self.extractionMapper.Update()
        self.objectSurface = self.extractionMapper.GetInput()
    
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
        property = self.extractionActor.GetProperty()
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

    def compute_normals_arrays(self, dataset):
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
        self.surface_pt_norms_np = np_pt_norms_arr
        self.surface_cell_norms_np = np_cell_norms_arr

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

        self.glyph = vtk.vtkGlyph3D()  # glyph is customisable shape at each point - in this case arrows
        self.glyph.SetSourceConnection(arrow_source.GetOutputPort())
        self.glyph.SetInputConnection(masker.GetOutputPort())
        self.glyph.SetVectorModeToUseVector()  # use vector data for orientation and scaling
        self.glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vWSS[dyn/cm^2]")
        self.glyph.SetScaleModeToScaleByVector()  # scaling by tau magnitude
        self.glyph.OrientOn()  # make sure arrows are oriented along vector direction

        self.glyph_mapper = vtk.vtkPolyDataMapper()
        self.glyph_mapper.SetInputConnection(self.glyph.GetOutputPort())
        self.glyph_actor = vtk.vtkActor()
        self.glyph_actor.SetMapper(self.glyph_mapper)
        self.glyph_mapper.ScalarVisibilityOff()  # disconnect from scalar colouring
        self.glyph_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        # make glyph actor invisible by default
        self.glyph_actor.SetVisibility(False)
       
    def setup_grids_arrays_actors(self):
        """
        Setup the grids, arrays, and actors for rendering.
        """
        if self.renderingObject:
            print("Setting up object surface and contours...")
            if self.velocityName:
                self.generate_gradient(self.velocityName)
            if self.pressureName:
                self.generate_gradient(self.pressureName)
            self.get_grid_surface()
            self.compute_normals_arrays(self.objectSurface)
            if self.velocityName:
                self.find_wall_shear_stress(self.objectSurface)
            self.setup_arrays_in_dataset(self.objectSurface)  # ensure WSS arrays are included
            if self.velocityName:
                if not (self.velocityName in self.array_names.keys() and self.array_names[self.velocityName] == "Vector"):
                    print(f"Velocity array '{self.velocityName}' not found in object surface.")
                    self.renderingStreamlines = False  # cannot render streamlines without velocity field
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
            self.setup_shear_stress_glyphs()
            self.setup_scalar_bar()
            self.renderer.AddActor(self.extractionActor)
            self.renderer.AddActor(self.glyph_actor)

        if self.renderingPointCloud:
            print("Setting up point cloud layers...")
            self.setup_point_cloud_layers()

        if self.renderingStreamlines:
            print("Setting up streamlines...")
            self.add_streamlines()
            self.renderer.AddActor(self.streamlineActor)

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
        self.extractionMapper.ScalarVisibilityOff()
        self.extractionMapper.ScalarVisibilityOn()
        self.extractionMapper.Modified()
        self.controller.view_update()
    
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
            total_points += conversionPointCloud.GetOutput().GetNumberOfPoints()
            if total_points > self.totalPointCloudPoints:  # limit total points
                total_points -= conversionPointCloud.GetOutput().GetNumberOfPoints()
                self.min_level = i
                break
            if conversionPointCloud.GetOutput().GetNumberOfPoints() == 0:  # skip empty layers
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
            self.numPoints[i//5] = conversionPointCloud.GetOutput().GetNumberOfPoints()
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

    def render(self):
        """Return a Trame LocalView for the render window."""
        view = trame_vtk.VtkLocalView(self.render_window)  # NOW LOCAL VIEW - MUCH FASTER
        view.enable_interaction = True  # enable rotation, pan, zoom
        self.controller.view_update = view.update
        self.view = view


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

    state.show_image = True
    state.animation = False
    state.show_streamlines = True
    state.show_plot = False
    state.vWSS_glyph_scale = 0.005

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
        if viewer.array_names.get(selected_colour) == "Vector":
            array_to_colour_by = f"{selected_colour}_{state.selected_component}"
            state.component_options = ["x", "y", "z", "mag"]
            state.flush()  # ensure component selection updates in GUI
        else:
            array_to_colour_by = selected_colour 
            state.component_options = ["mag"]
            state.selected_component = "mag"
            state.flush()  # ensure component selection updates in GUI
        viewer.set_colouring_by_dataset(
            array_to_colour_by,
            viewer.extractionMapper
        )
        viewer.update_scalar_bar()
        # if colouring by WSS, enable glyphs
        if selected_colour == "vWSS[dyn/cm^2]":
            if viewer.renderingStreamlines is True:
                viewer.streamlineActor.SetVisibility(0)  # hide streamlines to reduce clutter
                state.show_streamlines = False
                state.flush()
            viewer.glyph_actor.SetVisibility(1)
        else:
            viewer.glyph_actor.SetVisibility(0)
        controller.view_update()

    @state.change("selected_component")
    def update_component(selected_component, **kwargs):
        # if selected colouring is a vector field, append component suffix
        if viewer.array_names.get(state.selected_colour) == "Vector":
            array_to_colour_by = f"{state.selected_colour}_{selected_component}"
        else:
            array_to_colour_by = state.selected_colour        
        viewer.set_colouring_by_dataset(
            array_to_colour_by,
            viewer.extractionMapper
        )
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
                viewer.view.update()
                viewer.render_window.Render()

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

    @state.change("show_plot")
    def toggle_plot(show_plot, **kwargs):
        if show_plot is True:
            viewer.extractionActor.GetProperty().SetOpacity(0.3)  # make semi-transparent to see skeleton
        else:
            viewer.extractionActor.GetProperty().SetOpacity(1.0)  # reset opacity
        controller.view_update()
    
    @state.change("vWSS_glyph_scale", debounce=250)
    def update_glyph_scale(vWSS_glyph_scale, **kwargs):
        viewer.glyph.SetScaleFactor(vWSS_glyph_scale)
        viewer.glyph_mapper.Update()
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
        # VTK view content
        with layout.content:
            view = viewer.render()

    # Start server
    server.start(host="0.0.0.0", port=1234)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    main()