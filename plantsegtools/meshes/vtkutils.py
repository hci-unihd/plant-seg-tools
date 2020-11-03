from vtk import vtkPolyData, vtkCellArray, vtkPoints, vtkPolygon
from vtk import vtkPLYWriter, vtkDecimatePro, vtkSmoothPolyDataFilter, vtkPolyDataNormals
import numpy as np


def ndarray2vtkMesh(inVertexArray, inFacesArray):
    ''' Code inspired by https://github.com/selaux/numpy2vtk '''
    # Handle the points & vertices:
    z_index = 0
    vtk_points = vtkPoints()
    for p in inVertexArray:
        z_value = p[2] if inVertexArray.shape[1] == 3 else z_index
        vtk_points.InsertNextPoint([p[0], p[1], z_value])
    number_of_points = vtk_points.GetNumberOfPoints()

    indices = np.array(range(number_of_points), dtype=np.int)
    vtk_vertices = vtkCellArray()
    for v in indices:
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(v)

    # Handle faces
    number_of_polygons = inFacesArray.shape[0]
    poly_shape = inFacesArray.shape[1]
    vtk_polygons = vtkCellArray()
    for j in range(0, number_of_polygons):
        polygon = vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(poly_shape)
        for i in range(0, poly_shape):
            polygon.GetPointIds().SetId(i, inFacesArray[j, i])
        vtk_polygons.InsertNextCell(polygon)

    # Assemble the vtkPolyData from the points, vertices and faces
    poly_data = vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetVerts(vtk_vertices)
    poly_data.SetPolys(vtk_polygons)

    return poly_data


def decimation(vtkPoly, reduction=0.25):
    # decimate and copy data
    decimate = vtkDecimatePro()
    decimate.SetInputData(vtkPoly)
    decimate.SetTargetReduction(reduction)  # (float) set 0 for no reduction and 1 for 100% reduction
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    return decimatedPoly


def smooth(vtkPoly, iterations=100, relaxation=0.1, edgesmoothing=True):
    # Smooth mesh with Laplacian Smoothing
    smooth = vtkSmoothPolyDataFilter()
    smooth.SetInputData(vtkPoly)
    smooth.SetRelaxationFactor(relaxation)
    smooth.SetNumberOfIterations(iterations)
    if edgesmoothing:
        smooth.FeatureEdgeSmoothingOn()
    else:
        smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    smoothPoly = vtkPolyData()
    smoothPoly.ShallowCopy(smooth.GetOutput())

    # Find mesh normals (Not sure why)
    normal = vtkPolyDataNormals()
    normal.SetInputData(smoothPoly)
    normal.ComputePointNormalsOn()
    normal.ComputeCellNormalsOn()
    normal.Update()

    normalPoly = vtkPolyData()
    normalPoly.ShallowCopy(normal.GetOutput())

    return normalPoly


class CreateMeshVTK:
    def __init__(self, reduction=0, smoothing=False):
        self.reduction = reduction
        self.smoothing = smoothing

    def __call__(self, vertx, faces, normals):
        vtk_poly = ndarray2vtkMesh(vertx, faces.astype(int))

        if self.reduction > 0:
            vtk_poly = decimation(vtk_poly, reduction=self.reduction)

        if self.smoothing:
            vtk_poly = smooth(vtk_poly)

        return vtk_poly


def create_ply(savepath, vtkPoly):
    # write results to output file

    writer = vtkPLYWriter()
    writer.SetInputData(vtkPoly)
    writer.SetFileName(savepath)
    writer.Write()
