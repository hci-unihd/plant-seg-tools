import trimesh
from trimesh.exchange import ply
from trimesh.smoothing import filter_laplacian


class CreateTriMesh:
    def __init__(self, reduction, smoothing):
        self.reduction = reduction
        self.smoothing = smoothing

    def __call__(self, vertx, faces, normals):
        mesh = trimesh.Trimesh(vertices=vertx, faces=faces, vertex_normals=normals)

        if self.reduction > 0:
            mesh = mesh.simplify_quadratic_decimation(len(faces) * self.reduction)

        if self.smoothing > 0:
            mesh = filter_laplacian(mesh, iterations=self.smoothing)

        return mesh


def create_ply(path, mesh):
    mesh_ply = ply.export_ply(mesh)
    with open(path, "wb") as f:
        # write ply to file
        f.write(mesh_ply)
