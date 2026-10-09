import FreeCAD
import Mesh
import Part


def main() -> None:
    import json
    import os
    import time

    started = time.perf_counter()
    result = {"ok": False, "version": ".".join(FreeCAD.Version()[:3])}
    try:
        mesh = Mesh.Mesh(os.environ["UNMESH_FC_IN"])
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh.Topology, float(os.environ["UNMESH_FC_TOLERANCE"]))
        solid = Part.Solid(Part.Shell(shape.Faces)).removeSplitter()
        solid.exportStep(os.environ["UNMESH_FC_OUT"])
        kinds = {}
        for face in solid.Faces:
            name = type(face.Surface).__name__
            kinds[name] = kinds.get(name, 0) + 1
        result.update(
            ok=True,
            faces=len(solid.Faces),
            surfaces=kinds,
            valid=solid.isValid(),
            volume=solid.Volume,
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    result["seconds"] = time.perf_counter() - started
    print("RESULT " + json.dumps(result), flush=True)


main()
