import sys

import FreeCAD  # noqa: F401
import Part


def check(paths):
    for path in paths:
        shape = Part.Shape()
        shape.read(path)
        try:
            errors = shape.check(True)
        except Exception as error:  # FreeCAD raises on a failed BOP check
            errors = sorted(
                {
                    line.strip()
                    for line in str(error).splitlines()
                    if "Invalid" in line or "Error" in line
                }
            )
        print(
            f"{path}: isValid={shape.isValid()} faces={len(shape.Faces)} "
            f"volume={shape.Volume:.6g} check={errors if errors else 'clean'}"
        )


check(sys.argv[2:])
