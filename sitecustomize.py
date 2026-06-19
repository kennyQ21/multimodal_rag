import os
import sys

if sys.platform == "win32":
    try:
        import paddle

        libs = os.path.join(
            os.path.dirname(paddle.__file__),
            "libs"
        )

        if os.path.exists(libs):
            os.add_dll_directory(libs)

    except Exception:
        pass
