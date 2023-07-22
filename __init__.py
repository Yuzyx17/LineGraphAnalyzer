import sys
import pathlib
current_path = pathlib.Path(__file__).parent.resolve()

sys.path.append(current_path)
print("HELLO")