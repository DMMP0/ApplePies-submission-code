import os
import shutil

paths = os.listdir('./')

for path in paths:
	if os.path.isdir(path):
		new_name = path[1:] if path[0] == '.' else path
		shutil.move(path, new_name)
