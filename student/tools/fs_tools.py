from pathlib import Path

def read_file(filepath, start_line, end_line):
	"""
	Read the content of a file with line numbers
	"""
	if end_line < start_line:
		print("ERROR: end line before start line")
		return
	try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            print(f"ERROR: The file at: '{filepath}' is empty.")
            return
        total_lines = len(lines)
		if total_lines < start_line:
			print("ERROR: Start line is greater than the total lines")
			return
		elif total_lines < end_line:
			print("ERROR: End line is greater than the total lines")
			return
		expected_lines = lines[start_line - 1 : end_line]
		output = []
		for line_num, line_content in enumerate(expected_lines, start=start_line):
			output.append(f"{line_num}: {line_content.rstrip()}")
    	return "\n".join(output)
    except FileNotFoundError:
        print(f"ERROR : The file at '{filepath}' can't be found.")
    except PermissionError:
        print(f"ERROR : You do not have access to the file at: '{filepath}'.")
    except UnicodeDecodeError:
        print(f"ERROR : File not readable at: '{filepath}'.")


def edit_file(filepath, old_str, new_str):
	"""
	Replace an exact string in a file with a new string.
	"""
	if old_str == "":
		print("ERROR: old_str cannot be empty.")
		return
	try:
        with open(filepath, 'r', encoding='utf-8') as f:
			content = f.read()

			occurrence_count = content.count(old_str)

		    if occurrence_count == 0:
				print(f"ERROR: {old_str} has not been found in the file at {filepath}")
				return
			elif occurrence_count > 1:
				print(f"ERROR: '{old_str}' has been found {occurrence_count} times and not only one.")
			else:
				updated_content = content.replace(old_str, new_str)

				with open(filepath, 'w', encoding='utf-8') as f:
					f.write(updated_content)
					return
	except FileNotFoundError:
		 print(f"ERROR : The file at '{filepath}' can't be found.")
    except PermissionError:
        print(f"ERROR : You do not have access to the file at: '{filepath}'.")
    except UnicodeDecodeError:
        print(f"ERROR : File not readable at: '{filepath}'.")


EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "node_modules", "dist", "build"}
MAX_RESULTS = 100


def list_files(directory, pattern):
    """
    Recursively list files in a directory matching a given shell-style pattern,
    returning deterministically sorted absolute paths.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"ERROR: The directory '{directory}' does not exist.")
        return
    if not dir_path.is_dir():
        print(f"ERROR: The path '{directory}' is not a directory.")
        return

    try:
        matched_files = []

        for path in dir_path.rglob(pattern):
            if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
                continue

            if path.is_file():
                matched_files.append(path.resolve())

        matched_files = sorted(matched_files)
        total_count = len(matched_files)

        if total_count == 0:
            print(f"No files matching pattern '{pattern}' were found in '{directory}'.")
            return

        truncated = False
        if total_count > MAX_RESULTS:
            matched_files = matched_files[:MAX_RESULTS]
            truncated = True

        output = [str(p) for p in matched_files]

        if truncated:
            output.append(f"... (truncated: showing {MAX_RESULTS} of {total_count} files)")

        return "\n".join(output)
        
    except FileNotFoundError:
        print(f"ERROR : The directory at '{directory}' can't be found.")
    except PermissionError:
        print(f"ERROR : You do not have access to the directory at: '{directory}'.")
    except Exception as e:
        print(f"ERROR : {str(e)}")