import os
import subprocess
import shutil

WORKSPACE_ROOT = "c:/vernex"

def read_file(path):
    full_path = os.path.join(WORKSPACE_ROOT, path)
    if not os.path.exists(full_path):
        return f"Error: File {path} not found."
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path, content):
    full_path = os.path.join(WORKSPACE_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def edit_file(path, search, replacement):
    content = read_file(path)
    if content.startswith("Error"):
        return content
    if search not in content:
        return f"Error: Could not find exact match in {path}"
    
    new_content = content.replace(search, replacement)
    return write_file(path, new_content)

def run_command(command):
    # Safety: Basic block list
    blocked = ["rm ", "del ", "format ", "> /dev/", "mkfs"]
    if any(b in command.lower() for b in blocked):
        return "Error: Command blocked for safety."
        
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=WORKSPACE_ROOT)
        output = result.stdout + result.stderr
        return output if output else "Command executed successfully (no output)."
    except Exception as e:
        return f"Error running command: {e}"

def grep_search(pattern):
    try:
        # Simple recursive search
        matches = []
        for root, dirs, files in os.walk(WORKSPACE_ROOT):
            if ".git" in root or "node_modules" in root: continue
            for file in files:
                if file.endswith((".py", ".cpp", ".h", ".txt", ".md")):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if pattern in line:
                                rel_path = os.path.relpath(path, WORKSPACE_ROOT)
                                matches.append(f"{rel_path}:{i+1}: {line.strip()}")
        return "\n".join(matches[:20]) if matches else "No matches found."
    except Exception as e:
        return f"Error searching: {e}"
