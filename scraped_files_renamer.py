import os

def prepend_original_paths(folder_path, mapping_file_path):
    """
    Reads a mapping file and prepends the original path to the corresponding .txt files.
    """
    
    # 1. Parse the mapping file into a dictionary
    # Structure: {'48623': '/users/dev/script.py', ...}
    file_mapping = {}
    
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split only on the first comma to avoid issues if the path itself contains commas
                parts = line.split(',', 1)
                
                if len(parts) == 2:
                    file_id = parts[0].strip()
                    original_path = parts[1].strip()
                    
                    # Store in dictionary
                    file_mapping[file_id] = original_path
                else:
                    print(f"Skipping malformed line in mapping: {line}")
    except FileNotFoundError:
        print(f"Error: Could not find mapping file at {mapping_file_path}")
        return

    # 2. Iterate through the files and modify them
    count_success = 0
    count_missing = 0

    print(f"Found {len(file_mapping)} entries in mapping. Starting processing...")

    for file_id, original_path in file_mapping.items():
        # Construct the current filename (assuming extensions are .nvn)
        current_filename = f"{file_id}.nvn"
        file_path = os.path.join(folder_path, current_filename)

        if os.path.exists(file_path):
            try:
                # Read the existing content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Create the new header line
                # We add a newline characters to separate it from the original code
                header = f"///ORIGINAL_PATH: {original_path}\n\n"
                
                # Combine header and content
                new_content = header + content

                # Write the new content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"[OK] Updated {current_filename}")
                count_success += 1

            except Exception as e:
                print(f"[Error] Failed to process {current_filename}: {e}")
        else:
            print(f"[Missing] Could not find file: {current_filename}")
            count_missing += 1

    # 3. Summary
    print("-" * 30)
    print(f"Processing complete.")
    print(f"Files updated: {count_success}")
    print(f"Files missing: {count_missing}")

# --- CONFIGURATION ---
# Update these paths before running
if __name__ == "__main__":
    
    # The folder containing your scraped .txt files (e.g., "scraped_files")
    TARGET_FOLDER = "./env_scripts" 
    
    # The file containing the list of "ID, Original Path" (e.g., "mapping.txt")
    MAPPING_FILE = "./mapping.txt"

    # Check if paths are just placeholders and warn user
    if not os.path.exists(TARGET_FOLDER):
        print(f"Please create the folder '{TARGET_FOLDER}' or update the TARGET_FOLDER variable.")
        # For testing purposes, uncomment the lines below to generate dummy data:
        # os.makedirs(TARGET_FOLDER, exist_ok=True)
        # with open(MAPPING_FILE, 'w') as f: f.write("48623, /src/app/main.py\n48624, /src/utils/helper.js")
        # with open(f"{TARGET_FOLDER}/48623.txt", 'w') as f: f.write("print('hello world')")
    else:
        prepend_original_paths(TARGET_FOLDER, MAPPING_FILE)