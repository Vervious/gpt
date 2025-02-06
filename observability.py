import inspect
import sys

def extract_flagged_code():
    # Get the current script's source code
    # current_frame = inspect.currentframe()
    # source_lines = inspect.getsourcelines(current_frame)[0]
    # print("HELLO")
    # print(source_lines)
    # print("GOODBU")
    main_script_path = sys.argv[0]
    
    # Read the entire source code of the main script
    with open(main_script_path, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()

    # Find start and end indices of the flagged section
    start_marker = f"# @flag"
    end_marker = f"# @endflag"
    in_section = False
    flagged_code = []
    current_name = ""
    
    for line in source_lines:
        line_clean = line.strip()
        if line_clean.startswith(start_marker):
            current_name = line_clean[len(start_marker):]
            flagged_code.append(current_name.rstrip('\n'))
            in_section = True
        elif line_clean.startswith(end_marker):
            flagged_code.append("----------------")
            in_section = False
        elif in_section:
            flagged_code.append(line.rstrip('\n'))  # Remove newline for clean output
    
    return '\n'.join(flagged_code)
