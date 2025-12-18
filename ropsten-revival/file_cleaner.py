
def remove_duplicate_lines(input_file, output_file):
    seen = set()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            stripped_line = line.strip()
            if stripped_line not in seen:
                seen.add(stripped_line)
                outfile.write(line)