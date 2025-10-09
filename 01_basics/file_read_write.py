filename = "note.txt"

# 1. Create and write initial text to file
with open(filename, "w") as f:
    f.write("Hello! This is the first line.\n")

# 2. Append another line 
with open(filename, "a") as f:
    f.write("This is the second line.\n")

# 3. Read the entire content of the file
with open(filename, "r") as f:
    content = f.read()
print("Full content of the file:\n", content)

# 4. Read all lines into a list
with open(filename, "r") as f:
    lines = f.readlines()
print("All lines as a list:\n", lines)

# 5. Clean each line (remove newline characters)
cleaned = [line.strip() for line in lines]
print("Cleaned lines:\n", cleaned)