import multiprocessing


# Function to process a line
def process_line(line, processed_ids):
    # Extract the ID from the line (assuming it's in the first column)
    parts = line.strip().split('\t')
    if len(parts) >= 1:
        id = parts[0]

        # Check if the ID has been processed before
        with processed_ids.Lock():
            if id not in processed_ids:
                # Process the line (replace with your processing logic)
                print(f"Processing ID: {id}, Line: {line}")
                processed_ids.add(id)
            else:
                print(f"Skipping duplicate ID: {id}")


# Function to read and process a file
def process_file(filename, processed_ids):
    try:
        with open(filename, 'r') as file:
            for line in file:
                process_line(line, processed_ids)
    except FileNotFoundError:
        print(f"File not found: {filename}")


if __name__ == '__main__':
    # Create a manager for the set of processed IDs
    with multiprocessing.Manager() as manager:
        processed_ids = manager.list()

        # List of input files to be processed in parallel
        input_files = ['file1.txt', 'file2.txt', 'file3.txt']

        # Create a multiprocessing pool
        pool = multiprocessing.Pool()

        # Use the pool to process files in parallel
        pool.starmap(process_file,
                     [(filename, processed_ids) for filename in input_files])

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        print("All processing completed.")
        print(processed_ids)