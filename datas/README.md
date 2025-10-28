# Data Processing Script

This folder contains the `data.py` script, which is used to download and process datasets from the Hugging Face Hub.

## Features

- Downloads the `benchmark.jsonl` file for a specified split (e.g., `test`) from the Hugging Face Hub.
- Saves the processed data as a JSON file.
- Downloads and extracts the `csv_files.zip` file into the corresponding split directory.

## Usage

1. Ensure the required dependencies are installed:
   ```bash
   pip install huggingface_hub
   ```

2. Run the script:
   ```bash
   python data.py
   ```

3. After processing, the output file structure will look like this:
   ```
   ./<split>/benchmark.json
   ./<split>/csv_files
   ```

## Notes

- The default Hugging Face dataset repository used is `momo006/PARROT`.
- The output directory is located under the current folder at `./<split>/`.
