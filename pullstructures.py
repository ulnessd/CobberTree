import os
import requests
import time

# Dictionary mapping our molecule names to their verified PubChem Compound ID (CID)
PUBCHEM_CIDS = {
    "Butane": 7843,
    "Pentane": 8003,
    "Hexane": 8058,
    "Heptane": 8900,
    "Isobutane": 6360,
    "Isopentane": 6556,
    "Neopentane": 10041,
    "Isohexane": 7892,
    "Octane": 356,
    "2-Methylheptane": 11594,
    "2,2-Dimethylhexane": 11551,
    "3-Methylpentane": 7282,
}

# The name of the subdirectory to save the files into
OUTPUT_FOLDER = "AlkaneStructures"


def download_sdf_files():
    """
    Downloads 3D SDF files for a predefined list of alkanes from PubChem.
    """
    print(f"Starting download of 3D structure files from PubChem...")

    # Create the output folder if it doesn't already exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: ./{OUTPUT_FOLDER}/")

    # Base URL for PubChem PUG REST API to get 3D structures
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/SDF?record_type=3d"

    for name, cid in PUBCHEM_CIDS.items():
        # Construct the full URL for the current molecule
        url = base_url.format(cid)

        # Define the output filename
        filename = f"{name.lower()}.sdf"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        # Check if the file already exists to avoid re-downloading
        if os.path.exists(output_path):
            print(f"- Skipping '{name}' (file already exists).")
            continue

        try:
            print(f"- Requesting data for '{name}' (CID: {cid})...", end="")
            response = requests.get(url, timeout=10)  # Added a timeout

            # Raise an error if the request was unsuccessful
            response.raise_for_status()

            # A simple check to ensure we're getting a plausible molecule file
            if "I" in response.text.splitlines()[3]:
                print(f" Error: Received unexpected data for {name}. Aborting this file.")
                continue

            # Write the downloaded content to the file
            with open(output_path, 'w') as f:
                f.write(response.text)

            print(f" Success! Saved to {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"\nError downloading data for {name}: {e}")

        # Be polite to the PubChem servers by waiting a moment between requests
        time.sleep(0.5)

    print("\nDownload process complete.")


if __name__ == "__main__":
    download_sdf_files()