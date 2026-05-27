import subprocess

STEPS = [
    "scripts/00_validate_dataset.py",
    "scripts/01_preprocess_dataset.py",
    "scripts/02_split_dataset.py",
    "scripts/03_extract_embeddings.py",
    "scripts/04_run_clustering.py",
    "scripts/05_run_probing.py",
    "scripts/06_generate_analysis.py",
]

if __name__ == "__main__":

    for step in STEPS:

        print(f"Running: {step}")

        subprocess.run(
            ["python", step],
            check=True
        )

    print("Full pipeline completed.")