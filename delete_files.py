import os
import shutil

def delete_items_in_folders(base_folder, items_to_delete):
    """
    Iterates over all subfolders in a base folder and deletes specified files and folders 
    by looking for items whose names match any in the items_to_delete list.

    :param base_folder: The base folder to iterate through.
    :param items_to_delete: List of file and folder names to delete.
    """
    # Use topdown=False so that directories are processed after the files inside them.
    for root, dirs, files in os.walk(base_folder, topdown=False):
        # Delete matching files
        for filename in files:
            if filename in items_to_delete:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete file {file_path}. Error: {e}")

        # Delete matching directories
        for dirname in dirs:
            if dirname in items_to_delete:
                dir_path = os.path.join(root, dirname)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted folder: {dir_path}")
                except Exception as e:
                    print(f"Failed to delete folder {dir_path}. Error: {e}")

if __name__ == "__main__":
    # Base folder to iterate through
    base_folder = "_graphs_for_paper/Spec-MorphEvo_1073_01-12_23-36-12-412793"

    # List of folder and file names to delete (exact names)
    items_to_delete = [
        "gen_tensors",
        "screenshots",
        "gen_score_pandas_df.csv",
        "fitness_scores.json",
        "pandas_logger_df.csv",
        "E_established.pkl",
        "E_var.pkl",
        "env_fitnesses_test.pkl",
        "env_fitnesses_training.pkl",
        "G_var.pkl",
    ]

    # Execute the deletion
    delete_items_in_folders(base_folder, items_to_delete)
