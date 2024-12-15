import os
import shutil

def delete_items_in_folders(base_folder, items_to_delete):
    """
    Iterates over all subfolders in a base folder and deletes specified files and folders.

    :param base_folder: The base folder to iterate through.
    :param items_to_delete: List of file and folder names to delete within each subfolder.
    """
    for root, dirs, files in os.walk(base_folder):
        for item in items_to_delete:
            item_path = os.path.join(root, item)
            try:
                if os.path.isdir(item_path):
                    # Check if the item is a directory
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                elif os.path.isfile(item_path):
                    # Check if the item is a file
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
                else:
                    print(f"Path does not exist: {item_path}")
            except Exception as e:
                print(f"Failed to delete {item_path}. Error: {e}")

if __name__ == "__main__":
    # Base folder to iterate through
    base_folder = "./_graphs_for_paper/Spec-MorphEvo_1115_08-12_08-19-00-263340/specialist"

    # List of folders and files to delete
    items_to_delete = [
        "gen_tensors",
        "morph_params_evolution_plots",
        "pca_plots",
        "screenshots",
        "evolution_video.mp4",
        "pandas_logger_df.csv"

    ]

    # Call the delete function
    delete_items_in_folders(base_folder, items_to_delete)
