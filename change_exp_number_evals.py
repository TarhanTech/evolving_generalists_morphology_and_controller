import argparse
from pathlib import Path
import re
import pickle
import sys
import torch
from source.training_env import TrainingSchedule, TerrainType

MAX_EVALS: int = 303600

def vscode_sorting_key(path):
    # Natural sorting: split into parts (digits and text)
    import re
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', path.name)
    ]

def full_gen(exp_path: Path):
    max_generations: int = MAX_EVALS // (23 * 33)
    print(f"Maximum allowed generations: {max_generations}")

    for run_path in exp_path.iterdir():
        if not run_path.is_dir():
            print(f"Skipping {run_path}, not a directory.")
            continue

        tensors_path: Path = run_path / "partition_1" / "gen_tensors"

        if not tensors_path.exists() or not tensors_path.is_dir():
            print(f"Tensor directory not found: {tensors_path}. Skipping.")
            continue

        best_tensor_files = [
            f for f in tensors_path.iterdir()
            if f.is_file() and re.match(r"tensor_\d+_best\.pt", f.name)
        ]

        if not best_tensor_files:
            print(f"No best tensor files found in {tensors_path}. Skipping.")
            continue
        
        print(f"best tensors: {best_tensor_files}")

        eligible_tensors = []
        for f in best_tensor_files:
            match = re.match(r"tensor_(\d+)_best\.pt", f.name)
            if match:
                gen_num = int(match.group(1))
                if gen_num <= max_generations:
                    eligible_tensors.append((gen_num, f))

        if not eligible_tensors:
            print(f"No eligible tensors (<= {max_generations}) found in {tensors_path}. Skipping.")
            continue

        best_num, best_file = max(eligible_tensors, key=lambda x: x[0])
        print(f"Selected best tensor: {best_file.name} (Generation {best_num})")

        try:
            tensor = torch.load(best_file)
        except Exception as e:
            print(f"Error loading tensor from {best_file.name}: {e}")
            continue

        g_var_path = run_path / "G_var.pkl"
        try:
            with g_var_path.open('wb') as f:
                pickle.dump([tensor], f)
            print(f"Saved G_var.pkl in {run_path}")
        except Exception as e:
            print(f"Error saving G_var.pkl in {run_path}: {e}")
            continue

        for fname in ["env_fitnesses_test.pkl", "env_fitnesses_training.pkl"]:
            fpath = run_path / fname
            if fpath.exists() and fpath.is_file():
                try:
                    fpath.unlink()
                    print(f"Deleted {fname} from {run_path}")
                except Exception as e:
                    print(f"Error deleting {fname} from {run_path}: {e}")
            else:
                print(f"File {fname} does not exist in {run_path}, skipping deletion.")

def spec(exp_path: Path):
    max_generations: int = (MAX_EVALS // 81) // 23
    print(f"Maximum allowed generations: {max_generations}")

    for run_path in exp_path.iterdir():
        if not run_path.is_dir():
            print(f"Skipping {run_path}, not a directory.")
            continue

        spec_path: Path = run_path / "specialist"

        if not spec_path.exists() or not spec_path.is_dir():
            print(f"Specialist directory not found: {spec_path}. Skipping.")
            continue
        
        g: list[Tensor] = []
        e: list[list[TerrainType]] = []
        t: TrainingSchedule = TrainingSchedule()
        for env_path in sorted(spec_path.iterdir(), key=lambda p: (not p.is_dir(), vscode_sorting_key(p))):
            if not env_path.is_dir():
                print(f"Skipping {env_path}, not a directory.")
                continue
            
            tensor_path = env_path / "gen_tensors"

            if not tensor_path.exists() or not tensor_path.is_dir():
                print(f"Tensor directory not found: {tensor_path}. Skipping.")
                continue

            best_tensor_files = [
                f for f in tensor_path.iterdir()
                if f.is_file() and re.match(r"tensor_\d+_best\.pt", f.name)
            ]

            eligible_tensors = []
            for f in best_tensor_files:
                match = re.match(r"tensor_(\d+)_best\.pt", f.name)
                if match:
                    gen_num = int(match.group(1))
                    if gen_num <= max_generations:
                        eligible_tensors.append((gen_num, f))

            if not eligible_tensors:
                print(f"No eligible tensors (<= {max_generations}) found in {tensor_path}. Skipping.")
                continue

            best_num, best_file = max(eligible_tensors, key=lambda x: x[0])
            print(f"{env_path.name}: {best_num}")
            print(best_num)
            try:
                tensor: Tensor = torch.load(best_file)
            except Exception as e:
                print(f"Error loading tensor from {best_file.name}: {e}")
                continue
            g.append(tensor)

            env_folder_name = env_path.name
            matched_terrains = [terrain for terrain in t.all_terrains if str(terrain) == env_folder_name]
            if not matched_terrains:
                raise ValueError(f"No terrain matched for folder: {env_folder_name}")
            if len(matched_terrains) > 1:
                raise ValueError(f"Multiple terrains matched for folder: {env_folder_name}. Matches: {matched_terrains}")
            
            e.append(matched_terrains)

        g_var_path = run_path / "G_var.pkl"
        try:
            with g_var_path.open('wb') as f:
                pickle.dump(g, f)
            print(f"Saved G_var.pkl in {run_path}")
        except Exception as e:
            print(f"Error saving G_var.pkl in {run_path}: {e}")
            continue

        e_var_path = run_path / "E_var.pkl"
        try:
            with e_var_path.open('wb') as f:
                pickle.dump(e, f)
            print(f"Saved E_var.pkl in {run_path}")
        except Exception as e:
            print(f"Error saving E_var.pkl in {run_path}: {e}")
            continue

        for fname in ["env_fitnesses_test.pkl", "env_fitnesses_training.pkl"]:
            fpath = run_path / fname
            if fpath.exists() and fpath.is_file():
                try:
                    fpath.unlink()
                    print(f"Deleted {fname} from {run_path}")
                except Exception as e:
                    print(f"Error deleting {fname} from {run_path}: {e}")
            else:
                print(f"File {fname} does not exist in {run_path}, skipping deletion.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_path',
        type=Path,
        required=True,
        help='Path to the experiments directory.'
    )
    args = parser.parse_args()

    exp_path_str = str(args.exp_path)

    if exp_path_str.startswith("runs/FullGen"):
        print("Experiment type detected: FullGen. Processing with full_gen.")
        full_gen(args.exp_path)
    elif exp_path_str.startswith("runs/Spec"):
        print("Experiment type detected: Spec. Processing with spec.")
        spec(args.exp_path)
    else:
        print(f"Unknown experiment type for path: {exp_path_str}")
        sys.exit(1)

if __name__ == "__main__":
    main()
