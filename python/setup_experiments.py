import argparse
import os
import subprocess
import yaml
from pathlib import Path
from copy import deepcopy
import time
import shutil
import json
from itertools import product

TNT_STUD_PARTITION = "cpu_short_stud"
TNT_WIMI_PARTITION = "cpu_short"
LUIS_PARTITION = "amo,gih,isd"
TNT_EXCLUDES = "epyc1,epyc2,epyc3,epyc4,cc1l01"

SLURM_RUN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output={cwd}/slurm_outputs/slurm-%a.out
#SBATCH --error={cwd}/slurm_outputs/slurm-%a.err
#SBATCH --time={runtime}
#SBATCH --partition={partition}
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{number_of_commands}
#SBATCH --verbose
#SBATCH --ntasks=1
{excludes}

srun -c 1 -v {python} {runpath}/run_experiments.py --index $SLURM_ARRAY_TASK_ID --path {cwd}/experiment.yaml
"""

CMAKE_DIR = "/home/schmoeck/nobackup/IDEs/clion-2024.1.4/bin/cmake/linux/x64/bin/cmake"
BUILD_DIR = "/home/schmoeck/Research/AbstractionsInSearch/BenchmarkGames/cmake-build-debug"

SLURM_EPYC_COMPILE = f"""#!/bin/bash
#SBATCH --job-name=compile
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --partition=epyc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

{CMAKE_DIR} --build {BUILD_DIR} --target BenchmarkGamesDebug -j 8
{CMAKE_DIR} --build {BUILD_DIR} --target BenchmarkGamesRelease -j 8
"""

def create_slurm_enviroment(config: dict, repeats: int):

    ### First Check if the environment can be created or if the config has inconsistencies ###

    if not Path(".git").exists():
        raise RuntimeError("You can only create a slurm environment in the root folder of the git repository.")

    if config["folder_name"] is not None and (Path("nobackup") / config["folder_name"]).exists():
        raise RuntimeError(f"Folder {config['folder_name']} already exists in nobackup. Please choose a different name.")

    if config["folder_name"] is not None:
        output_folder = Path("nobackup") / config["folder_name"]
    else:
        date = time.strftime("%d-%m-%Y_%H-%M-%S")
        output_folder = Path("nobackup") / (date + "_" + Path(config["path"]).stem)
    if not (Path(config["executable"])).exists():
        raise RuntimeError(f"Executable {config['executable']} not found in the root folder of the git repository.")

    if config["slurm_config"] == "stud":
        partition = TNT_STUD_PARTITION + (",epyc" if config["epyc"] else "")
        python = "python3"
        excludes = "" if config["epyc"] else f"#SBATCH --exclude={TNT_EXCLUDES}"
    elif config["slurm_config"] == "wimi":
        partition = TNT_WIMI_PARTITION + (",epyc" if config["epyc"] else "")
        excludes = "" if config["epyc"] else f"#SBATCH --exclude={TNT_EXCLUDES}"
        python = "python"
    elif config["slurm_config"] == "luis":
        partition = LUIS_PARTITION
        excludes = ""
        python = "python"
    else:
        raise RuntimeError("Invalid slurm configuration. Use --cfg [stud,wimi,luis]")

    ### Create the environment folder and setup executables ###

    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(config["executable"], output_folder / Path(config["executable"]).name)
    if "executable_dry_run" in config:
        shutil.copy(config["executable_dry_run"], output_folder / Path(config["executable_dry_run"]).name)

    config = deepcopy(config)
    config["executable"] = str(output_folder / Path(config["executable"]).name)
    if "executable_dry_run" in config:
        config["executable_dry_run"] = str(output_folder / Path(config["executable_dry_run"]).name)
    with open(output_folder / "experiment.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(output_folder / "slurm.sh", "w") as f:
        runtime =  f"{(config.get('runtime') // 60):02}:{config.get('runtime')%60:02}:00"
        f.write(SLURM_RUN_TEMPLATE.format(number_of_commands=number_of_commands(config) * repeats - 1, partition=partition, excludes=excludes, python=python, runtime = runtime,cwd=output_folder.resolve(), runpath= os.path.dirname(os.path.abspath(__file__))))

    (output_folder / "slurm_outputs").mkdir(parents=True, exist_ok=True)

    os.sync()

    return output_folder

def get_configs_hash_set(folder: Path):
    configs_set = set()
    exp_paths = [(f / "experiment.yaml") for f in folder.iterdir() if (f / "experiment.yaml").exists()]
    for exp in exp_paths:
        with open(exp, "r") as f:
            config = yaml.safe_load(f)
            configs_set.add(hash(json.dumps(config, sort_keys=True)))
    return configs_set

def decompose_pairing(original_config: dict):
    config = deepcopy(original_config)
    #For each arg comb in each agent, make this a new agent with only that arg comb
    new_agents = []
    for agent in config["agents"]:
        agent_name = next(iter(agent.keys()))
        agent_type = next(iter(agent.values()))['agent_type']
        args_dict = next(iter(agent.values()))['agent_args']
        if args_dict is not None:
            keys, values = [], []
            for key,value in args_dict.items():
                keys.append(key)
                values.append(value if isinstance(value, list) else [value])
            cross_product = product(*values)
            result = [dict(zip(keys, combination)) for combination in cross_product]
            new_agents += [ {agent_name : {'agent_type': agent_type, 'agent_args' : arg_comb}} for arg_comb in result ]
        else:
            new_agents.append(agent)
    config["agents"] = new_agents

    #Perhaps the same for models...?

    return config

def decompose_config(original_config: dict):
    pairings = []
    def get_config_index(key, name):
        for i, entry in enumerate(config[key]):
            if next(iter(entry.keys())) == name:
                return i
        raise ValueError(f"Could not find {name}.")

    for pair in original_config["pairings"]:
        config = deepcopy(original_config)

        #retrieve agent configs
        agents = []
        for agent_name in list(pair.values())[0][0]['agents']:
            agents.append(config["agents"][get_config_index("agents", agent_name)])
        config["agents"] = agents

        #retrieve models
        model_names = list(pair.values())[0][1]['models']
        config["models"] = [ config["models"][get_config_index("models", model)] for model in model_names]

        del config["pairings"]
        pairings.append(decompose_pairing(config))

    return pairings



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple benchmarks on different agents and models.")

    parser.add_argument("--cfg", type=str, default = "wimi", help="Which slurm template config to choose. Options: wimi, stud, luis")
    parser.add_argument("--time", type=int, default = 719, help="Job runtime in minutes.")
    parser.add_argument("--epyc", action="store_true", help="If set do compilation on an EPYC machine.")
    parser.add_argument("--dc", action="store_true", help="If set no compilation.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the experiment.")
    parser.add_argument(type=str, dest="path", help="Path to the configuration file")
    parser.add_argument("--run", action="store_true", help="If set, directly run the slurm job after environment creation.")
    parser.add_argument("--name", type=str, default=None, help="If specified, the experiment folder will carry this name. Otherwise, the name of the config file with the date of creation is used.")

    return parser.parse_args()


def number_of_commands(config: dict):
    decomposed_pairing_configs = decompose_config(config)
    return sum([len(pair["agents"]) * len(pair["models"]) for pair in decomposed_pairing_configs])

def get_indices(config: dict, index: int):
    decomposed_pairing_configs = decompose_config(config)

    cumul_sums = []
    for i,pair in enumerate(decomposed_pairing_configs):
        cumul_sums.append( (cumul_sums[-1] if i > 0 else 0) + len(pair["agents"]) * len(pair["models"]))

    pair_number = 0
    for i, cumul_sum in enumerate(cumul_sums):
        if index < cumul_sum:
            pair_number = i
            index -= cumul_sums[i-1] if i > 0 else 0
            break

    return pair_number, index % len(decomposed_pairing_configs[pair_number]["agents"]), index // len(decomposed_pairing_configs[pair_number]["agents"])



def compile_project(epyc_compile = False):

    if epyc_compile:

        #Sanity checks
        if Path("./tmp_slurm_compile.sh").exists():
            raise RuntimeError("tmp_slurm_compile.sh already exists. Please remove or rename it before running the script.")

        if not Path(BUILD_DIR).exists():
            raise RuntimeError("Build directory does not exist. Please create it before running the script.")

        if not Path(CMAKE_DIR).exists():
            raise RuntimeError("CMake directory does not exist. Please create it before running the script.")

        #Delete old compilates
        Path(BUILD_DIR).joinpath("BenchmarkGamesDebug").unlink(missing_ok=True)
        Path(BUILD_DIR).joinpath("BenchmarkGamesRelease").unlink(missing_ok=True)

        #Do compilation on EPYC partition
        tmp_name = "tmp_slurm_compile.sh"
        with open(Path(tmp_name), "w") as f:
            f.write(SLURM_EPYC_COMPILE)
        submit_result = subprocess.run(["sbatch", tmp_name], capture_output=True, text=True, check=True)
        Path(tmp_name).unlink()

        output = submit_result.stdout.strip()
        job_id = None
        for word in output.split():
            if word.isdigit():
                job_id = word
                break
        if job_id is None:
            raise RuntimeError("Could not find job id in slurm output.")

        while True:
            check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            if job_id not in check_result.stdout:
                break  # Job is no longer in the queue
            print("Waiting for compile job completion...")
            time.sleep(15)  # Wait and check again

        # Check SLURM job exit status
        sacct_result = subprocess.run(["sacct", "-j", job_id, "--format=State"], capture_output=True, text=True)

        os.sync()
        for f in Path(BUILD_DIR).iterdir():
            pass

        if (not "COMPLETED" in sacct_result.stdout) or (not Path(BUILD_DIR).joinpath("BenchmarkGamesDebug").exists()) or (not Path(BUILD_DIR).joinpath("BenchmarkGamesRelease").exists()):
            raise RuntimeError(f"Compilation job failed {sacct_result.stdout} {sacct_result.stderr}")
        else:
            print("Compilation job completed successfully.")

    else:
        result = subprocess.run([
            "bash", "-c",
            f"{CMAKE_DIR} --build {BUILD_DIR} --target BenchmarkGamesDebug -j 8 && "
            f"{CMAKE_DIR} --build {BUILD_DIR} --target BenchmarkGamesRelease -j 8"
        ], check=True, capture_output=True, text=True)
        print(result.stdout)

def main():
    parsed_args = parse_arguments()

    # Create commands for all benchmarks
    with open(parsed_args.path, "r") as f:
        config = yaml.safe_load(f)

    if not parsed_args.dc:
        compile_project(parsed_args.epyc)

    config["epyc"] = parsed_args.epyc
    config["slurm_config"] = parsed_args.cfg
    config["path"] = parsed_args.path
    config["runtime"] = parsed_args.time
    config["folder_name"] = parsed_args.name
    output_folder = create_slurm_enviroment(config, parsed_args.repeat)

    if parsed_args.run:
        subprocess.run(["sbatch", output_folder / "slurm.sh"])

if __name__ == "__main__":
    main()
