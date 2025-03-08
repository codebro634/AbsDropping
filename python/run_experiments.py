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

def run_index(config: dict, index: int):

    cycle = index // number_of_commands(config)
    index = index % number_of_commands(config)

    # Check if the current exp config already exists
    pairing, agent_index, model_index = get_indices(config, index)
    decomposed_pairing_configs = decompose_config(config)
    reduced_config = deepcopy(decomposed_pairing_configs[pairing])
    reduced_config["agents"] = [reduced_config["agents"][agent_index]]
    reduced_config["models"] = [reduced_config["models"][model_index]]
    reduced_config["seed"] += cycle

    # Allow compatibility with modified config files
    output_folder = Path(config["dir"] + "/" + str(index) + "_" + str(cycle))
    num_exists = 1
    while output_folder.exists():
        output_folder = Path(config["dir"] + "/" + str(index) + "_" + str(cycle) + "_" + str(num_exists))
        num_exists += 1
    output_folder.mkdir(parents=True)

    start = time.strftime("%d_%m_%Y___%H_%M_%S")
    with open(output_folder / (start + ".start"), "w") as f:
        pass

    with open(output_folder / "experiment.yaml", "w") as f:
        yaml.safe_dump(reduced_config, f, sort_keys=False)

    command, dry_run_command = create_commands(reduced_config)

    # First conduct a dry run with checks on
    if dry_run_command is not None:
        try:
            with open(output_folder / "error.txt", "w") as err:
                exit_code = subprocess.run(dry_run_command, stdout=subprocess.DEVNULL, stderr=err, timeout=120).returncode
        except subprocess.TimeoutExpired:
            exit_code = 0

        error = False
        if exit_code != 0:
            error = True
            with open(output_folder / "error.txt", "a") as err:
                err.write("Dry run failed with exit code " + str(exit_code) + "\n")
        else:
            with open(output_folder / "error.txt", "r") as err:
                if err.read() != "": error = True
        if error:
            raise RuntimeError("Dry run failed. Check error.txt for more information.")


    # Actual run with optimized non-check, optimized version
    with open(output_folder / "output.txt", "w") as out, open(output_folder / "error.txt", "w") as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    if result.returncode != 0:
        with open(output_folder / "error.txt", "a") as err:
            err.write("Run failed with exit code " + str(result.returncode) + "\n")

    stop = time.strftime("%d_%m_%Y___%H_%M_%S")
    with open(output_folder / (stop + ".stop"), "w") as f:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple benchmarks on different agents and models.")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--index", type=int, help="Index of the benchmark to run")
    action.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--path",type=str, required=True, help="Path to the configuration file")

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

def create_commands(config: dict):
    agent = next(iter(config["agents"][0].values()))
    agent_type = agent["agent_type"]
    model = next(iter(config["models"][0].values()))
    model_type = model["model_type"]

    command  = [
        config["executable"],
        "--seed", str(config["seed"]),
        "--agent", agent_type,
        "--model", model_type,
        "--n_games", str(config["episodes"]),
        "--csv"
    ]

    if "planning_beyond_execution_horizon" in config and config["planning_beyond_execution_horizon"] == True:
        command .append("--planning_beyond_execution_horizon")
    if "deterministic_init" in config and config["deterministic_init"] == True:
        command .append("--deterministic_init")
    if "qtable" in config:
        command  += ["--qtable", config["qtable"]]

    if agent['agent_args']:
        for key,val in agent['agent_args'].items():
            command  += ["--aargs", f"{str(key)}={str(val)}"]
    if model['model_args']:
        for key,val in model['model_args'].items():
            command  += ["--margs", f"{str(key)}={str(val)}"]
    if 'horizons' in model:
        e_horizon,p_horizon = model['horizons']
        command  += ["--p_horizon", str(p_horizon)]
        command  += ["--e_horizon", str(e_horizon)]

    if "executable_dry_run" not in config:
        dry_run_command = None
    else:
        dry_run_command = command.copy()
        dry_run_command[0] = config["executable_dry_run"]

    return command, dry_run_command

def main():
    parsed_args = parse_arguments()

    # Create commands for all benchmarks
    with open(parsed_args.path, "r") as f:
        config = yaml.safe_load(f)

    config["dir"] = str(os.path.dirname(Path(parsed_args.path)))

    if parsed_args.all:
        for i in range(number_of_commands(config) * parsed_args.repeat):
            run_index(config, i)
    elif parsed_args.index is not None:
        run_index(config, parsed_args.index)

if __name__ == "__main__":
    main()
