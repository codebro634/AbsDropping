import math
import os
from copy import deepcopy

import pandas as pd
import yaml
import numpy as np
from datetime import datetime
import shutil
from scipy.stats import bootstrap
from typing import List, Callable, Union
from functools import cmp_to_key
import importlib.util
from pathlib import Path


def delete_experiment_files(directory, args_to_remove):

    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        if 'experiment.yaml' in filenames:
            with open(os.path.join(dirpath, 'experiment.yaml')) as f:
                args = yaml.load(f, Loader=yaml.FullLoader)

            try:
                model_name = list(args["models"][0].keys())[0].split("/")[-1]
            except:
                print(f"Failed to extract model name from {args['models']}")
                continue
            try:
                agent_type =list(args['agents'][0].values())[0]['agent_type']
            except:
                print(f"Failed to extract agent type from {args['agents']}")
            try:
                agent_args = list(args['agents'][0].values())[0]['agent_args']
            except:
                print(f"Failed to extract agent args from {args['agents']}")
            def condition(x):
                if x[0] == "agent": return agent_type == x[1]
                if x[0] == "model": return model_name == x[1]
                return x[0] in agent_args and agent_args[x[0]] == x[1]

            if all([condition(x) for x in args_to_remove]):
                print(f"Deleting {dirpath}")
                shutil.rmtree(dirpath)

#delete_experiment_files("./nobackup/",  [("model","tr")])
#exit(1)

def dir_iterator(start_directory, substrings):
    for root, dirs, files in os.walk(start_directory):
        # Filter dirs in place to skip non-matching subdirectories
        dirs[:] = [d for d in dirs if any(s in os.path.join(root, d) for s in substrings)]

        # If the current root matches any substring, yield it
        if any(s in root for s in substrings):
            yield root, dirs, files

def get_progress_in_experiment(root, files):

    start_filenames = [f for f in files if f.endswith('.start')]
    end_filenames = [f for f in files if f.endswith('.stop')]
    assert len(start_filenames) == 1
    start_filename = start_filenames[0]
    end_filename = end_filenames[0] if len(end_filenames) == 1 else None
    did_finish = len(end_filenames) == 1

    assert  "output.txt" in files
    if did_finish:
        progress = math.inf
    else:
        with open(os.path.join(root, 'output.txt')) as f:
            data = pd.read_csv(f, sep=';')
            progress = len(data.index)

    if did_finish:
        time_format = "%d_%m_%Y___%H_%M_%S"
        time1 = datetime.strptime(start_filename.split(".")[0], time_format)
        time2 = datetime.strptime(end_filename.split(".")[0], time_format)
        runtime = round(abs((time2 - time1).total_seconds()) / 60,2)
    else:
        runtime = 0

    return progress, runtime

def get_exp_mean_and_conf(metric, start_dir, dir_substrings, show_progress = True, bootstrap_samples = None, max_over_entries = False, skip=1):
    csv_results = {} #Key is experiment identifier, value is tuple of mean and confint
    longest_runtime, least_progress, least_progress_file, exp_count = 0, math.inf, None, 0

    ctr = 0
    for root, dirs, files in dir_iterator(start_dir, dir_substrings):
        ctr += 1
        if ctr % skip != 0:
            continue

        if "experiment.yaml" not in files and 'slurm.sh' not in files:
            continue

        if "error.txt" not in files:
            print(f"WARNING: Experiment {root} has no error.txt file., Skipping")
            continue

        with open(os.path.join(root, 'error.txt')) as f:
            error = f.read()
            if len(error) > 0:
                print(f"WARNING: Error in {root}: {error}, Skipping")
                continue

        if "output.txt" not in files:
            print(f"WARNING: Experiment {root} has no output.txt file., Skipping")
            continue

        progress, runtime = get_progress_in_experiment(root, files)
        if runtime > longest_runtime: longest_runtime = runtime
        if progress < least_progress:
            least_progress = progress
            least_progress_file = root

        try:
            data = pd.read_csv(os.path.join(root, 'output.txt'), sep=';')
        except:
            print(f"Failed to read {os.path.join(root, 'output.txt')}")
            continue

        if 'Rewards Player 0' not in data:
            continue
        exp_count += 1
        with open(os.path.join(root, 'experiment.yaml')) as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        agent_name = list(args["agents"][0].keys())[0]
        agent_type = args["agents"][0][agent_name]['agent_type']
        agent_args = tuple(args["agents"][0][agent_name]['agent_args'].items()) if args["agents"][0][agent_name]['agent_args'] is not None else tuple([])

        model_name = list(args["models"][0].keys())[0]
        model_type = args["models"][0][model_name]['model_type']
        model_args = tuple(args["models"][0][model_name]['model_args'].items()) if args["models"][0][model_name]['model_args'] is not None else tuple([])
        horizons = tuple([])
        if 'horizons' in args["models"][0][model_name]:
            e_horizon, p_horizon = args["models"][0][model_name]['horizons']
            horizons = (('e_horizon',e_horizon), ('p_horizon',p_horizon))

        key = ((("model",model_type),("agent",agent_type))+horizons+agent_args+model_args)

        # if model_type == "gol" and dict(key).get("confidence",0.0) == 0.8 and dict(key).get("expfacs",0.0)  == "-1;2" and dict(key).get("distribution_layers",0.0) == 4 and dict(key)["iterations"] == 2000:
        #     print(f"{root}")
        #     exit(1)

        if len(key) != len(set([x[0] for x in key])):
            raise ValueError(f"Key {key} contains duplicates.")

        if key not in csv_results:
            csv_results[key] = []

        if metric == "Rewards Player 0":
            results = pd.Series(data[metric].tolist())
        elif metric == "Optimal Actions Chosen 0":
            total = sum([ len(x.strip().split(" ")) for x in data['Regrets Player 0'].tolist() if str(x) != 'nan'])
            actual =  sum([ sum([1 if abs(float(y)) < 0.0001 else 0 for y in x.strip().split(" ")]) for x in data['Regrets Player 0'].tolist() if str(x) != 'nan'])
            results = pd.Series([1]*actual + [0]*(total-actual))
        elif metric in ["Regrets Player 0","Times Player 0"]:
            concat_vals = []
            for vals in data[metric].tolist():
                if str(vals) == 'nan':
                    continue
                concat_vals += [float(x) for x in vals.strip().split(" ")]
            results = pd.Series(concat_vals)
        else:
            raise ValueError(f"Metric {metric} not recognized")

        csv_results[key].append(results)
        if show_progress:
            print(f"Processed {exp_count} experiments")

    mean_and_conf = {}
    bootstrapped = 0
    for key in csv_results.keys():
        concat = pd.concat(csv_results[key], axis=0).to_numpy()

        if len(concat) == 0:
            continue

        if max_over_entries:
            mean_and_conf[key] = (np.max(concat), 0)
        else:
            mean = np.mean(concat)
            if len(concat)> 1:
                if bootstrap_samples is None:
                    upper_conf_bound = np.std(concat) * 2.33 / math.sqrt(len(concat)) + mean
                else:
                    upper_conf_bound = bootstrap((concat,), np.mean, confidence_level=0.99, n_resamples=bootstrap_samples, method='percentile').confidence_interval[1]
            else:
                upper_conf_bound = mean
            mean_and_conf[key] = (mean, upper_conf_bound-mean)

        bootstrapped += 1
        if show_progress:
            print(f"Conf interval for {bootstrapped}/{len(csv_results.keys())} experiments")

    print(f"Longest experiment duration: {longest_runtime} minutes")
    print(f"Least progress in an experiment that did not stop: {least_progress} at {least_progress_file}")

    return mean_and_conf


def reduce_data(mean_and_confint,
                reduction_func: Callable,
                filter_by: list,
                dependent_var: str,
                assert_unique = False):

    var_to_values = {}
    for key in mean_and_confint.keys():
        key_dict = dict(key)
        if dependent_var not in key_dict:
            continue
        if all([filter[0] in key_dict and key_dict[filter[0]] == filter[1] for filter in filter_by]):
            if key_dict[dependent_var] in var_to_values:
                assert not assert_unique
                var_to_values[key_dict[dependent_var]].append(mean_and_confint[key])
            else:
                var_to_values[key_dict[dependent_var]] = [mean_and_confint[key]]

    key_to_val = {}
    for key in var_to_values.keys():
        key_to_val[key] = reduction_func(var_to_values[key], key = lambda x: x[0])

    x_list = list(key_to_val.keys())
    try:
        x_list = sorted(x_list)
    except:
        x_list = sorted(x_list, key = lambda x: str(x))

    return x_list, [key_to_val[x][0] for x in x_list], [key_to_val[x][1] for x in x_list] #x_values, y_values, error_values


def merge_cols(table1, table2, row_names1, row_names2, col_names1, col_names2):
    if len(table2) < len(table1):
        old_table = deepcopy(table1)
        table1 = []
        for i in range(len(old_table)):
            if row_names1[i] in row_names2:
                table1.append(old_table[i])
        row_names1 = deepcopy(row_names2)
    elif len(table2) > len(table1):
        raise ValueError("Table 2 has more rows than table 1. Table 1 must have at least as many rows as table 2.")


    table = deepcopy(table1)
    col_names = deepcopy(col_names1)
    for i in range(len(table)):
        table[i] += table2[i]
    col_names += col_names2
    return table, deepcopy(row_names1), deepcopy(col_names)

def table_to_string(table, row_names,col_names,

                #For latex output
                latex_output = False,
                env = "table",
                env_options = "",
                mark_best_per_column=False,
                mark_best_per_row = False,
                label = None,
                scalebox = 1.0,
                caption="Caption"):

    if len(table) == 0:
        print ("WARNING: You are trying to print an empty table")
        return ""

    #Determine highest value per column
    col_maxima = [-math.inf for _ in range(len(table[0]))]
    for i in range(len(table)):
        for j in range(len(table[i])):
            mean_and_confint = table[i][j]
            if mean_and_confint[0] > col_maxima[j]:
                col_maxima[j] = mean_and_confint[0]

    row_maxima = [-math.inf for _ in range(len(table))]
    for i in range(len(table)):
        for j in range(len(table[i])):
            mean_and_confint = table[i][j]
            if mean_and_confint[0] > row_maxima[i]:
                row_maxima[i] = mean_and_confint[0]

    #Prepate the strings to be output for each table entry
    table_entries = deepcopy(table)
    for i in range(len(table_entries)):
        for j in range(len(table_entries[i])):
            mean_and_confint = table[i][j]

            digits = 2 if abs(mean_and_confint[0]) < 10 else 1
            if latex_output:
                if (mark_best_per_column and mean_and_confint[0] == col_maxima[j]) or (mark_best_per_row and mean_and_confint[0] == row_maxima[i]):
                    table_entries[i][j] = f"$\\boldsymbol {{ {mean_and_confint[0]:.{digits}f} \pm {mean_and_confint[1]:.{digits}f} }}$"
                else:
                    table_entries[i][j] = f"${mean_and_confint[0]:.{digits}f} \pm {mean_and_confint[1]:.{digits}f}$"
            else:
                table_entries[i][j] = f"{mean_and_confint[0]:.{digits}f} ± {mean_and_confint[1]:.{digits}f}"

    if latex_output:
        #print table to latex format
        latex = "\\begin{" + env + "}[" + env_options +"]"
        latex += "\\centering\n"
        latex += "\\scalebox{" + str(scalebox) + "}{\n"
        latex += "\\setlength{\\tabcolsep}{1mm}"
        latex += "\\begin{tabular}{|c|"+ "|".join(["c" for _ in range(len(table_entries[0]))]) + "|}\n"
        latex += "\\hline\n"
        latex +="&" + " & ".join(col_names) + "\\\\\n"
        latex += "\\hline\n"
        for i in range(len(row_names)):
            row_str = row_names[i]
            latex += row_str + " & " + " & ".join(table_entries[i]) + "\\\\\n"
        latex += "\\hline\n"
        latex += "\\end{tabular}"
        latex += "}\n"
        latex += "\\caption{" + caption + ".}\n"
        if label is not None:
            latex += "\\label{" + label + "}\n"
        latex += "\\end{" + env + "}\n"
        return latex

    else:
        col_widths = [max(len(col), max(len(row[col_idx]) for row in table_entries)) for col_idx, col in enumerate(col_names)]
        row_header_width = max(len(row) for row in row_names)

        # Add extra spacing for readability
        col_widths = [w + 2 for w in col_widths]
        row_header_width += 2

        # Helper function to format rows
        def format_row(row):
            return " | ".join(f"{str(cell).center(width)}" for cell, width in zip(row, col_widths))

        # Prepare the string output
        output = []

        # Add the header row
        header = col_names
        output.append("-" * (sum(col_widths) + len(col_widths) * 3 + row_header_width))
        output.append(f"{' ' * row_header_width}| {format_row(header)}")
        output.append("-" * (sum(col_widths) + len(col_widths) * 3 + row_header_width))

        # Add each row of the table
        for row_idx, row_name in enumerate(row_names):
            row_data = table_entries[row_idx]
            formatted_row = f"{row_name.ljust(row_header_width - 1)}| {format_row(row_data)}"
            output.append(formatted_row)

        # Add footer
        output.append("-" * (sum(col_widths) + len(col_widths) * 3 + row_header_width))
        output.append(f"Caption: {caption}")

        return "\n".join(output)

def parse_to_table(mean_and_confint,
                         reduction_func: Callable,
                         filter_by: list,
                         row_by: Union[str, List[str]],
                         col_by: Union[str, List[str]],
                         row_name_map: Callable = lambda x: str(x[0] if len(x) == 1 else x),
                         col_name_map: Callable = lambda x: str(x[0] if len(x) == 1 else x),
                         row_order: List[str] = None, #if rows are known beforehand, one can explicitly specify their order
                         num_top_key_prints = None,
                         assert_unique = False,
                         missing_attr_default = {}
                         ):

    if len(filter_by) == 0 or not isinstance(filter_by[0],list):
        filter_by = [filter_by]

    if not isinstance(row_by,list):
        row_by = [row_by]
    if not isinstance(col_by,list):
        col_by = [col_by]
    filter_by = [[(filter[0], [filter[1]]) if not isinstance(filter[1],list) else filter for filter in filter_attr] for filter_attr in filter_by]

    for key in list(mean_and_confint.keys()):
        key_dict = dict(key)

        missing_attrs = []
        for row in row_by:
            if row not in key_dict:
                if row in missing_attr_default:
                    missing_attrs.append((row,missing_attr_default[row]))
                else:
                    missing_attrs.append((row,"None"))

        for col in col_by:
            if col not in key_dict:
                if col in missing_attr_default:
                    missing_attrs.append((col,missing_attr_default[col]))
                else:
                    missing_attrs.append((col,"None"))

        for filter in filter_by:
            for filter_attr in filter:
                if filter_attr[0] not in key_dict:
                    if filter_attr[0] in missing_attr_default:
                        missing_attrs.append((filter_attr[0],missing_attr_default[filter_attr[0]]))
                    else:
                        missing_attrs.append((filter_attr[0],"None"))

        old_key = key
        old_val = mean_and_confint[key]
        del mean_and_confint[key]
        new_key = tuple(list(old_key) + missing_attrs)
        if new_key in mean_and_confint:
            raise ValueError(f"Duplicate keys: {new_key} (possible obtained after adding missing attributes to {old_key})")
        mean_and_confint[new_key] = old_val


    rows_cols_to_values = {}
    for key in mean_and_confint.keys():
        key_dict = dict(key)

        table_key =  tuple([key_dict[row] for row in row_by] + [key_dict[col] for col in col_by])
        if any([all([filter[0] in key_dict and any(key_dict[filter[0]] == filter_val for filter_val in filter[1]) for filter in filter_attr]) for filter_attr in filter_by]):
            if table_key in rows_cols_to_values:
                assert not assert_unique
                rows_cols_to_values[table_key].append((key,mean_and_confint[key][0]))
            else:
                rows_cols_to_values[table_key] = [(key,mean_and_confint[key][0])]

    row_cols_to_key = {}
    for key in rows_cols_to_values.keys():
        row_cols_to_key[key] = reduction_func(rows_cols_to_values[key], key = lambda x: x[1])[0]

    if num_top_key_prints is not None:

        def cmp(x,y):
            x = x[1]
            y = y[1]
            if reduction_func(x,y) == x and reduction_func(y,x) == y: return 0
            if reduction_func(x,y) == x: return 1
            if reduction_func(x,y) == y: return -1
            #default
            if x < y: return -1
            if x > y: return 1
            return 0

        # subkey_to_val = {}
        for key in row_cols_to_key.keys():
            sorted_keys = sorted(rows_cols_to_values[key], key = cmp_to_key(cmp), reverse=True)
            print(f"Top {num_top_key_prints} keys for table entry {key}:")
            for i in range(min(num_top_key_prints,len(sorted_keys))):
                print(f"{sorted_keys[i][0]}: {sorted_keys[i][1]}")

        #     range_val = sorted_keys[0][1] - sorted_keys[-1][1]
        #     print(f"Range: {range_val}")
        #     if range_val > 0:
        #         for subkey in sorted_keys:
        #             subkey_val = subkey[1]
        #             subkey = dict(subkey[0])
        #             assert len(row_by) == 1 and len(col_by) == 1
        #             del subkey[row_by[0]]
        #             del subkey[col_by[0]]
        #             assert False #find better solution
        #             if "map" in subkey: del subkey["map"]
        #             if "dense_rewards" in subkey: del subkey["dense_rewards"]
        #             if "stds" in subkey: del subkey["stds"]
        #             if "means" in subkey: del subkey["means"]
        #             if "repeats" in subkey: del subkey["repeats"]
        #
        #             if tuple(subkey.items()) not in subkey_to_val:
        #                 subkey_to_val[tuple(subkey.items())] = 0
        #             subkey_to_val[tuple(subkey.items())] += (subkey_val - sorted_keys[-1][1]) / range_val
        #
        # best_subkey = max(subkey_to_val, key = lambda x: subkey_to_val[x])
        # print(f"Best subkey: {best_subkey} with value {subkey_to_val[best_subkey]}")



    #create table data
    rows = list(set([ tuple([dict(key)[row] for row in row_by]) for key in row_cols_to_key.values()]))
    cols = list(set([ tuple([dict(key)[col] for col in col_by]) for key in row_cols_to_key.values()]))

    #check if all entries of rows are of the same type:
    rows = sorted(rows, key = lambda x: row_name_map(x))
    if row_order is not None:
        if set(row_order) != set(rows):
            raise ValueError(f"Row order does not contain all rows {set(row_order)} != {set(rows)}")
        rows = row_order

    #chcek if all entries of cols are of the same type:
    cols = sorted(cols, key = lambda x: col_name_map(x))

    table = [[(math.inf,0) for _ in range(len(cols))] for _ in range(len(rows))]
    row_names = [row_name_map(row) for row in rows]
    col_names = [col_name_map(col) for col in cols]

    for key in row_cols_to_key.values():
        row = rows.index(tuple([dict(key)[row] for row in row_by]))
        col = cols.index(tuple([dict(key)[col] for col in col_by]))
        table[row][col] = mean_and_confint[key]

    return table, row_names, col_names