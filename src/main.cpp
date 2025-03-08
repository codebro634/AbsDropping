#define DEBUG


#include <random>
#include "../include/Arena.h"
#include "../include/Agents/Oga/OgaAgent.h"
#include "../include/Agents/Mcts/MctsAgent.h"
#include "../include/Agents/RandomAgent.h"
#include "../include/Games/MDPs/Traffic.h"
#include "../include/Agents/HumanAgent.h"
#include "../include/Agents/Mcts/MctsAgent.h"
#include "../include/Games/MDPs/SailingWind.h"
#include "../include/Games/MDPs/Navigation.h"
#include "../include/Games/MDPs/SkillsTeaching.h"
#include "../include/Games/MDPs/SysAdmin.h"
#include "../include/Games/MDPs/TriangleTireworld.h"
#include "../include/Games/MDPs/EarthObservation.h"
#include "../include/Games/MDPs/Manufacturer.h"
#include "../include/Games/MDPs/GameOfLife.h"
#include "../include/Games/MDPs/Wildfire.h"
#include "../include/Games/MDPs/Tamarisk.h"
#include "../include/Games/Wrapper/FiniteHorizon.h"
#include "../include/Games/MDPs/RaceTrack.h"
#include "../include/Games/MDPs/AcademicAdvising.h"
#include "../include/Games/MDPs/CooperativeRecon.h"
#include "../include/Utils/Argparse.h"
#include "../include/Utils/Distributions.h"
#include "../include/Utils/ValueIteration.h"
#include "../include/Utils/MiscAnalysis.h"

void debug(){

}

std::string extraArgs(std::map<std::string, std::string>& given_args, const std::set<std::string>& acceptable_args){
    for (auto& [key, val] : given_args) {
        if(!acceptable_args.contains(key))
            return key;
    }
    return "";
}

Agent* getDefaultAgent(){
    return new RandomAgent();
}

inline ABS::Model* getModel(const std::string& model_type, const std::vector<std::string>& m_args)
{

    std::map<std::string, std::string> model_args;
    for(auto &arg : m_args) {
        //split at '='
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid agent argument");
        }
        model_args[arg.substr(0, pos)] = arg.substr(pos + 1);
    }

    ABS::Model *model = nullptr;
    std::set<std::string> acceptable_args;

    if(model_type == "man"){
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model = new MAN::Model(model_args["map"]);
    }
    else if (model_type == "wf") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model =  new WF::Model(model_args["map"]);
    }
    else if (model_type == "tam") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model =  new TAM::Model(model_args["map"]);
    }else if(model_type == "recon") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model = new RECON::ReconModel(model_args["map"]);
    }
    else if(model_type == "tr") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model = new TR::TrafficModel(model_args["map"]);
    }
    else if(model_type == "st") {
        assert (model_args.contains("map"));
        acceptable_args = {"map", "idle_action", "reduced_action_space"};
        bool idle_action = model_args.contains("idle_action") ? std::stoi(model_args["idle_action"]) : false;
        bool reduced_action_space = model_args.contains("reduced_action_space") ? std::stoi(model_args["reduced_action_space"]) : true;
        model = new ST::SkillsTeachingModel(model_args["map"],idle_action, reduced_action_space);
    }else if(model_type == "trt") {
        assert (model_args.contains("map"));
        acceptable_args = {"map", "idle_action", "reduced_action_space"};
        bool idle_action = model_args.contains("idle_action") ? std::stoi(model_args["idle_action"]) : false;
        bool reduced_action_space = model_args.contains("reduced_action_space") ? std::stoi(model_args["reduced_action_space"]) : true;
        model = new TRT::Model(model_args["map"],idle_action, reduced_action_space);
    }
    else if (model_type == "sa") {
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model = new SA::Model(model_args["map"]);
    }
    else if (model_type == "eo"){
        assert (model_args.contains("map"));
        acceptable_args = {"map"};
        model = new EO::Model(model_args["map"]);
    }
    else if (model_type == "sw") {
        assert (model_args.contains("size"));
        acceptable_args = {"size", "deterministic"};
        bool deterministic = model_args.contains("deterministic") ? std::stoi(model_args["deterministic"]) : false;
        model =  new SW::Model(std::stoi(model_args["size"]), std::stoi(model_args["size"]), deterministic);
    }
    else if (model_type == "rt") {
        assert (model_args.contains("map"));
        acceptable_args = {"map","reset_at_crash","fail_prob"};
        bool reset_at_crash = model_args.contains("reset_at_crash") ? std::stoi(model_args["reset_at_crash"]) : false;
        double fail_prob = model_args.contains("fail_prob") ? std::stod(model_args["fail_prob"]) : 0.0;
        model =  new RT::Model(model_args["map"], fail_prob, reset_at_crash);
    }
    else if (model_type == "gol") {
        assert (model_args.contains("map"));
        acceptable_args = {"map", "action_mode"};
        GOL::ActionMode action_mode = GOL::ActionMode::SAVE_ONLY;
        if (model_args.contains("action_mode")){
            if (model_args["action_mode"] == "all")
                action_mode = GOL::ActionMode::ALL;
            else if (model_args["action_mode"] == "save_only")
                action_mode = GOL::ActionMode::SAVE_ONLY;
            else if (model_args["action_mode"] == "revive_only")
                action_mode = GOL::ActionMode::REVIVE_ONLY;
            else{
                std::cout << "Invalid action mode" << std::endl;
                throw std::runtime_error("Invalid action mode");
            }
        }
        model =  new GOL::Model(model_args["map"],action_mode);
    }
    else if (model_type == "aa") {
        assert (model_args.contains("map") && model_args.contains("dense_rewards"));
        acceptable_args = {"map", "dense_rewards", "idle_action"};
        bool idle_action = model_args.contains("idle_action") ? std::stoi(model_args["idle_action"]) : false;
        model =  new AA::Model(model_args["map"], std::stoi(model_args["dense_rewards"]), idle_action);
    }
    else if (model_type == "navigation") {
        assert (model_args.contains("map"));
        acceptable_args = {"map", "idle_action"};
        bool idle_action = model_args.contains("idle_action") ? std::stoi(model_args["idle_action"]) : false;
        model =  new Navigation::Model(model_args["map"],idle_action);
    }


    if (model != nullptr) {
        if (!extraArgs(model_args, acceptable_args).empty()) {
            std::string err_string = "Invalid model argument: " + extraArgs(model_args, acceptable_args);
            std::cout << err_string << std::endl;
            throw std::runtime_error(err_string);
        }
        return model;
    }else {
        std::cout << "Invalid model" << std::endl;
        throw std::runtime_error("Invalid model");
    }

}

inline Agent* getAgent(const std::string& agent_type, const std::vector<std::string>& a_args)
{

    //Parse named args
    std::map<std::string, std::string> agent_args;
    for(auto &arg   : a_args) {
        //split at '='
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            std::cout << "Invalid agent argument: " << arg << ". It must be of the form arg_name=arg_val" << std::endl;
            return nullptr;
        }
        agent_args[arg.substr(0, pos)] = arg.substr(pos + 1);
    }
    std::set<std::string> acceptable_args;

    Agent* agent;
    if (agent_type == "random") {
        acceptable_args = {};
        agent =  new RandomAgent();
    }
    else if(agent_type == "mcts")
    {
        assert (agent_args.contains("iterations"));
        if(agent_args.contains("wirsa"))
            assert (agent_args.contains("a") && agent_args.contains("b"));
        acceptable_args = {"iterations", "rollout_length", "discount", "num_rollouts", "dag", "dynamic_exp_factor", "expfacs", "wirsa", "a", "b"};

        int iterations = std::stoi(agent_args["iterations"]);
        int rollout_length = agent_args.find("rollout_length") == agent_args.end() ? -1 : std::stoi(agent_args["rollout_length"]);
        double discount = agent_args.find("discount") == agent_args.end() ? 1.0 : std::stod(agent_args["discount"]);
        int num_rollouts = agent_args.find("num_rollouts") == agent_args.end() ? 1: std::stoi(agent_args["num_rollouts"]);
        bool dag = agent_args.find("dag") == agent_args.end() ? false : std::stoi(agent_args["dag"]);
        bool dynamic_exp_factor = agent_args.find("dynamic_exp_factor") == agent_args.end() ? false : std::stoi(agent_args["dynamic_exp_factor"]);
        bool wirsa = agent_args.find("wirsa") == agent_args.end() ? false : std::stoi(agent_args["wirsa"]);
        double a = agent_args.find("a") == agent_args.end() ? 0.0 : std::stod(agent_args["a"]);
        double b = agent_args.find("b") == agent_args.end() ? 0.0 : std::stod(agent_args["b"]);
        std::string exp_facs = agent_args.find("expfacs") == agent_args.end() ? "1" : agent_args["expfacs"];
        std::vector<double> expfac;
        std::stringstream ss(exp_facs);
        double i;
        while (ss >> i){
            expfac.push_back(i);
            if (ss.peek() == ';')
                ss.ignore();
        }
        auto args = Mcts::MctsArgs{.budget = {iterations, "iterations"}, .exploration_parameters = expfac, .discount = discount,
            .num_rollouts = num_rollouts,
            .rollout_length = rollout_length,
            .dag=dag,
            .dynamic_exploration_factor=dynamic_exp_factor,
            .wirsa = wirsa,
            .a=a,.b=b};
        agent =  new Mcts::MctsAgent(args);
    }
     else if (agent_type == "oga") {
        assert (agent_args.contains("iterations"));
        acceptable_args = {"iterations", "discount", "expfac", "K", "exact_bookkeeping", "group_terminal_states", "group_partially_expanded_states",
            "partial_expansion_group_threshold", "ignore_partially_expanded_states", "eps_a", "eps_t",
            "drop_check_point", "drop_threshold", "drop_confidence", "drop_at_visits", "num_rollouts", "rollout_length"};

        int iterations = std::stoi(agent_args["iterations"]);
        double discount = agent_args.find("discount") == agent_args.end() ? 1.0 : std::stod(agent_args["discount"]);
        double expfac = agent_args.find("expfac") == agent_args.end() ? 2.0 : std::stod(agent_args["expfac"]);
        unsigned K = agent_args.find("K") == agent_args.end() ? 1 : std::stoi(agent_args["K"]);
        bool exact_bookkeeping = agent_args.find("exact_bookkeeping") == agent_args.end() ? true : std::stoi(agent_args["exact_bookkeeping"]);
        bool group_terminal_states = agent_args.find("group_terminal_states") == agent_args.end() ? true : std::stoi(agent_args["group_terminal_states"]);
        bool group_partially_expanded_states = agent_args.find("group_partially_expanded_states") == agent_args.end() ? false : std::stoi(agent_args["group_partially_expanded_states"]);
        unsigned partial_expansion_group_threshold = agent_args.find("partial_expansion_group_threshold") == agent_args.end() ? std::numeric_limits<int>::max() : std::stoi(agent_args["partial_expansion_group_threshold"]);
        double eps_a = agent_args.find("eps_a") == agent_args.end() ? 0.0 : std::stod(agent_args["eps_a"]);
        double eps_t = agent_args.find("eps_t") == agent_args.end() ? 0.0 : std::stod(agent_args["eps_t"]);
        double drop_check_point = agent_args.find("drop_check_point") == agent_args.end() ? 1.1 : std::stod(agent_args["drop_check_point"]);
        double drop_threshold = agent_args.find("drop_threshold") == agent_args.end() ? std::numeric_limits<double>::max() : std::stod(agent_args["drop_threshold"]);
        double drop_confidence = agent_args.find("drop_confidence") == agent_args.end() ? -1.0 : std::stod(agent_args["drop_confidence"]);
        unsigned drop_at_visits = agent_args.find("drop_at_visits") == agent_args.end() ? std::numeric_limits<int>::max() : std::stoi(agent_args["drop_at_visits"]);
        int num_rollouts = agent_args.find("num_rollouts") == agent_args.end() ? 1 : std::stoi(agent_args["num_rollouts"]);
        int rollout_length = agent_args.find("rollout_length") == agent_args.end() ? -1 : std::stoi(agent_args["rollout_length"]);

        auto args = OGA::OgaArgs{
            .budget = {iterations, "iterations"},
            .recency_count_limit = K,
            .exploration_parameter = expfac,
            .discount = discount,
            .num_rollouts = num_rollouts,
            .rollout_length = rollout_length,
            .behavior_flags = {
                .exact_bookkeeping=exact_bookkeeping,
                .group_terminal_states=group_terminal_states,
                .group_partially_expanded_states=group_partially_expanded_states,
                .partial_expansion_group_threshold=partial_expansion_group_threshold,
                .eps_a = eps_a,
                .eps_t = eps_t,
                .drop_confidence = drop_confidence,
                .drop_at_visits = drop_at_visits
            },
            .drop_check_point = drop_check_point,
            .drop_threshold = drop_threshold
        };
        agent =  new OGA::OgaAgent(args);
    }
    else{
        throw std::runtime_error("Invalid agent");
    }

    if (agent != nullptr) {
        if (!extraArgs(agent_args, acceptable_args).empty()) {
            std::string err_string = "Invalid agent argument: " + extraArgs(agent_args, acceptable_args);
            std::cout << err_string << std::endl;
            throw std::runtime_error(err_string);
        }
        return agent;
    }else {
        std::cout << "Invalid agent" << std::endl;
        throw std::runtime_error("Invalid agent");
    }
}

int main(const int argc, char **argv) {

    argparse::ArgumentParser program("Executable");

    program.add_argument("-s", "--seed")
        .help("Seed for the random number generator")
        .action([](const std::string &value) { return std::stoi(value); })
        .required();

    program.add_argument("-a", "--agent")
        .help("Agent to benchmark")
        .required();

    program.add_argument("--aargs")
        .help("Extra arguments for agent")
        .default_value(std::vector<std::string>{})
        .append();

    program.add_argument("-m", "--model")
        .help("Model to benchmark")
        .required();

    program.add_argument("--margs")
        .help("Extra arguments for model")
        .default_value(std::vector<std::string>{})
        .append();

    program.add_argument("-n", "--n_games")
        .help("Number of games to play")
        .action([](const std::string &value) { return std::stoi(value); })
        .required();

    program.add_argument("-v", "--csv")
        .help("CSV mode")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-p_horizon", "--p_horizon")
        .help("Planning horizon")
        .action([](const std::string &value) { return std::stoi(value); })
        .default_value(50);

    program.add_argument("-e_horizon", "--e_horizon")
    .help("Execution horizon")
    .action([](const std::string &value) { return std::stoi(value); })
    .default_value(50);

    program.add_argument("--qtable")
    .help("If available, the path to the qtable to load.")
    .default_value("");

    program.add_argument("--planning_beyond_execution_horizon")
    .help("Whether the agent should plan beyond the execution horizon, i.e. always plan for the full planning horizon.")
    .default_value(false)
    .implicit_value(true);

    program.add_argument("--deterministic_init")
    .help("Whether to cycle through the same deterministic init states or sample random ones.")
    .default_value(false)
    .implicit_value(true);

    if (argc == 1) {
        std::cout << "Since no arguments were provided, for IDE convenience, the debug function will be called." << std::endl;
        debug();
        return 0;
    }

    program.parse_args(argc, argv);

    const auto seed = program.get<int>("--seed");
    std::mt19937 rng(seed);

    auto* model = getModel(program.get<std::string>("--model"), program.get<std::vector<std::string>>("--margs"));
    if (model == nullptr ) {
        throw std::runtime_error("Invalid model");
        return 1;
    }
    if (model->getNumPlayers() != 1){
        throw std::runtime_error("Model must be a single player model. Encapsulate it in a MPTOMDP model if necessary.");
        return 1;
    }

    Agent* agent = getAgent(program.get<std::string>("--agent"), program.get<std::vector<std::string>>("--aargs"));
    if (agent == nullptr) {
        delete model;
        return 1;
    }

    auto horizons = std::make_pair(program.get<int>("--e_horizon"), program.get<int>("--p_horizon"));
    bool planning_beyond_execution_horizon = program.get<bool>("--planning_beyond_execution_horizon");
    bool random_init_state = !program.get<bool>("--deterministic_init");
    std::unordered_map<std::pair<FINITEH::Gamestate*,int> , double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare> Q_map = {};
    if (!program.get<std::string>("--qtable").empty()) {
        auto tmp_model = FINITEH::Model(model,1 << 16, false);
        VALUE_IT::loadQTable(&tmp_model, &Q_map, program.get<std::string>("--qtable"));
    }

    playGames(*model, program.get<int>("--n_games"), {agent}, rng, program.get<bool>("--csv") ? CSV: VERBOSE, horizons, planning_beyond_execution_horizon, random_init_state, &Q_map);

    delete model;
    delete agent;
    for (auto& [key, val] : Q_map)
        delete key.first;

    return 0;
}
