#include "../../include/Utils/MiscAnalysis.h"
#include "../../include/Arena.h"
#include "../../include/Utils/ValueIteration.h"
#include <fstream>
#include <iomanip>
#include <sstream>

#include "../../include/Agents/RandomAgent.h"

#include "../../include/Games/MDPs/SailingWind.h"
#include "../../include/Games/MDPs/Navigation.h"
#include "../../include/Games/MDPs/SkillsTeaching.h"
#include "../../include/Games/MDPs/SysAdmin.h"
#include "../../include/Games/MDPs/TriangleTireworld.h"
#include "../../include/Games/MDPs/EarthObservation.h"
#include "../../include/Games/MDPs/Manufacturer.h"
#include "../../include/Games/MDPs/GameOfLife.h"
#include "../../include/Games/MDPs/Wildfire.h"
#include "../../include/Games/MDPs/Tamarisk.h"
#include "../../include/Games/Wrapper/FiniteHorizon.h"
#include "../../include/Games/MDPs/RaceTrack.h"
#include "../../include/Games/MDPs/AcademicAdvising.h"
#include "../../include/Games/MDPs/Traffic.h"
#include "../../include/Games/MDPs/CooperativeRecon.h"
#include "../../include/Utils/Argparse.h"
#include "../../include/Utils/Distributions.h"
#include "../../include/Agents/Oga/OgaAgent.h"

#include <filesystem>
#include <algorithm>
#include <iostream>

#include "../../include/Utils/MemoryAnalysis.h"

namespace MISC{

double round(double d, int precision) {
    return std::round(d * std::pow(10, precision)) / std::pow(10, precision);
}

void estimateAbsDropNumbers() {

    std::vector<std::pair<std::string,ABS::Model*>> model_list = {};
    model_list.push_back({"Academic Advising",new  AA::Model("../resources/AcademicAdvisingCourses/2_Anand.txt",false,false)});
    model_list.push_back({"Cooperative Recon", new  RECON::ReconModel("../resources/CooperativeReconSetups/3_IPPC.txt")});
    model_list.emplace_back("Game of Life", new  GOL::Model("../resources/GameOfLifeMaps/3_Anand.txt", GOL::ActionMode::SAVE_ONLY));
    model_list.emplace_back("Earth of Observation", new  EO::Model("../resources/EarthObservationMaps/1_IPPC.txt"));
    model_list.emplace_back("Manufacturer", new  MAN::Model("../resources/ManufacturerSetups/3_IPPC.txt"));
    model_list.emplace_back("Navigation", new  Navigation::Model("../resources/NavigationMaps/3_Anand.txt",false));
    model_list.emplace_back("Racetrack", new  RT::Model("../resources/Racetracks/ring-2.track", 0.0, false));
    model_list.emplace_back( "Sailing Wind", new  SW::Model(15,15, false));
    model_list.emplace_back("Skills Teaching", new  ST::SkillsTeachingModel("../resources/SkillsTeachingSkills/5_IPPC.txt",false, true));
    model_list.emplace_back("SysAdim", new  SA::Model("../resources/SysAdminTopologies/4_Anand.txt"));
    model_list.emplace_back("Tamarisk", new  TAM::Model("../resources/TamariskMaps/2_IPPC.txt"));
    model_list.emplace_back("Traffic", new  TR::TrafficModel("../resources/TrafficModels/1_IPPC.txt"));
    model_list.emplace_back("Triangle Tireworld", new  TRT::Model("../resources/TriangleTireworlds/5_IPPC.txt",false,true));

    std::map<std::string,std::string> abbreviation_map = {
        {"Academic Advising","aa"},
        {"Cooperative Recon","recon"},
        {"Game of Life","gol"},
        {"Earth of Observation","eo"},
        {"Manufacturer", "man"},
        {"Navigation","navigation"},
        {"Racetrack","rt"},
        {"Sailing Wind","sw"},
        {"Skills Teaching","st"},
        {"SysAdim","sa"},
        {"Tamarisk","tam"},
        {"Traffic","tr"},
        {"Triangle Tireworld","trt"}};

    std::cout << R"(\begin{table*}[]\centering \scalebox{1.0}{\setlength{\tabcolsep}{1mm}\begin{tabular}{ |c|c|c|c|c|c|c|} \hline Domain & C-1 & C-2 & C-T  & F-1 & F-2 & F-T\\ \hline)" << std::endl;

    for(auto& [name, model] : model_list) {
        //std::cout << "----------- Model:" << name << " ----------- " << std::endl;

        std::cout << name << " & ";

        /*
         * Find the parameters for the oga agent for this environment
         */

        int rolloutLength = std::numeric_limits<int>::max();
        double max_eps_a = std::numeric_limits<double>::lowest();
        double expfac = std::numeric_limits<double>::lowest();

        std::ifstream file("../ExperimentConfigs/AbsDropping/evalLowVarCAD.yml");
        if (!file.is_open()) {
            std::cerr << "Error opening config file" << std::endl;
            return;
        }
        bool foundAgent = false;
        std::string line;
        while (std::getline(file, line)) {

            if (!foundAgent && line.find(std::string("oga_eps_") + abbreviation_map[name]) != std::string::npos)
                foundAgent = true;

            if (foundAgent && line.find("rollout_length:") != std::string::npos) {
                // Find the first '[' and ']' characters
                size_t start = line.find('[');
                size_t end = line.find(']', start);
                if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                    std::string valueStr = line.substr(start + 1, end - start - 1);
                    rolloutLength = std::stoi(valueStr);
                }
            }

            if (foundAgent && line.find("expfac") != std::string::npos) {
                // Find the first '[' and ']' characters
                size_t start = line.find('[');
                size_t end = line.find(']', start);
                if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                    std::string valueStr = line.substr(start + 1, end - start - 1);
                    expfac = std::stoi(valueStr);
                }
            }

            if (foundAgent && line.find("eps_a:") != std::string::npos) {
                //Iterate through all values. A line has the following format: eps_a: [0, 0.75, 1.5]
                size_t start = line.find('[');
                size_t end = line.find(']', start);
                if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                    std::string valueStr = line.substr(start + 1, end - start - 1);
                    std::stringstream ss(valueStr);
                    double i;
                    while (true) {
                        ss >> i;
                        max_eps_a = std::max(max_eps_a, i);
                        if (ss.peek() == ',' || ss.peek() == ' ')
                            ss.ignore();
                        if (ss.eof())
                            break;
                    }
                }
            }

            if (rolloutLength != std::numeric_limits<int>::max() && max_eps_a != std::numeric_limits<double>::lowest() && expfac != std::numeric_limits<double>::lowest())
                break;

        }
        file.close();

        if (rolloutLength == std::numeric_limits<int>::max())
            throw std::runtime_error("Could not find the rollout_length in the AbsDropping config file");
        if (max_eps_a == std::numeric_limits<double>::lowest())
            throw std::runtime_error("Could not find the eps_a in the AbsDropping config file");
        if (expfac == std::numeric_limits<double>::lowest())
            throw std::runtime_error("Could not find the expfac in the AbsDropping config file");

        //std::cout << "Rollout length: " << rolloutLength << " Max eps_a: " << max_eps_a << " Expfac: " << expfac << std::endl;

        /*
         * Setup the OGA agents
         */
        double p = 0.9;
        auto oga_fine = OGA::OgaAgent({
                       .budget={
                           1000,
                           "iterations"
                       },
                       .recency_count_limit=1,
                       .exploration_parameter=expfac,
                       .discount=1.0,
                       .num_rollouts = 10,
                        .rollout_length = rolloutLength,
                       .behavior_flags={
                           .exact_bookkeeping=true,
                           .group_terminal_states=true,
                           .group_partially_expanded_states=false,
                           .partial_expansion_group_threshold=999,
                           .eps_a = 0,
                           .eps_t = 0.0,
                           .drop_confidence = p,
                           .drop_at_visits = 10000000
                       },
                        .drop_check_point = 1.1,
                        .drop_threshold = 10000000000
                   });

        auto oga_coarse = OGA::OgaAgent({
                        .budget={
                           1000,
                           "iterations"
                        },
                        .recency_count_limit=1,
                        .exploration_parameter=expfac,
                        .discount=1.0,
                        .num_rollouts = 10,
                        .rollout_length = rolloutLength,
                        .behavior_flags={
                           .exact_bookkeeping=true,
                           .group_terminal_states=true,
                           .group_partially_expanded_states=true,
                           .partial_expansion_group_threshold=999,
                           .eps_a = max_eps_a,
                           .eps_t = 2.0,
                           .drop_confidence = p,
                           .drop_at_visits = 10000000
                        },
                        .drop_check_point = 1.1,
                        .drop_threshold = 10000000000
                        });

        const int num_maps = 20;
        const double shift = 100;

        std::mt19937 rng1(static_cast<unsigned int>(42));
        playGames(*model, num_maps, {&oga_coarse}, rng1, MUTED, {50,50}, false, true);
        auto drops = oga_coarse.getNumAbsDrops();
        auto abs = oga_coarse.getTotalNontrivialAbs();
        double root_rate, depth1_rate,total_drops=0, total_abs=0;
        root_rate = abs.contains(0)? drops[0] / (double) abs[0] : std::nan("");
        depth1_rate = abs.contains(1)? drops[1] / (double) abs[1] : std::nan("");
        for (auto& [depth, drop] : drops){
            if (depth >= 2) {
                total_drops += drop;
                total_abs += abs[depth];
            }
        }
        double tree_rate = total_abs > 0? total_drops / (double) total_abs  : std::numeric_limits<double>::quiet_NaN();

        //std::cout << "Coarse average drop rates: " << shift * root_rate << " " << shift * depth1_rate << " " << shift * tree_rate << std::endl;
        std::cout <<  round(shift * root_rate,4) << "\\% & " << round(shift * depth1_rate,4) << "\\%  & " << round(shift * tree_rate,4)<< "\\% & ";

        std::mt19937 rng2(static_cast<unsigned int>(42));
        playGames(*model, num_maps, {&oga_fine}, rng2, MUTED, {50,50}, false, true);
        drops = oga_fine.getNumAbsDrops();
        abs = oga_fine.getTotalNontrivialAbs();
        root_rate = abs.contains(0)? drops[0] / (double) abs[0] : std::nan("");
        depth1_rate = abs.contains(1)? drops[1] / (double) abs[1] : std::nan("");
        total_drops = 0;
        total_abs = 0;
        for (auto& [depth, drop] : drops){
            if (depth >= 2) {
                total_drops += drop;
                total_abs += abs[depth];
            }
        }
        tree_rate = total_abs > 0? total_drops / (double) total_abs  : std::nan("");

        //std::cout << "Fine average drop rates: "  << shift * root_rate << " " << shift * depth1_rate << " " << shift * total_drops / (double) total_abs << std::endl;
        std::cout <<  round(shift * root_rate,4) << "\\% & " << round(shift * depth1_rate,4) << "\\% & " << round(shift * tree_rate,4) << R"(\% \\ \hline)" << std::endl;
    }

    std::cout << R"(\end{tabular}} \caption{Average ratio of abstraction-dropped Q nodes in OGA-CAD with $p=0.9$ that are part of non-trivial abstractions. The average is denoted for the entire tree, the first, or the second layer only. The columns have the format in which the first entry denotes whether a coarse (C) or fine abstraction has been used (F) and the second entry denotes the layer where T denotes that the entire tree starting at layer 3 has been taken.}  \label{tab:oga_cad_stats} \end{table*})" << std::endl;


    //free models
    for(auto& [name, model] : model_list)
        delete model;
   
}

void createQTable(ABS::Model* ground_model, std::unordered_map<std::pair<FINITEH::Gamestate*,int> , double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare>* Q_map_ptr,std::string save_path, bool verbose, int time_limit){
    int horizon = 50;
    FINITEH::Model* model = new FINITEH::Model(ground_model,horizon, true);

    std::unordered_map<std::pair<FINITEH::Gamestate*,int> , double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare> Q_map = {};
    std::mt19937 rng(static_cast<unsigned int>(42));
    int num_start_samples = 1000;
    VALUE_IT::runValueIteration(model,num_start_samples,1.0,Q_map,rng,verbose, time_limit);
    if (!save_path.empty())
        VALUE_IT::saveQTable(Q_map,save_path, true);
    if (Q_map_ptr != nullptr)
        *Q_map_ptr = Q_map;

//    double val = 0;
//    int num_samples = 1;
//    int done = 0;
//    while(done < num_samples) {
//        if(done % 1000 == 0)
//            std::cout << "Done:" << done << std::endl;
//        auto init_state = dynamic_cast<FINITEH::Gamestate *>(model.getInitialState(rng));
//        double max_aval = std::numeric_limits<double>::lowest();
//        bool broke = false;
//        for(int action : model.getActions(init_state)) {
//            if(!Q_map.contains({init_state,action})){
//                broke = true;
//                 break;
//            }
//            max_aval = std::max(max_aval,Q_map.at({init_state,action}));
//        }
//        if(!broke) {
//            done++;
//            val += max_aval;
//        }
//    }
//    std::cout << "Value:" << val/num_samples << std::endl;
//    auto init_state = dynamic_cast<FINITEH::Gamestate *>(model->getInitialState(rng));
//    for(int action : model->getActions(init_state)) {
//        std::cout << "Action:" << Q_map.at({init_state,action}) << std::endl;
//    }
}


std::vector<ABS::Gamestate*> gatherSmallQDiffStates(ABS::Model& uncasted_model, unsigned int num_states,
            std::unordered_map<std::pair<FINITEH::Gamestate*,int>, double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare>* Q_map,
            Agent* agent, std::mt19937& rng) {
    assert (dynamic_cast<FINITEH::Model*>(&uncasted_model));
    auto model = dynamic_cast<FINITEH::Model*>(&uncasted_model);

    auto cmp = [](std::pair<ABS::Gamestate*,double> a, std::pair<ABS::Gamestate*,double> b) {
        return a.second < b.second;
    };
    std::set<std::pair<ABS::Gamestate*,double>,decltype(cmp)> states;

    if(num_states >= Q_map->size()) {
        for(auto& [key, value] : *Q_map) {
            if (key.first->remaining_steps == model->getHorizonLength())
                states.insert({model->unwrapState(key.first), value});
        }
    }else {

        double TOL = 1e-4;

        //filter for states with planning horizon steps
        auto reduced_map = std::vector<std::pair<std::pair<FINITEH::Gamestate*,int>, double>>();
        for(auto& [key, value] : *Q_map)
            if(key.first->remaining_steps == model->getHorizonLength())
                reduced_map.emplace_back(key,value);

        double top_ratio = 0.1;
        int num_samples = std::ceil(num_states / top_ratio);

        for(int i = 0; i < num_samples; i++){
            //sample from reduced map using rng
            assert (!reduced_map.empty());
            auto dist = std::uniform_int_distribution<int>(0,reduced_map.size()-1);
            auto& [key, value] = reduced_map[dist(rng)];

            //get value diff between optimal and second best action
            double best_val = std::numeric_limits<double>::lowest();
            double second_best_val = std::numeric_limits<double>::lowest();
            for(int action : model->getActions(key.first)) {
                double aval = Q_map->at(std::make_pair(key.first, action));
                if(aval > best_val + TOL) {
                    second_best_val = best_val;
                    best_val = aval;
                } else if(std::fabs(aval - best_val) <= TOL) {
                    //pass
                }
                else if(aval > second_best_val + TOL)
                    second_best_val = aval;
            }
            double diff = best_val - second_best_val;

            //insert into sorted list
            states.insert({key.first,diff});
            if(states.size() > num_states)
                states.erase(--states.end());

        }
    }

    auto states_vec = std::vector<ABS::Gamestate*>();
    for(auto& [state, diff] : states){
        std::cout << "State: " << state << " Diff: " << diff << std::endl;
        states_vec.push_back(state);
    }

    return states_vec;
}

}
