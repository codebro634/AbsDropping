#pragma once

#ifndef MCTSAGENT_H
#define MCTSAGENT_H
#include "../Agent.h"
#include "MctsNode.h"
#endif


namespace Mcts
{
    struct MctsBudget{
        int amount;
        std::string quantity;
    };

    struct MctsSearchStats
    {
        MctsBudget budget;
        int completed_iterations{};
        int total_forward_calls{};
        int max_depth{};

        std::map<int,gsToNodeMap<MctsNode*>>* layerMap;

        //For global std exploration factor
        std::vector<double> total_squared_v;
        std::vector<double> total_v;
        int global_num_vs = 0;

        //For possible hindsight determinizations
        bool deterministic_env=false;
        bool determ_var_reduction=false;
        std::vector<int> layer_seeds;
        int determinization_salt;
    };

    struct MctsArgs
    {
        MctsBudget budget;
        std::vector<double> exploration_parameters = {1};
        double discount = 1.0;
        int num_rollouts = 1;
        int rollout_length = -1;
        bool dag = false;
        bool dynamic_exploration_factor = false;

        //For the paper "Monte Carlo Tree Search With Iteratively RefiningState Abstractions"
        bool wirsa = false;
        double a{},b{};
    };

    class MctsAgent final : public Agent
    {
    public:
        explicit MctsAgent(const MctsArgs& args);
        int getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng) override;
        MctsNode* buildTree(ABS::Model* model, ABS::Gamestate* state, MctsSearchStats& search_stats, std::mt19937& rng, bool determinize_env = false, bool determ_var_reduction = false);
        void cleanupTree(MctsNode* root);

    private:
        std::vector<std::tuple<MctsNode*,int,std::vector<double>>> treePolicy(ABS::Model* model, MctsNode* node, std::mt19937& rng, MctsSearchStats& search_stats);
        MctsNode* selectNode(ABS::Model* model, MctsNode* node, bool& reached_leaf, int &chosen_action, std::vector<double>& rewards, std::mt19937& rng, MctsSearchStats& search_stats);
        int selectAction(MctsNode* node, bool greedy, std::mt19937& rng, MctsSearchStats& search_stats);
        std::vector<double> rollout(ABS::Model* model, MctsNode* node, std::mt19937& rng) const;
        void backup(std::vector<double> values, std::vector<std::tuple<MctsNode*,int,std::vector<double>>>& path, MctsSearchStats& search_stats) const;

        std::vector<double> exploration_parameters;
        double discount;
        MctsBudget budget;
        int num_rollouts;
        bool dag;
        bool dynamic_exploration_factor;
        int rollout_length;
        bool wirsa;
        double a,b;
        constexpr static double TIEBREAKER_NOISE = 1e-6;
    };

}