
#pragma once

#ifndef OGAAGENT_H
#define OGAAGENT_H
#include <map>

#include "OgaGroundNodes.h"
#include "../Agent.h"
#endif

namespace OGA
{
    struct OgaBudget{
        int amount;
        std::string quantity;
    };

    struct OgaSearchStats{
        OgaBudget budget;
        int completed_iterations{};
        unsigned total_forward_calls{};
        unsigned max_depth{};

        //For global std exploration factor
        double total_squared_v{};
        double total_v{};
        int global_num_vs = 0;

        //For abstraction dropping
        bool stop_abstraction = false;
    };

    struct OgaArgs
    {
        OgaBudget budget;
        unsigned recency_count_limit = 3;
        double exploration_parameter = 2.0; //multiplied with the dynamic (global std) exp param
        double discount = 1.0;
        int num_rollouts = 1;
        int rollout_length = -1;
        OgaBehaviorFlags behavior_flags;

        /*
         * For abstraction dropping
         */

        //Compression rate dependent all-at-once dropping (drop_threshold = inf => EMCTS naive dropping)
        double drop_check_point = 1.1; //If > 1 then no smart-abstraction dropping
        double drop_threshold = std::numeric_limits<double>::max(); //If compression rate less than this threshold, then drop
    };


    class OgaAgent final : public Agent
    {
    private:
        OgaStateNode* selectSuccessorState(OgaTree* tree, OgaStateNode* node, ABS::Model* model, OgaSearchStats& search_stats, std::mt19937& rng,
                                           bool* new_state) const;
        OgaStateNode* treePolicy(OgaTree* tree, ABS::Model* model, OgaSearchStats& search_stats, std::mt19937& rng);
        double rollout(const OgaStateNode* leaf, ABS::Model* model, std::mt19937& rng) const;
        void backup(OgaTree* tree, OgaStateNode* leaf, double values, OgaSearchStats& search_stats) const;
        int selectAction(const OgaStateNode* node, bool greedy, OgaSearchStats& search_stats, std::mt19937& rng) const;

        double exploration_parameter;
        double discount;
        int num_rollouts;
        int rollout_length;
        unsigned recency_count_limit;
        OgaBudget budget;
        const OgaArgs args;

        //Drop params
        double drop_check_point;
        double drop_threshold;

        //Statistics
        std::map<size_t,int> num_abs_drops;
        std::map<size_t,int> total_nontrivial_abs;
        int total_action_calls = 0;

        constexpr static double TIEBREAKER_NOISE = 1e-6;

    public:
        explicit OgaAgent(const OgaArgs& args);
        int getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng) override;
        int getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng, OgaTree** treePtr); // Used for testing

        [[nodiscard]] std::map<size_t,int> getNumAbsDrops() const {return num_abs_drops;}
        [[nodiscard]] std::map<size_t,int> getTotalNontrivialAbs() const {return total_nontrivial_abs;}
        [[nodiscard]] int getTotalActionCalls() const {return total_action_calls;}

        static void runTests();
    };
}
