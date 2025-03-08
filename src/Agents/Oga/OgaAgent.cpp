#include "../../../include/Agents/Oga/OgaAgent.h"
#include "../../../include/Agents/Oga/OgaTree.h"
#include "../../../include/Agents/Oga/OgaUtils.h"
#include "../../../include/Agents/Oga/OgaAbstractNodes.h"
#include "../../../include/Agents/Oga/OgaGroundNodes.h"

#include <cassert>
#include <cmath>
#include <chrono>
#include <utility>

#include "../../../include/Utils/Distributions.h"

using namespace OGA;

OgaAgent::OgaAgent(const OgaArgs& args) :
    exploration_parameter(args.exploration_parameter),
    discount(args.discount),
    num_rollouts(args.num_rollouts),
    rollout_length(args.rollout_length),
    recency_count_limit(args.recency_count_limit),
    budget(args.budget),
    args(args),
    drop_check_point(args.drop_check_point),
    drop_threshold(args.drop_threshold)
{
    assert (args.exploration_parameter >= 0);
}

int OgaAgent::getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng)
{
    return getAction(model, state, rng, nullptr);
}

int OgaAgent::getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng, OgaTree** treePtr)
{
    assert (model->getNumPlayers() == 1 && model->hasTransitionProbs());

    const auto start = std::chrono::high_resolution_clock::now();

    OgaSearchStats search_stats = {budget, 0, 0,0,0,0,0, false};
    const auto total_forward_calls_before = model->getForwardCalls();

    auto tree = new OgaTree{state, model, args.behavior_flags,rng}; // Empty tree

    bool done = false;
    int abs_check_in = 0;
    while (!done){
        OgaStateNode* leaf = treePolicy(tree, model, search_stats, rng);
        const auto rewards = rollout(leaf, model, rng);
        backup(tree, leaf, rewards, search_stats);
        tree->performUpdateAbstractions(recency_count_limit);

        search_stats.completed_iterations++;
        search_stats.total_forward_calls = model->getForwardCalls() - total_forward_calls_before;

        double done_ratio = 0.0;
        if(budget.quantity == "iterations"){
            done = search_stats.completed_iterations >= budget.amount;
            done_ratio = static_cast<double>(search_stats.completed_iterations) / budget.amount;
        } else if (budget.quantity == "forward_calls"){
            done = static_cast<int>(search_stats.total_forward_calls) >= budget.amount;
            done_ratio = static_cast<double>(search_stats.total_forward_calls) / budget.amount;
        } else if (budget.quantity == "milliseconds"){
            done = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() >= budget.amount;
            done_ratio = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() / budget.amount;
        }

        if (!search_stats.stop_abstraction && done_ratio >= drop_check_point && abs_check_in-- <= 0) {
            abs_check_in = 10; // Check every 10 iterations. Any arbitrary number here would work, but too small numbers may incur a noticeable runtime overhead
            if (tree->getCompressionRate() < drop_threshold)
                search_stats.stop_abstraction = true;
        }
    }

    const int best_action = selectAction(tree->getRoot(), true, search_stats, rng);

    //print layer 0 Q abstractions
    // model->printState(state);
    // tree->printAbsTree(1);
    // //print drop stati
    // for (const auto& q_node : tree->getRoot()->getChildren()){
    //     std::cout << "Q-Node: " << q_node->getAction() << " AbsDropped: " << q_node->isAbsDropped() << " Visits: " << q_node->getVisits() << std::endl;
    //     //print confidence interval and abs q value
    //     auto conf_interval = distr::confidence_interval(q_node->getValues(), q_node->getSquaredValues(), q_node->getVisits(), args.behavior_flags.drop_confidence);
    //     std::cout << "Confidence Interval: " << conf_interval.first << " - " << conf_interval.second << " | " << q_node->getAbsValues() / q_node->getAbsVisits() << std::endl;
    // }

    //update global statistics
    auto [num_abs_drops, total_abs_nodes] = tree->getNumAbsDrops();
    for (const auto& [depth, drops] : num_abs_drops)
        this->num_abs_drops[depth] += drops;
    for (const auto& [depth, total] : total_abs_nodes)
        this->total_nontrivial_abs[depth] += total;
    total_action_calls ++;

    if (treePtr != nullptr)
        *treePtr = tree;
    else
        delete tree;

    return best_action;
}

OgaStateNode* OgaAgent::treePolicy(OgaTree* tree,  ABS::Model* model, OgaSearchStats& search_stats,std::mt19937& rng){
    auto* curr_node = tree->getRoot();

    while (!curr_node->isTerminal())
    {
        if (!curr_node->isFullyExpanded())
        {
            auto* state = curr_node->getStateCopy(model);

            const int action = curr_node->popUntriedAction();
            auto [q_node, found_q] = tree->findOrCreateQState(state, curr_node->getDepth(), action, rng);
            assert(!found_q);

            auto [rewards, prob] = model->applyAction(state, action, rng);
            auto [successor, found] = tree->findOrCreateState(state, curr_node->getDepth() + 1, rng);

            q_node->addChild(model->copyState(state), prob, successor);
            q_node->setRewards(rewards[0]);
            q_node->setParent(curr_node);

            delete state;

            // Trajectory bookkeeping
            successor->setTrajectoryParent(q_node);

            if (!found){
                if (successor->getDepth() > search_stats.max_depth)
                    search_stats.max_depth = successor->getDepth();

                return successor;
            }
            curr_node = successor;
            continue;
        }

        bool new_state;
        curr_node = selectSuccessorState(tree, curr_node, model, search_stats, rng, &new_state);
        if (new_state)
            return curr_node;
    }

    return curr_node;
}

OgaStateNode* OgaAgent::selectSuccessorState(OgaTree* tree, OgaStateNode* node, ABS::Model* model, OgaSearchStats& search_stats, std::mt19937& rng,
                                             bool* new_state) const
{
    const int best_action = selectAction(node, false, search_stats, rng);

    const auto sample_state = node->getStateCopy(model);
    auto [q_node, found_q] = tree->findOrCreateQState(sample_state, node->getDepth(), best_action, rng);
    assert(found_q);

    // Sample successor of state-action-pair
    auto [rewards, prob] = model->applyAction(sample_state, best_action, rng);
    auto [successor, found] = tree->findOrCreateState(sample_state, node->getDepth() + 1, rng);
    *new_state = !found;

    auto state_cpy = model->copyState(sample_state);
    bool added_copy = q_node->addChild(state_cpy, prob, successor);
    if (!added_copy)
        delete state_cpy;

    // Trajectory bookkeeping
    successor->setTrajectoryParent(q_node);

    delete sample_state;
    return successor;
}

int OgaAgent::selectAction(const OgaStateNode* node, const bool greedy, OgaSearchStats& search_stats,std::mt19937& rng) const
{
    // UCT Formula: w/n + c * sqrt(ln(N)/n)
    assert(node->isPartiallyExpanded());

    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    //Determine N from the UCT formula
    double parent_node_visits = 0;
    for (const auto child_q_node : node->getChildren())
        parent_node_visits += (search_stats.stop_abstraction || child_q_node->isAbsDropped())? child_q_node->getVisits() : child_q_node->getAbsVisits();


    //Determine c from the UCT formula
    const double var = std::max(0.0,search_stats.total_squared_v / search_stats.global_num_vs - (search_stats.total_v / search_stats.global_num_vs) *  (search_stats.total_v / search_stats.global_num_vs));
    double dynamic_exp_factor = sqrt(var);
    double exploration_factor = greedy? 0 : args.exploration_parameter * dynamic_exp_factor;

    double best_value = -std::numeric_limits<double>::infinity();
    int best_action = -0x0EADBEEF;
    for (const auto child_q_node : node->getChildren())
    {
        //Get visits and Q used in uct formula
        double action_visits, Q_value;
        if (child_q_node->isAbsDropped() || search_stats.stop_abstraction) {
            action_visits = child_q_node->getVisits();
            Q_value = child_q_node->getValues() / action_visits;
        }
        else {
            action_visits = child_q_node->getAbsVisits();
            Q_value = child_q_node->getAbsValues() / action_visits;
        }

        //Exploration term in uct formula
        const double exploration_term = exploration_factor * sqrt(log(parent_node_visits) / action_visits);

        //Get final uct score
        double score = Q_value + exploration_term;
        score += TIEBREAKER_NOISE * dist(rng); //trick to efficiently break ties

        if (score > best_value){
            best_value = score;
            best_action = child_q_node->getAction();
        }
    }

    assert(best_action != -0x0EADBEEF);
    return best_action;
}

double OgaAgent::rollout(const OgaStateNode* leaf, ABS::Model* model, std::mt19937& rng) const
{
    auto reward_sum = 0.0;
    if (leaf->isTerminal())
        return reward_sum; // No rewards to collect

    for (int i = 0; i < num_rollouts; i++){
        double total_discount = 1;
        auto* rollout_state = leaf->getStateCopy(model);
        int episode_steps = 0;
        while (!rollout_state->terminal && (rollout_length == -1 || episode_steps < rollout_length)){
            // Sample action
            auto available_actions = model->getActions(rollout_state);
            std::uniform_int_distribution<int> dist(0, static_cast<int>(available_actions.size()) - 1);
            const int action = available_actions[dist(rng)];

            // Apply action and get rewards
            auto [rewards, outcome_and_probability] = model->applyAction(rollout_state, action, rng);
            reward_sum += rewards[0] * total_discount;
            total_discount *= discount;

            episode_steps++;
        }
        delete rollout_state;
    }
    return reward_sum / (double) num_rollouts;
}

void OgaAgent::backup(OgaTree* tree, OgaStateNode* leaf, double values, OgaSearchStats& search_stats) const
{
    auto* child_node = leaf;
    auto* parent_q_node = child_node->popTrajectoryParent();
    while (parent_q_node != nullptr)
    {
        auto rewards = parent_q_node->getRewards();
        values = values * discount + rewards;

        parent_q_node->addExperience(values);
        parent_q_node->addRecencyCount();
        if (!search_stats.stop_abstraction && parent_q_node->getRecencyCount() >= recency_count_limit)
            tree->addUpdateQStateNodeAbstraction(parent_q_node);

        child_node = parent_q_node->getParent();
        child_node->addVisit();

        //Dynamic exploration factor bookkeeping
        if (parent_q_node->getVisits() == 1)
            search_stats.global_num_vs++;
        if(parent_q_node->getVisits() > 1) { //only remove value if it was present before
            double old_q = (parent_q_node->getValues() - values) / ((double) parent_q_node->getVisits()-1);
            search_stats.total_v -= old_q;
            search_stats.total_squared_v -= old_q * old_q;
        }
        double q = parent_q_node->getValues() / (double) parent_q_node->getVisits();
        search_stats.total_v += q;
        search_stats.total_squared_v+= q*q;

        parent_q_node = child_node->popTrajectoryParent();
    }
}
