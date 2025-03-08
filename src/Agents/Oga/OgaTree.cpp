#include <random>
#include <ranges>

#include "../../../include/Agents/Oga/OgaGroundNodes.h"
#include "../../../include/Agents/Oga/OgaTree.h"
#include "../../../include/Agents/Oga/OgaUtils.h"
#include "../../../include/Agents/Oga/OgaAbstractNodes.h"

#include <cassert>
#include <fstream>
#include <set>
#include <algorithm>
#include <deque>

using namespace OGA;


OgaTree::OgaTree(
    ABS::Gamestate* root_state,
    ABS::Model* model,
    const OgaBehaviorFlags& behavior_flags,
    std::mt19937& rng
) : behavior_flags(behavior_flags), model(model)
{
    auto [root, found] = findOrCreateState(root_state, 0, rng);
    assert(!found);
    this->root = root;
}

OgaTree::~OgaTree()
{
    for (const auto* state_node : d_states)
        delete state_node;
    for (const auto* q_state_node : q_states)
        delete q_state_node;

    for (const auto& abstract_q_state_node_map : abstract_q_state_node_map){
        for (const auto* next_distribution : abstract_q_state_node_map | std::views::keys)
            delete next_distribution;
    }
    for (const auto& next_distribution_map : next_distribution_map){
        for (const auto* next_distribution : next_distribution_map | std::views::values)
            delete next_distribution;
    }

    for (const auto& abstract_state_node_map : abstract_state_node_map){
        for (const auto* next_abstract_q_states : abstract_state_node_map | std::views::keys)
            delete next_abstract_q_states;
    }

    for (const auto &val: abstract_state_nodes | std::views::values) {
        for (const auto* abstract_node : val){
            assert (abstract_node->getCount() > 0);
            delete abstract_node;
        }
    }

    for (auto corpse : abstract_q_corpses)
        delete corpse;

    for (auto corpse : abstract_state_corpses)
        delete corpse;

    for (const auto &val: abstract_q_state_nodes | std::views::values){
        for (const auto* abstract_node : val){
            assert (abstract_node->getCount() > 0);
            delete abstract_node;
        }
    }
}

double OgaTree::getCompressionRate() const{

    int total_abs_nodes = 0;
    for (const auto& val: abstract_state_nodes | std::views::values)
        total_abs_nodes += val.size();
    double state_compression = static_cast<double>(d_states.size()) / static_cast<double>(total_abs_nodes);

    int total_abs_q_nodes = 0;
    for (const auto& val: abstract_q_state_nodes | std::views::values)
        total_abs_q_nodes += val.size();
    double q_state_compression = static_cast<double>(q_states.size()) / static_cast<double>(total_abs_q_nodes);

    return std::max(state_compression, q_state_compression);
}

OgaStateNode* OgaTree::getRoot() const
{
    return root;
}

std::pair<OgaStateNode*, bool> OgaTree::findOrCreateState(ABS::Gamestate* state, const unsigned depth, std::mt19937& rng)
{
    auto* state_node = new OgaStateNode(model->copyState(state), depth);

    if (const auto it = d_states.find(state_node); it != d_states.end()){
        // State already exists: Delete the new state node and return the existing one
        delete state_node;
        return {*it, true};
    }

    // Initialize state abstraction
    OgaAbstractStateNode* abstract_state_node;
    if (state->terminal && behavior_flags.group_terminal_states)
    {
        // Use one terminal abstract state node per depth
        // Resize terminal abstract state nodes if necessary and create terminal abstract state node
        if (terminal_abstract_state_nodes.size() <= depth)
            terminal_abstract_state_nodes.resize(depth + 1);
        if (terminal_abstract_state_nodes[depth] == nullptr){
            terminal_abstract_state_nodes[depth] = new OgaAbstractStateNode(depth);
            abstract_state_nodes[static_cast<int>(depth)].insert(terminal_abstract_state_nodes[depth]);
        }

        abstract_state_node = terminal_abstract_state_nodes[depth];
    }
    else if (behavior_flags.group_partially_expanded_states)
    {
        if (unexplored_abstract_state_nodes.size() <= depth)
            unexplored_abstract_state_nodes.resize(depth + 1);
        if (unexplored_abstract_state_nodes[depth] == nullptr){
            unexplored_abstract_state_nodes[depth] = new OgaAbstractStateNode(depth);
            abstract_state_nodes[static_cast<int>(depth)].insert(unexplored_abstract_state_nodes[depth]);
        }

        abstract_state_node = unexplored_abstract_state_nodes[depth];
    }
    else
    {
        abstract_state_node = new OgaAbstractStateNode(depth);
        abstract_state_nodes[static_cast<int>(depth)].insert(abstract_state_node);
    }

    // Init node and insert into tree
    OgaAbstractStateNode::transfer(state_node, nullptr, abstract_state_node);

    state_node->initUntriedActions(state->terminal ? std::vector<int>{} : model->getActions(state), rng);
    d_states.insert(state_node);

    return {state_node, false};
}

std::pair<OgaQStateNode*, bool> OgaTree::findOrCreateQState(ABS::Gamestate* state, const unsigned depth,const int action, std::mt19937& rng){
    auto* q_state_node = new OgaQStateNode(model->copyState(state), depth, action);

    if (const auto it = q_states.find(q_state_node); it != q_states.end())
    {
        // Q-state already exists: Delete the new q-state node and return the existing one
        delete q_state_node;
        return {*it, true};
    }

    // Initialize q-state abstraction
    auto* abstract_q_state_node = new OgaAbstractQStateNode(q_state_node->getDepth());
    OgaAbstractQStateNode::transfer(q_state_node, nullptr, abstract_q_state_node, abstract_q_state_nodes[static_cast<int>(depth)], behavior_flags.exact_bookkeeping, behavior_flags);

    // Init q state node and insert into tree
    auto [parent, found] = findOrCreateState(state, depth, rng);
    assert(found);
    parent->addChild(q_state_node);
    q_states.insert(q_state_node);

    return {q_state_node, false};
}

void OgaTree::insert(OgaStateNode* state_node)
{
    d_states.insert(state_node);
}

void OgaTree::insert(OgaQStateNode* state_node)
{
    q_states.insert(state_node);
}

void OgaTree::_stageForUpdate(OgaStateNode* state_node)
{
    auto depth = state_node->getDepth();
    if (to_update_states.size() <= depth){
        to_update_states.resize(depth + 1);
        abstract_state_node_map.resize(depth + 1);
    }
    to_update_states[depth].insert(state_node);
}
void OgaTree::_stageForUpdate(OgaQStateNode* q_state_node)
{
    auto depth = q_state_node->getDepth();
    if (to_update_q_states.size() <= depth)
    {
        to_update_q_states.resize(depth + 1);
        abstract_q_state_node_map.resize(depth + 1);
        next_distribution_map.resize(depth + 1);
    }
    to_update_q_states[depth].insert(q_state_node);
}

void OgaTree::updateQAbstractions(unsigned K, int depth) {

    //Sort by id for reproducibility (for eps > 0 case this might make a difference)
    std::vector<OgaQStateNode*> sorted_to_update_nodes = std::vector(to_update_q_states[depth].begin(), to_update_q_states[depth].end());
    std::sort(sorted_to_update_nodes.begin(), sorted_to_update_nodes.end(),
              [](const OgaQStateNode* lhs, const OgaQStateNode* rhs) {
                  return lhs->getId() < rhs->getId();  // Sort by ID
              });

    //Update the abstraction
    for (auto* q_state_node : sorted_to_update_nodes){

        // Calculate Successor distribution and update Certainty
        auto* next_distribution = new NextDistribution(q_state_node->getRewards());
        for (const auto& probability_successor : *q_state_node->getChildren() | std::views::values){
            auto& [probability, successor] = probability_successor;
            assert (successor->getAbstractNode()->getCount() > 0);
            next_distribution->addProbability(successor->getAbstractNode(), probability);
        }

        // Reset recency count
        q_state_node->resetRecencyCount();
        OgaAbstractQStateNode* old_abstract_q_state_node = q_state_node->getAbstractNode();
        assert (old_abstract_q_state_node != nullptr && old_abstract_q_state_node->getCount() > 0);
        assert (old_abstract_q_state_node->getRepresentant() != nullptr);
        OgaAbstractQStateNode* new_abstract_q_state_node = old_abstract_q_state_node;

        // Determine new abstract node
        if (behavior_flags.eps_a != 0 || behavior_flags.eps_t != 0) {

            [[maybe_unused]] bool contains_rep = next_distribution_map[depth].contains(old_abstract_q_state_node->getRepresentant());
            assert (contains_rep || old_abstract_q_state_node->getRepresentant() == q_state_node); //if !contain_rep then fresh q-node
            if (old_abstract_q_state_node->getRepresentant() == q_state_node && contains_rep) { //Try merging with bigger abs nodes.

               //Find biggest abs node that current abs node can merge with
                OgaAbstractQStateNode* merge = nullptr;
                for (auto it = abstract_q_state_nodes[depth].rbegin() ; it != abstract_q_state_nodes[depth].rend(); ++it){ //sorted by abs size
                    auto abs_other_node = *it;
                    if (!next_distribution_map[depth].contains(abs_other_node->getRepresentant()))
                        continue;
                    assert (abs_other_node->getRepresentant() != nullptr && abs_other_node->getCount() > 0 && abs_other_node->getRepresentant()->getAbstractNode() == abs_other_node);
                    bool approx = next_distribution->approxEqual(next_distribution_map[depth].at(abs_other_node->getRepresentant()), behavior_flags.eps_a, behavior_flags.eps_t);
                    if (approx) {
                        merge = abs_other_node;
                        break;
                    }
                }

                if (merge != nullptr && merge != old_abstract_q_state_node) {
                    new_abstract_q_state_node = merge;
                }else if (merge == nullptr) { //Create new abstract node
                    if (old_abstract_q_state_node->getCount() > 1)
                        new_abstract_q_state_node = new OgaAbstractQStateNode(depth);
                    else// Recycle old state
                        new_abstract_q_state_node = old_abstract_q_state_node;
                }

            }
            else if ( !contains_rep || !next_distribution->approxEqual(next_distribution_map[depth].at(old_abstract_q_state_node->getRepresentant()), behavior_flags.eps_a, behavior_flags.eps_t)) {

                //find new matching abstract node with minimal distance
                double min_dist = std::numeric_limits<double>::infinity();
                unsigned min_id = -1;
                OgaAbstractQStateNode* min_dist_node = nullptr;
                for (auto abs_other_node : abstract_q_state_nodes[depth]) {
                    if ((abs_other_node->getRepresentant() == q_state_node && !contains_rep) || !next_distribution_map[depth].contains(abs_other_node->getRepresentant()))
                        continue;
                    assert (abs_other_node->getRepresentant() != q_state_node && abs_other_node->getRepresentant() != nullptr && abs_other_node->getCount() > 0 && abs_other_node->getRepresentant()->getAbstractNode() == abs_other_node);
                    double dist = next_distribution->dist(next_distribution_map[depth].at(abs_other_node->getRepresentant()));
                    if ( dist < min_dist || (dist == min_dist&& abs_other_node->getId() < min_id)) {
                        min_dist = dist;
                        min_id = abs_other_node->getId();
                        min_dist_node = abs_other_node;
                    }
                }
                //No match found?
                bool transferrable = min_dist_node != nullptr && next_distribution->approxEqual(next_distribution_map[depth].at(min_dist_node->getRepresentant()), behavior_flags.eps_a, behavior_flags.eps_t);
                bool create_new_abs_node = min_dist != std::numeric_limits<double>::infinity() && !transferrable;

                // Create new abstract node
                if (create_new_abs_node) {
                    if (old_abstract_q_state_node->getCount() > 1)
                        new_abstract_q_state_node = new OgaAbstractQStateNode(depth);
                    else // Recycle old state
                       new_abstract_q_state_node = old_abstract_q_state_node;
                } else if (transferrable)
                    new_abstract_q_state_node = min_dist_node;

            }else {
               assert (next_distribution_map.at(depth).contains(q_state_node));
            }

            auto old_distr = next_distribution_map.at(depth)[q_state_node];
            delete old_distr;
            next_distribution_map.at(depth)[q_state_node] = next_distribution;

        } else{ //Case eps_a = 0 and eps_t = 0 can be handled more efficiently

            bool exists = abstract_q_state_node_map[depth].contains(next_distribution);
            bool move_to_existing = exists && abstract_q_state_node_map[depth][next_distribution]->getCount() > 0;
            if (!move_to_existing){

                if (old_abstract_q_state_node->getCount() > 1){
                    new_abstract_q_state_node = new OgaAbstractQStateNode(depth);
                    [[maybe_unused]] auto* old_key = new_abstract_q_state_node->popAndSetKey(next_distribution);
                    assert(old_key == nullptr);
                }
                else{
                    // Recycle old state
                    new_abstract_q_state_node = old_abstract_q_state_node;
                    auto* old_next_distribution = static_cast<NextDistribution*>(old_abstract_q_state_node->popAndSetKey(next_distribution));
                    if (old_next_distribution != nullptr && old_next_distribution != next_distribution){
                        abstract_q_state_node_map[depth].erase(old_next_distribution);
                        delete old_next_distribution;
                    }
                }
                if (exists) {
                    auto old_next = abstract_q_state_node_map[depth].find(next_distribution)->first;
                    abstract_q_state_node_map[depth].erase(old_next);
                    delete old_next;
                }
                abstract_q_state_node_map[depth][next_distribution] = new_abstract_q_state_node;
            }
            else{
                new_abstract_q_state_node = abstract_q_state_node_map[depth][next_distribution];
                delete next_distribution;
            }
        }

        bool abstract_node_changed = (old_abstract_q_state_node != new_abstract_q_state_node) || (q_state_node->getVisits() == K);
        if (abstract_node_changed)
            _stageForUpdate(q_state_node->getParent());

        if (abstract_node_changed) {
            assert (abstract_q_state_nodes[depth].contains(old_abstract_q_state_node));
            assert (old_abstract_q_state_node->getCount() > 0);
            OgaAbstractQStateNode::transfer(q_state_node, old_abstract_q_state_node, new_abstract_q_state_node, abstract_q_state_nodes[depth], behavior_flags.exact_bookkeeping, behavior_flags);
            if (old_abstract_q_state_node->getCount() == 0)
                abstract_q_corpses.insert(old_abstract_q_state_node);
        }else
            q_state_node->updateAbsDropStatus(behavior_flags);
    }

}

void OgaTree::updateStateAbstractions(int depth) {

    //Sort by id for reproducibility (for eps > 0 case this might make a difference)
    std::vector<OgaStateNode*> sorted_to_update_nodes = std::vector(to_update_states[depth].begin(), to_update_states[depth].end());
    std::sort(sorted_to_update_nodes.begin(), sorted_to_update_nodes.end(),
              [](const OgaStateNode* lhs, const OgaStateNode* rhs) {
                  return lhs->getId() < rhs->getId();  // Sort by ID
              });

    for (auto* state_node : sorted_to_update_nodes)
    {
        if (!state_node->isFullyExpanded() && state_node->numTriedActions() <= behavior_flags.partial_expansion_group_threshold)
            continue;

        // Algorithm 3: Part 1
        OgaAbstractStateNode* old_abstract_state_node = state_node->getAbstractNode();
        assert (old_abstract_state_node == nullptr || state_node->getAbstractNode()->getCount() > 0);
        assert (old_abstract_state_node == nullptr || abstract_state_nodes[depth].contains(old_abstract_state_node));

        OgaAbstractStateNode* new_abstract_state_node = old_abstract_state_node;

        // Algorithm 4
        auto* next_abstract_q_states = new NextAbstractQStates();
        for (const auto* q_state : state_node->getChildren()){
            assert (q_state->getAbstractNode()->getCount() > 0);
            next_abstract_q_states->addAbstractQState(q_state->getAbstractNode());
        }

        bool exists = abstract_state_node_map[depth].contains(next_abstract_q_states);
        bool move_to_existing = exists && abstract_state_node_map[depth][next_abstract_q_states]->getCount() > 0;
        if (!move_to_existing){

            if (old_abstract_state_node->getCount() > 1 ||
                (behavior_flags.group_partially_expanded_states && old_abstract_state_node == unexplored_abstract_state_nodes[depth])){
                new_abstract_state_node = new OgaAbstractStateNode(depth);
                abstract_state_nodes[depth].insert(new_abstract_state_node);
                [[maybe_unused]] auto* old_key = new_abstract_state_node->popAndSetKey(next_abstract_q_states);
                assert(old_key == nullptr);
            }
            else{ //recycle old state
                new_abstract_state_node = old_abstract_state_node;
                auto* old_next_abstract_q_states = static_cast<NextAbstractQStates*>(old_abstract_state_node->popAndSetKey(next_abstract_q_states));
                if (old_next_abstract_q_states != nullptr && old_next_abstract_q_states != next_abstract_q_states){
                    abstract_state_node_map[depth].erase(old_next_abstract_q_states);
                    delete old_next_abstract_q_states;
                }
            }

            if (exists) {
                auto old_next = abstract_state_node_map[depth].find(next_abstract_q_states)->first;
                abstract_state_node_map[depth].erase(old_next);
                delete old_next;
            }
            abstract_state_node_map[depth][next_abstract_q_states] = new_abstract_state_node;
        }
        else{
            new_abstract_state_node = abstract_state_node_map[depth][next_abstract_q_states];
            delete next_abstract_q_states;
        }

        // Algorithm 3: Part 2
        bool abstract_node_changed = old_abstract_state_node != new_abstract_state_node;
        if (abstract_node_changed){
            for (auto* q_state_node_parent : state_node->getParents())
                _stageForUpdate(q_state_node_parent);
        }
        if (abstract_node_changed) {
            OgaAbstractStateNode::transfer(state_node, old_abstract_state_node, new_abstract_state_node);
            if (old_abstract_state_node->getCount() == 0) {
                abstract_state_nodes[depth].erase(old_abstract_state_node);
                abstract_state_corpses.insert(old_abstract_state_node);
                if (behavior_flags.group_partially_expanded_states && unexplored_abstract_state_nodes[depth] == old_abstract_state_node)
                    unexplored_abstract_state_nodes[depth] = nullptr;
            }
        }

    }

}

void OgaTree::performUpdateAbstractions(unsigned K){

    assert (to_update_q_states.size() >= to_update_states.size());
    for (int depth = static_cast<int>(to_update_q_states.size()) - 1; depth >= 0; depth--){
        updateQAbstractions(K, depth);
        if (depth < static_cast<int>(to_update_states.size()))
            updateStateAbstractions(depth);
    }

    // Clear all update sets
    for (auto& update_set : to_update_q_states)
        update_set.clear();
    for (auto& update_set : to_update_states)
        update_set.clear();
}

void OgaTree::addUpdateQStateNodeAbstraction(OgaQStateNode* q_state_node){
    _stageForUpdate(q_state_node);
}

std::pair<std::map<size_t,int>,std::map<size_t,int>> OgaTree::getNumAbsDrops() {
    std::set<OgaStateNode*> visited;
    std::deque<OgaStateNode*> q = {root};

    std::map<size_t,int> layer_to_abs_drops;
    std::map<size_t,int> layer_to_abs_count;

    while (!q.empty()){
        auto* state_node = q.front();
        q.pop_front();
        if (visited.contains(state_node))
            continue;
        visited.insert(state_node);
        for (const auto* q_state_node : state_node->getChildren()){
            if (q_state_node->getAbstractNode()->getCount() > 1) {
                layer_to_abs_count[q_state_node->getDepth()]++;
                if (q_state_node->isAbsDropped())
                    layer_to_abs_drops[q_state_node->getDepth()]++;
            }

            for (const auto child : (*q_state_node->getChildren()))
                q.push_back(child.second.second);
        }
    }
    return {layer_to_abs_drops, layer_to_abs_count};
}

void OgaTree::printAbsTree(size_t layers) const {
    // std::cout << "-------- OGA abs tree ------------ " << std::endl;
    // for (size_t depth = 0; depth < abstract_state_nodes.size(); depth++) {
    //     auto astates = abstract_state_nodes.at(depth);
    //     std::cout << "Depth " << depth << " states:" << astates.size() << std::endl;
    //     if (abstract_q_state_nodes.contains(depth)) {
    //         auto qstates = abstract_q_state_nodes.at(depth);
    //         std::cout << "Depth " << depth << " Q-states:" << qstates.size() << std::endl;
    //     }
    // }

    std::cout << "-------- OGA abs tree ------------ " << std::endl;

    for (size_t depth = 0; depth < std::min(layers,abstract_state_nodes.size()); depth++) {
        std::cout << "Depth " << depth << " states:" << std::endl;
        auto astates = abstract_state_nodes.at(depth);
        for (auto* state_node : astates) {
            std::cout << " [ ";
            for (const auto q_state_node : state_node->getGroundNodes()) {
                std::cout << q_state_node << ", ";
            }
            std::cout << " ]    ";
        }
        std::cout << std::endl;

        std::cout << "Depth " << depth << " Q-states:" << std::endl;
        if (abstract_q_state_nodes.find(depth) == abstract_q_state_nodes.end())
            continue;
        auto aqstates = abstract_q_state_nodes.at(depth);
        for (auto* q_state_node : aqstates) {
            std::cout << " [ ";
            for (const auto ground_q_state_node : q_state_node->getGroundNodes()) {
                std::cout << ground_q_state_node << ":" << ground_q_state_node->getAction() << ", ";
            }
            std::cout << " ]    ";
        }
        std::cout << std::endl;

    }

}
