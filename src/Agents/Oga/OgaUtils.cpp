#include <random>
#include <ranges>

#include "../../../include/Agents/Oga/OgaUtils.h"
#include "../../../include/Agents/Oga/OgaAbstractNodes.h"

#include <cassert>
#include <fstream>

using namespace OGA;

NextDistribution::NextDistribution(double rewards) : rewards(rewards) {}

void NextDistribution::addProbability(OgaAbstractStateNode* next_abstract_state_node, const double probability)
{
    if (!distribution.contains(next_abstract_state_node))
    {
        distribution[next_abstract_state_node] = probability;
    }
    else
    {
        distribution[next_abstract_state_node] += probability;
    }
}

double NextDistribution::getRewards() const
{
    return rewards;
}

const Map<OgaAbstractStateNode, double>& NextDistribution::getDistribution() const{
    return distribution;
}

inline double NextDistribution::rewardDist(const NextDistribution* other) const{
    return std::fabs(rewards - other->getRewards());
}

inline double NextDistribution::transDist(const NextDistribution* other) const {
    double trans_dist = 0;
    Map<OgaAbstractStateNode,std::pair<bool,bool>> union_map;
    for(const auto &abs_node: other->getDistribution() | std::views::keys)
        union_map.insert({abs_node,{false,true}});
    for(const auto &abs_node: distribution | std::views::keys) {
        if(union_map.contains(abs_node))
            union_map.at(abs_node).first = true;
        else
            union_map.insert({abs_node,{true,false}});
    }
    for(auto [abs_node,occurences] : union_map) {
        double p1 = occurences.first ? distribution.at(abs_node) : 0;
        double p2 = occurences.second ? other->getDistribution().at(abs_node) : 0;
        trans_dist += std::fabs(p1 - p2);
    }

    return trans_dist;
}

double NextDistribution::dist(const NextDistribution* other) const {
    return std::max(rewardDist(other), transDist(other));
}

bool NextDistribution::approxEqual(const NextDistribution* other, double eps_a, double eps_t) const {
    return std::max(rewardDist(other) -  eps_a, 0.0) < 1e-6 && std::max(transDist(other) - eps_t, 0.0) < 1e-6;
}

bool NextDistribution::operator==(const NextDistribution& other) const
{

    if (rewards != other.getRewards())
        return false;

    const auto& other_distribution = other.getDistribution();
    for (const auto& [next_abstract_state_node, probability] : distribution){
        if (!other_distribution.contains(next_abstract_state_node) || std::fabs(probability - other_distribution.at(next_abstract_state_node)) > 1e-6)
            return false;
    }

    for (const auto& [next_abstract_state_node, probability] : other_distribution){
        if (!distribution.contains(next_abstract_state_node) || std::fabs(probability - distribution.at(next_abstract_state_node)) > 1e-6)
            return false;
    }

    return true;
}

size_t NextDistribution::hash() const
{
    size_t hash = 0;
    for (const auto& next_abstract_state_node : distribution | std::views::keys)
        hash ^= next_abstract_state_node->hash();
    return hash;
}

// NextAbstractQStates
void NextAbstractQStates::addAbstractQState(OgaAbstractQStateNode* next_abstract_q_state_node)
{
    states.insert(next_abstract_q_state_node);
}

const Set<OgaAbstractQStateNode>& NextAbstractQStates::getStates() const
{
    return states;
}

bool NextAbstractQStates::operator==(const NextAbstractQStates& other) const
{
    return states == other.getStates();
}

size_t NextAbstractQStates::hash() const{
    size_t hash = 0;
    for (const auto* next_abstract_q_state_node : states)
        hash ^= next_abstract_q_state_node->hash();
    return hash;
}
