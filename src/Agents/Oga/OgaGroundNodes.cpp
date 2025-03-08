#include <random>
#include <ranges>

#include "../../../include/Agents/Oga/OgaGroundNodes.h"
#include "../../../include/Agents/Oga/OgaUtils.h"
#include "../../../include/Agents/Oga/OgaAbstractNodes.h"

#include <cassert>
#include <fstream>
#include <algorithm>

#include "../../../include/Utils/Distributions.h"

using namespace OGA;

unsigned OgaStateNode::max_id = 0;
OgaStateNode::OgaStateNode(ABS::Gamestate* state, const unsigned depth) : id(max_id++), state(state), depth(depth)
{}

OgaStateNode::~OgaStateNode()
{
    delete state;
}

const ABS::Gamestate* OgaStateNode::getState() const
{
    return state;
}

ABS::Gamestate* OgaStateNode::getStateCopy(ABS::Model* model) const
{
    return model->copyState(state);
}

unsigned OgaStateNode::getDepth() const
{
    return depth;
}

bool OgaStateNode::isTerminal() const
{
    return state->terminal;
}

void OgaStateNode::addVisit()
{
    visits++;
}

unsigned OgaStateNode::getVisits() const
{
    return visits;
}

void OgaStateNode::initUntriedActions(const std::vector<int>& actions, std::mt19937& rng)
{
    untried_actions = actions;
    std::ranges::shuffle(untried_actions.begin(), untried_actions.end(), rng);
}

int OgaStateNode::popUntriedAction(){
    assert(!untried_actions.empty());
    int a = untried_actions.back();
    untried_actions.pop_back();
    tried_actions.push_back(a);
    return a;
}

bool OgaStateNode::isFullyExpanded() const
{
    return untried_actions.empty();
}

bool OgaStateNode::isPartiallyExpanded() const
{
    return !tried_actions.empty();
}

unsigned OgaStateNode::numTriedActions() const
{
    return tried_actions.size();
}

unsigned OgaStateNode::numUntriedActions() const
{
    return untried_actions.size();
}

void OgaStateNode::addChild(OgaQStateNode* child)
{
    children.push_back(child);
}

const std::vector<OgaQStateNode*>& OgaStateNode::getChildren() const
{
    return children;
}

void OgaStateNode::setAbstractNode(OgaAbstractStateNode* abstract_node)
{
    this->abstract_node = abstract_node;
}

OgaAbstractStateNode* OgaStateNode::getAbstractNode() const
{
    return abstract_node;
}

const Set<OgaQStateNode>& OgaStateNode::getParents() const
{
    return parents;
}

void OgaStateNode::setTrajectoryParent(OgaQStateNode* parent)
{
    assert(trajectory_parent == nullptr);
    trajectory_parent = parent;
    parents.insert(parent);
}

OgaQStateNode* OgaStateNode::popTrajectoryParent()
{
    auto* parent = trajectory_parent;
    trajectory_parent = nullptr;
    return parent;
}


bool OgaStateNode::operator==(const OgaStateNode& other) const
{
    return *state == *other.state && depth == other.depth;
}

size_t OgaStateNode::hash() const
{
    return state->hash() ^ std::hash<unsigned>{}(depth);
}

// OgaQStateNode

unsigned OgaQStateNode::max_id = 0;
OgaQStateNode::OgaQStateNode(ABS::Gamestate* state, const unsigned depth, const int action) : id(max_id++), state(state),
    depth(depth), action(action)
{
}

OgaQStateNode::~OgaQStateNode()
{
    delete state;
    for (const auto& state : children | std::views::keys)
        delete state;
}

ABS::Gamestate *OgaQStateNode::getState() const
{
    return state;
}

unsigned OgaQStateNode::getDepth() const
{
    return depth;
}

int OgaQStateNode::getAction() const
{
    return action;
}

unsigned OgaQStateNode::getVisits() const
{
    return visits;
}

double OgaQStateNode::getValues() const
{
    return values;
}

double OgaQStateNode::getSquaredValues() const
{
    return squared_values;
}

void OgaQStateNode::updateAbsDropStatus(OgaBehaviorFlags& flags)
{

    if (getVisits() >= flags.drop_at_visits)
        use_ground_stats = true;
    else {

        if (flags.drop_confidence < 0 || flags.drop_confidence > 1)
            return;

        auto conf_interval_ground = distr::confidence_interval(values, squared_values, visits, flags.drop_confidence);
        double abs_q = getAbsValues() / getAbsVisits();
        double dist_abs = std::min(std::fabs(abs_q - conf_interval_ground.first), std::fabs(conf_interval_ground.second - abs_q));
        double dist_ground = (conf_interval_ground.second - conf_interval_ground.first) / 2.0;
        // if (use_ground_stats != (dist_ground < dist_abs))
        //     std::cout << getAction() <<" " << getDepth() << ":  " << use_ground_stats << " -> " << (dist_ground < dist_abs) << " | " << getVisits() << std::endl;
        use_ground_stats = dist_ground < dist_abs;
    }
}

bool OgaQStateNode::isAbsDropped() const{
    return use_ground_stats;
}

double OgaQStateNode::getAbsVisits() const
{
    return abstract_node->getVisits();
}

double OgaQStateNode::getAbsValues() const
{
    return abstract_node->getValues();
}


void OgaQStateNode::addExperience(double values)
{
    getAbstractNode()->addExperience(values);
    this->values += values;
    this->squared_values += values * values;
    visits++;
}

bool OgaQStateNode::addChild(ABS::Gamestate* state, const double probability, OgaStateNode* child)
{
    if (!children.contains(state)) {
        children[state] = {probability, child};
        return true;
    }else
        return false;
}

const Map<ABS::Gamestate, std::pair<double, OgaStateNode*>>* OgaQStateNode::getChildren() const
{
    return &children;
}

void OgaQStateNode::setAbstractNode(OgaAbstractQStateNode* abstract_node)
{
    this->abstract_node = abstract_node;
}

OgaAbstractQStateNode* OgaQStateNode::getAbstractNode() const
{
    return abstract_node;
}

void OgaQStateNode::setRewards(double rewards)
{
    this->rewards = rewards;
}

double OgaQStateNode::getRewards() const
{
    return rewards;
}

void OgaQStateNode::addRecencyCount()
{
    recency_count++;
}

void OgaQStateNode::resetRecencyCount()
{
    recency_count = 0;
}

unsigned OgaQStateNode::getRecencyCount() const
{
    return recency_count;
}

void OgaQStateNode::setParent(OgaStateNode* parent)
{
    assert(this->parent == nullptr);
    this->parent = parent;
}

OgaStateNode* OgaQStateNode::getParent() const
{
    return parent;
}


bool OgaQStateNode::operator==(const OgaQStateNode& other) const
{
    return *state == *other.state && depth == other.depth && action == other.action;
}

size_t OgaQStateNode::hash() const
{
    return state->hash() ^ std::hash<unsigned>{}(depth) ^ std::hash<int>{}(action);
}