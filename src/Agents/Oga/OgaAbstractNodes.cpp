#include <random>
#include <ranges>

#include "../../../include/Agents/Oga/OgaGroundNodes.h"
#include "../../../include/Agents/Oga/OgaUtils.h"
#include "../../../include/Agents/Oga/OgaAbstractNodes.h"

#include <cassert>
#include <fstream>
#include <map>
#include <set>

using namespace OGA;

unsigned OgaAbstractNode::max_id = 0;

OgaAbstractNode::OgaAbstractNode(const unsigned depth) : id(max_id++), depth(depth)
{}

unsigned OgaAbstractNode::getId() const
{
    return id;
}

unsigned OgaAbstractNode::getDepth() const
{
    return depth;
}

void* OgaAbstractNode::popAndSetKey(void* key)
{
    assert(key != nullptr);
    void* temp = this->key;
    this->key = key;
    return temp;
}

unsigned OgaAbstractNode::getCount() const
{
    return count;
}

void OgaAbstractNode::increaseCount()
{
    count++;
}

void OgaAbstractNode::decreaseCount()
{
    assert(count > 0);
    count--;
}

bool OgaAbstractNode::operator==(const OgaAbstractNode& other) const
{
    return id == other.id && depth == other.depth;
}

size_t OgaAbstractNode::hash() const
{
    return std::hash<unsigned>{}(id) ^ std::hash<unsigned>{}(depth);
}

// OgaAbstractStateNode
void OgaAbstractStateNode::transfer(OgaStateNode* state_node, OgaAbstractStateNode* from, OgaAbstractStateNode* to){
    if (from != to) {
        to->add(state_node);
        if (from != nullptr) {
            from->remove(state_node);
        }
        state_node->setAbstractNode(to);
    }
}

OgaStateNode* OgaAbstractStateNode::getRepresentant() const
{
    return representant;
}

void OgaAbstractStateNode::setRepresentant(OgaStateNode* representant)
{
    this->representant = representant;
}

void OgaAbstractStateNode::add(OgaStateNode* state_node){
    ground_nodes.insert(state_node);
    if (getCount() == 0)
        setRepresentant(state_node);
    increaseCount();
}

void OgaAbstractStateNode::remove(OgaStateNode* state_node){
    assert (ground_nodes.contains(state_node));
    ground_nodes.erase(state_node);
    if (state_node == getRepresentant()) {
        OgaStateNode *representant = nullptr;
        unsigned min_id = std::numeric_limits<unsigned>::max();
        for (auto* other_node : ground_nodes) { //Min id for reproducibility
            if (other_node->getId() < min_id) {
                min_id = other_node->getId();
                representant = other_node;
            }
        }
        setRepresentant(representant);
    }
    decreaseCount();
}


// OgaAbstractQStateNode

void OgaAbstractQStateNode::setRepresentant(OgaQStateNode* representant)
{
    this->representant = representant;
}

OgaQStateNode* OgaAbstractQStateNode::getRepresentant() const
{
    return representant;
}

void OgaAbstractQStateNode::addExperience(double values)
{
    this->values += values;
    this->squared_values += values * values;
    visits++;
}

double OgaAbstractQStateNode::getValues() const
{
    return values;
}

double OgaAbstractQStateNode::getSquaredValues() const
{
    return squared_values;
}

double OgaAbstractQStateNode::getVisits() const
{
    return visits;
}

void OgaAbstractQStateNode::transfer(OgaQStateNode* q_state_node, OgaAbstractQStateNode* from,OgaAbstractQStateNode* to, AbsQSet& abs_q_set, bool exact_bookkeeping, OgaBehaviorFlags& flags){
    if (from != to) {
        abs_q_set.erase(to);
        to->add(q_state_node, exact_bookkeeping);
        abs_q_set.insert(to);
        if (from != nullptr) {
            abs_q_set.erase(from);
            from->remove(q_state_node, exact_bookkeeping);
            if (from->getCount() > 0){
                abs_q_set.insert(from);
                //update abs dropping status
                for (auto* other_node : from->getGroundNodes())
                    other_node->updateAbsDropStatus(flags);
            }
        }
        q_state_node->setAbstractNode(to);
        for (auto* other_node : to->getGroundNodes())
            other_node->updateAbsDropStatus(flags);
    }
}

void OgaAbstractQStateNode::add(OgaQStateNode* q_state_node, bool exact_bookkeeping){
    assert (!ground_nodes.contains(q_state_node));
    ground_nodes.insert(q_state_node);
    if (getCount() == 0)
        setRepresentant(q_state_node);
    if (exact_bookkeeping)
        _add_exact(q_state_node);
    else
        _add_approx(q_state_node);
}

void OgaAbstractQStateNode::remove(OgaQStateNode* q_state_node, bool exact_bookkeeping){
    assert (ground_nodes.contains(q_state_node));
    ground_nodes.erase(q_state_node);
    if (q_state_node == getRepresentant()) {
        OgaQStateNode *representant = nullptr;
        unsigned min_id = std::numeric_limits<unsigned>::max();
        for (auto* other_node : ground_nodes) { //Min id for reproducibility
            if (other_node->getId() < min_id) {
                min_id = other_node->getId();
                representant = other_node;
            }
        }
        setRepresentant(representant);
    }
    if (exact_bookkeeping)
        _remove_exact(q_state_node);
    else
        _remove_approx(q_state_node);
}

void OgaAbstractQStateNode::_add_approx(const OgaQStateNode* q_state_node){
    const auto* other_abstract_node = q_state_node->getAbstractNode();

    if (other_abstract_node == nullptr){
        increaseCount();
        return;
    }
    const auto other_count = other_abstract_node->getCount();
    const auto other_values = other_abstract_node->getValues();

    this->values += other_values / other_count;
    this->squared_values += other_abstract_node->squared_values / other_count;
    visits += other_abstract_node->getVisits() / other_count;

    increaseCount();
}

void OgaAbstractQStateNode::_remove_approx(const OgaQStateNode* q_state_node)
{
    //Paper:  "Maintaining Q-Values and [Visit-]Counts"
    values -= values / getCount();
    squared_values -= squared_values / getCount();
    visits -= visits / getCount();
    decreaseCount();
}

void OgaAbstractQStateNode::_add_exact(const OgaQStateNode* q_state_node)
{
    visits += q_state_node->getVisits();
    values += q_state_node->getValues();
    squared_values += q_state_node->getSquaredValues();
    increaseCount();
}

void OgaAbstractQStateNode::_remove_exact(const OgaQStateNode* q_state_node)
{
    visits -= q_state_node->getVisits();
    values -= q_state_node->getValues();
    squared_values -= q_state_node->getSquaredValues();
    decreaseCount();
}