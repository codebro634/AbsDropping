#pragma once

#ifndef MISCANALYSIS_H
#define MISCANALYSIS_H

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ValueIteration.h"
#include "../../include/Agents/Agent.h"

namespace MISC
{

    void createQTable(ABS::Model* ground_model, std::unordered_map<std::pair<FINITEH::Gamestate*,int> , double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare>* Q_map_ptr = nullptr,std::string save_path = "", bool verbose =true, int time_limit = std::numeric_limits<int>::max());

    //To find critical states
    std::vector<ABS::Gamestate*> gatherSmallQDiffStates(ABS::Model& model, unsigned int num_states,std::unordered_map<std::pair<FINITEH::Gamestate*,int>, double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare>* Q_map, Agent* agent, std::mt19937& rng);

    //For oga-cad paper to measure the drop rates
    void estimateAbsDropNumbers();

    template <class T>
struct PointedHash
    {
        size_t operator()(const T* p) const
        {
            return p->hash();
        }
    };

    template <class T>
    struct PointedCompare
    {
        bool operator()(const T* lhs, const T* rhs) const
        {
            return lhs == rhs || *lhs == *rhs;
        }
    };

    template <class T>
    using Set = std::unordered_set<T*, PointedHash<T>, PointedCompare<T>>;

}


#endif //MISCANALYSIS_H
