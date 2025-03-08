
#ifndef EARTHOBSERVATION_H
#define EARTHOBSERVATION_H

#pragma once

#include "../Gamestate.h"
#include <string>
#include <vector>
#include <random>

namespace EO {

    struct Gamestate : public ABS::Gamestate {
        std::vector<int> visibility;
        std::vector<bool> is_target;
        int focal_point;
        bool operator==(const ABS::Gamestate& other) const override;
        [[nodiscard]] size_t hash() const override;
    };

    class Model : public ABS::Model {

    private:
        double vis_change_probs[3][3]{};
        double failure_probs[3]{};
        std::vector<std::vector<int>> connection_per_direction;

        int init_focal_point{};
        std::vector<int> init_visibility;
        std::vector<bool> init_targets;

    protected:

        std::pair<std::vector<double>,double> applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) override;
        std::vector<int> getActions_(ABS::Gamestate* uncasted_state) override;

    public:
        explicit Model(const std::string& filePath);
        ~Model() override = default;
        void printState(ABS::Gamestate* uncasted_state) override;
        ABS::Gamestate* getInitialState(std::mt19937& rng) override;
        bool hasTransitionProbs() override {return true;}
        int getNumPlayers() override;
        ABS::Gamestate* copyState(ABS::Gamestate* uncasted_state) override;
    };

} // namespace ABS


#endif //EARTHOBSERVATION_H
