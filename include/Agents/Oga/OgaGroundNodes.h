#ifndef OGAGROUNDNODES_H
#define OGAGROUNDNODES_H

#include "OgaUtils.h"

namespace OGA {

    //Forward declarations
    class OgaQStateNode;
    class OgaAbstractStateNode;

    class OgaStateNode
    {
    private:
        // Information that identifies the node
        static unsigned max_id;
        unsigned id;
        ABS::Gamestate* state;
        unsigned depth;

        // Bookkeeping for MCTS
        unsigned visits = 0;
        std::vector<int> tried_actions{};
        std::vector<int> untried_actions{};
        std::vector<OgaQStateNode*> children{}; // Cached for better performance

        // OGA bookkeeping
        OgaAbstractStateNode* abstract_node = nullptr;
        Set<OgaQStateNode> parents{}; // Needed for updating state abstractions

        // Temporarily store the trajectory path and rewards to be able to do backpropagation
        OgaQStateNode* trajectory_parent = nullptr;


    public:

        OgaStateNode(
            ABS::Gamestate* state,
            unsigned depth
        );

        ~OgaStateNode();


        [[nodiscard]] const ABS::Gamestate* getState() const;
        [[nodiscard]] ABS::Gamestate* getStateCopy(ABS::Model* model) const;
        [[nodiscard]] unsigned getDepth() const;
        [[nodiscard]] bool isTerminal() const;
        [[nodiscard]] unsigned getId() const { return id; }

        // MCTS bookkeeping functions
        void addVisit();
        [[nodiscard]] unsigned getVisits() const;
        void initUntriedActions(const std::vector<int>& actions, std::mt19937& rng);
        [[nodiscard]] int popUntriedAction();
        [[nodiscard]] bool isFullyExpanded() const;
        [[nodiscard]] bool isPartiallyExpanded() const;
        [[nodiscard]] unsigned numTriedActions() const;
        [[nodiscard]] unsigned numUntriedActions() const;
        void addChild(OgaQStateNode* child);
        [[nodiscard]] const std::vector<OgaQStateNode*>& getChildren() const;

        // OGA bookkeeping functions
        void setAbstractNode(OgaAbstractStateNode* abstract_node);
        [[nodiscard]] OgaAbstractStateNode* getAbstractNode() const;
        [[nodiscard]] const Set<OgaQStateNode>& getParents() const;

        // Trajectory functions
        void setTrajectoryParent(OgaQStateNode* parent);
        [[nodiscard]] OgaQStateNode* popTrajectoryParent();

        // Functions for hashing and comparison needed for unordered_map and unordered_set
        bool operator==(const OgaStateNode& other) const;
        [[nodiscard]] size_t hash() const;
    };

    class OgaQStateNode
    {
    private:
        static unsigned max_id;
        unsigned id;

        // Information that identifies the node
        ABS::Gamestate* state;
        unsigned depth;
        int action;

        // Bookkeeping for MCTS
        unsigned visits = 0;
        double values = 0;
        double squared_values = 0;
        OgaStateNode* parent = nullptr; // Cached for better performance in backpropagation and state abstraction updates

        // Bookkeeping for OGA
        Map<ABS::Gamestate, std::pair<double, OgaStateNode*>> children{}; // Cached for better performance
        OgaAbstractQStateNode* abstract_node = nullptr;
        double rewards{}; // For q state abstractions, a q state needs a deterministic reward (cost)
        unsigned recency_count = 0;

        bool use_ground_stats = false;

    public:

        OgaQStateNode(
            ABS::Gamestate* state,
            unsigned depth,
            int action
        );

        ~OgaQStateNode();

        [[nodiscard]] ABS::Gamestate *getState() const;
        [[nodiscard]] unsigned getDepth() const;
        [[nodiscard]] int getAction() const;
        [[nodiscard]] unsigned getId() const { return id; }

        // MCTS bookkeeping functions
        [[nodiscard]] unsigned getVisits() const;
        [[nodiscard]] double getValues() const;
        [[nodiscard]] double getSquaredValues() const;
        void addExperience(double values);
        [[nodiscard]] double getAbsVisits() const;
        [[nodiscard]] double getAbsValues() const;

        void updateAbsDropStatus(OgaBehaviorFlags& flags);
        bool isAbsDropped() const;

        // OGA bookkeeping functions
        bool addChild(ABS::Gamestate* state, double probability, OgaStateNode* child);
        [[nodiscard]] const Map<ABS::Gamestate, std::pair<double, OgaStateNode*>>* getChildren() const;

        // OGA bookkeeping functions
        void setAbstractNode(OgaAbstractQStateNode* abstract_node);
        [[nodiscard]] OgaAbstractQStateNode* getAbstractNode() const;
        void setRewards(double rewards);
        [[nodiscard]] double getRewards() const;
        void addRecencyCount();
        void resetRecencyCount();
        [[nodiscard]] unsigned getRecencyCount() const;

        // Trajectory functions
        void setParent(OgaStateNode* parent);
        [[nodiscard]] OgaStateNode* getParent() const;

        // Functions for hashing and comparison needed for unordered_map and unordered_set
        bool operator==(const OgaQStateNode& other) const;
        [[nodiscard]] size_t hash() const;
    };


}

#endif //OGAGROUNDNODES_H
