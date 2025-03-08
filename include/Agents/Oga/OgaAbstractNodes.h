#ifndef OGAABSTRACTNODES_H
#define OGAABSTRACTNODES_H
#include <set>

#include "OgaGroundNodes.h"

namespace OGA {

 class OgaAbstractNode
    {
    private:
        static unsigned max_id;
        unsigned id;
        unsigned depth;
        void* key = nullptr;
        unsigned count = 0;

    protected:
        void increaseCount();
        void decreaseCount();

    public:
        explicit OgaAbstractNode(unsigned depth);

        [[nodiscard]] unsigned getId() const;
        [[nodiscard]] unsigned getDepth() const;
        void* popAndSetKey(void* key);
        [[nodiscard]] void* getKey() const { return key; }
        [[nodiscard]] unsigned getCount() const;

        [[nodiscard]] bool operator==(const OgaAbstractNode& other) const;
        [[nodiscard]] size_t hash() const;
    };

    class OgaAbstractStateNode : public OgaAbstractNode
    {
    public:
        using OgaAbstractNode::OgaAbstractNode;

        static void transfer(OgaStateNode* state_node, OgaAbstractStateNode* from, OgaAbstractStateNode* to);
        void add(OgaStateNode* state_node);
        void remove(OgaStateNode* state_node);
        [[nodiscard]] OgaStateNode* getRepresentant() const;
        void setRepresentant(OgaStateNode* representant);
        std::set<OgaStateNode*>& getGroundNodes() { return ground_nodes; }

    private:
        OgaStateNode* representant = nullptr; //only needed when eps_a > 0 or eps_t > 0
        std::set<OgaStateNode*> ground_nodes;
    };


    // Forward declaration
    class OgaAbstractQStateNode;
    struct AbsQCompare;
    using AbsQSet = std::set<OgaAbstractQStateNode*, AbsQCompare>;

    class OgaAbstractQStateNode : public OgaAbstractNode
    {
    private:
        OgaQStateNode * representant = nullptr; //only needed when eps_a or eps_t > 0
        std::set<OgaQStateNode*> ground_nodes;

        double values = 0;
        double visits = 0;
        double squared_values = 0; //only neede for std calculation for OGA-CAD (abs dropping)

        void _add_approx(const OgaQStateNode* q_state_node);
        void _remove_approx(const OgaQStateNode* q_state_node);

        void _add_exact(const OgaQStateNode* q_state_node);
        void _remove_exact(const OgaQStateNode* q_state_node);

    public:
        using OgaAbstractNode::OgaAbstractNode;
        [[nodiscard]] double getValues() const;
        [[nodiscard]] double getSquaredValues() const;
        [[nodiscard]] double getVisits() const;
        [[nodiscard]] OgaQStateNode* getRepresentant() const;
        void setRepresentant(OgaQStateNode* representant);
        void addExperience(double values);
        std::set<OgaQStateNode*>& getGroundNodes() { return ground_nodes; }

        static void transfer(OgaQStateNode* q_state_node, OgaAbstractQStateNode* from, OgaAbstractQStateNode* to, AbsQSet& abs_q_set, bool exact_bookkeeping, OgaBehaviorFlags& flags);
        void add(OgaQStateNode* q_state_node, bool exact_bookkeeping);
        void remove(OgaQStateNode* q_state_node, bool exact_bookkeeping);
    };

    struct AbsQCompare{
        bool operator()(const OgaAbstractQStateNode* lhs, const OgaAbstractQStateNode* rhs) const{
            if (lhs->getCount() < rhs->getCount())
                return true;
            else if (lhs->getCount() == rhs->getCount()) {
                return lhs->getId() < rhs->getId();
            }else
                return false;
        }
    };


}

#endif //OGAABSTRACTNODES_H
