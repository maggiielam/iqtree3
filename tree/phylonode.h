//
// C++ Interface: phylonode
//
// Description:
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef PHYLONODE_H
#define PHYLONODE_H

#include "node.h"
#include <vector>
#include <memory>

typedef unsigned short UBYTE;

/**
 * direction of a Neighbor from the root, for rooted tree only
 */
enum RootDirection {UNDEFINED_DIRECTION, TOWARD_ROOT, AWAYFROM_ROOT};

/**
 * Per-directed-edge cached partials for the SciPhy likelihood.
 * 
 * ancestors[i] is a possible tape state at this node and `partial_lh[i]` its
 * conditional likelihood; the true value is partial_lh[i] * exp(log_scale)
 * (log_scale keeps the stored partials O(1) to avoid underflow on deep trees).
 */
struct SciPhyAncestorPartials {
    std::vector<std::vector<int>> ancestors;
    std::vector<double> partial_lh;
    double log_scale = 0.0;
    // dropout: tape present but unobserved; no states, constant carried in log_scale
    bool wildcard = false;
    // Upward partials only. A wildcard up partial (the subtree below is all-missing) keeps its
    // state dependence here instead of collapsing to a constant:
    //   value(s) = exp(log_scale - totalMass(anchor)) * sum_j anchor(j) * Trans(anchor_j -> s, anchor_len)
    // anchor = the nearest non-wildcard up partial above (or the founder), so an observed subtree
    // moved below this edge can still be scored exactly. Null = no state dependence kept.
    std::shared_ptr<const SciPhyAncestorPartials> anchor;
    double anchor_len = 0.0;
};

/**
 * Per-directed-edge SciPhy partials in prefix-depth form, flat over all tapes.
 *
 * Every possible state of tape p is a prefix of tip tape state head[p].rep, of depth
 * 0..head[p].depth; lh[offset_p + k] is the conditional likelihood of the depth-k
 * prefix (offsets: PhyloTreeSciPhy::tape_lh_offset). True value: lh * exp(log_scale).
 */
struct SciPhyTapeHead {
    double log_scale = 0.0;
    int rep = -1;
    short depth = -1;
    // dropout: tape present but unobserved; no states, constant carried in log_scale
    bool wildcard = false;
};

struct SciPhyTapePartials {
    std::vector<SciPhyTapeHead> head;   // one per pattern
    std::vector<double> lh;
};

/**
A neighbor in a phylogenetic tree

    @author BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>
 */
class PhyloNeighbor : public Neighbor {
    friend class PhyloNode;
    friend class PhyloTree;
    friend class IQTree;
    friend class PhyloTreeSciPhy;   // SciPhy caches sciphy_partial per directed edge
    friend class PhyloSuperTree;
    friend class PhyloTreeMixlen;
    friend class MemSlotVector;
    friend class ParsTree;

public:
    friend class TinaTree;
    friend class PhyloSuperTreePlen;

    /**
        construct class with a node and length		
        @param anode the other end of the branch
        @param alength length of branch
     */
    PhyloNeighbor(Node *anode, double alength) : Neighbor(anode, alength) {
        partial_lh = NULL;
        scale_num = NULL;
        partial_lh_computed = 0;
        lh_scale_factor = 0.0;
        partial_pars = NULL;
        direction = UNDEFINED_DIRECTION;
        size = 0;
        sciphy_partial = NULL;
        sciphy_tape_partial = NULL;
    }

    /**
        construct class with a node and length
        @param anode the other end of the branch
        @param alength length of branch
        @param aid branch ID
     */
    PhyloNeighbor(Node *anode, double alength, int aid) : Neighbor(anode, alength, aid) {
        partial_lh = NULL;
        scale_num = NULL;
        partial_lh_computed = 0;
        lh_scale_factor = 0.0;
        partial_pars = NULL;
        direction = UNDEFINED_DIRECTION;
        size = 0;
        sciphy_partial = NULL;
        sciphy_tape_partial = NULL;
    }

    /**
     construct class with another Neighbor
     @param nei another Neighbor
     */
    PhyloNeighbor(PhyloNeighbor *nei) : Neighbor(nei) {
        partial_lh = NULL;
        scale_num = NULL;
        partial_lh_computed = 0;
        lh_scale_factor = 0.0;
        partial_pars = NULL;
        direction = nei->direction;
        size = nei->size;
        sciphy_partial = NULL;
        sciphy_tape_partial = NULL;
    }

    /**
     free the SciPhy partial cache (null for every non-SciPhy neighbor).
     Neighbor has a virtual destructor, so this runs on delete via Neighbor*.
     */
    virtual ~PhyloNeighbor() {
        delete sciphy_partial;
        delete sciphy_tape_partial;
    }


    /**
     allocate a new Neighbor by just copying from this one
     @return pointer to newly created Neighbor
     */
    virtual Neighbor* newNeighbor() {
        return (new PhyloNeighbor(this));
    }

    /**
        tell that the partial likelihood vector is not computed
     */
    inline void clearPartialLh() {
        partial_lh_computed = 0;
    }

    /**
     *  tell that the partial likelihood vector is computed
     */
    inline void unclearPartialLh() {
        partial_lh_computed = 1;
    }

    /**
        clear all partial likelihood recursively in forward direction
        @param dad dad of this neighbor
     */
    void clearForwardPartialLh(Node *dad);

    /**
        DEPRECATED, moved to PhyloTree
        if partial_lh is NULL, reorient partial_lh (LM_PER_NODE technique)
        @param dad dad of this neighbor
    */
//    void reorientPartialLh(Node *dad);

	/**
	* For Upper Bounds analysis: get partial likelihood and lh scale factor
	*/
	double* get_partial_lh(){
	return partial_lh;
	}

	double get_lh_scale_factor(){
	return lh_scale_factor;
	}

	int get_partial_lh_computed(){
	return partial_lh_computed;
	}

	/**
	 * true if this Neighbor is directed towards the root
	 */
	bool isTowardsRoot() {
		ASSERT(direction != UNDEFINED_DIRECTION);
		return (direction == TOWARD_ROOT);
	}

    int getSize() {
        return size;
    }

private:

    /**
        true if the partial likelihood was computed
     */
    int partial_lh_computed;

    /**
        vector containing the partial likelihoods
     */
    double *partial_lh;

    /**
        cache: one SciPhyAncestorPartials per alignment
        pattern for the subtree on the far side of this directed edge.
     */
    std::vector<SciPhyAncestorPartials> *sciphy_partial;

    /**
        same cache for the prefix-depth kernel (PhyloTreeSciPhy, default kernel)
     */
    SciPhyTapePartials *sciphy_tape_partial;

    /**
        likelihood scaling factor
     */
    double lh_scale_factor;

    /**
        vector containing number of scaling events per pattern // NEW!
     */
    UBYTE *scale_num;

    /**
        vector containing the partial parsimony scores
     */
    UINT *partial_pars;

    /**
     * direction of the Neighbor in a rooted tree
     */
    RootDirection direction;

    /** size of subtree below this neighbor in terms of number of taxa */
    int size;

};

/**
A node in a phylogenetic tree

    @author BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>
 */
class PhyloNode : public Node {
    friend class PhyloTree;

public:
    /**
        constructor
     */
    PhyloNode();

    /**
        constructor
        @param aid id of this node
     */
    PhyloNode(int aid);

    /**
        constructor
        @param aid id of this node
        @param aname name of this node
     */
    PhyloNode(int aid, int aname);

    /**
        constructor
        @param aid id of this node
        @param aname name of this node
     */
    PhyloNode(int aid, const char *aname);

    /**
        initialization
     */
    void init();

    /**
        add a neighbor
        @param node the neighbor node
        @param length branch length
        @param id branch ID
     */
    virtual void addNeighbor(Node *node, double length, int id = -1);



    /**
        tell that all partial likelihood vectors below this node are not computed
     */
    void clearAllPartialLh(bool make_null, PhyloNode *dad);

    /**
        tell that all partial likelihood vectors (in reverse direction) below this node are not computed
     */
    void clearReversePartialLh(PhyloNode *dad);

    void computeReversePartialLh(PhyloNode *dad);

    /** 
        compute the size (#taxa) of the subtree rooted at this node
        using buffered 'size' attribute if computed beforehand
        @param dad dad of this node
    */
    int computeSize(Node *dad);

};


/**
    Node vector
 */
typedef vector<PhyloNode*> PhyloNodeVector;


#endif
