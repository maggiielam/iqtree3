/*
 * phylotreesciphy.h
 *
 *  Created on: Jul 6, 2026
 *      Author: Maggie Lam
 */

#ifndef PHYLOTREESCIPHY_H
#define PHYLOTREESCIPHY_H

#include "iqtree.h"

#include <map>
#include <vector>

class BarcodeAlignment;

/**
 * Per-node ancestor list and parallel partial-likelihood values.
 * TODO(perf): currently rebuilt from scratch every likelihood evaluation, need to implement caching
 */
struct SciPhyAncestorPartials {
    std::vector<std::vector<int>> ancestors;
    std::vector<double> partial_lh;
};

/**
 * SciPhy likelihood tree
 */
class PhyloTreeSciPhy : public IQTree {
public:
    PhyloTreeSciPhy();
    explicit PhyloTreeSciPhy(Alignment *aln);
    virtual ~PhyloTreeSciPhy();

    virtual double computeLikelihood(double *pattern_lh = nullptr, bool save_log_value = true) override;

    // Single-edge likelihood evaluation used by Brent during branch-length optimization. 
    // always combines at the fixed origin pseudo-branch
    // TODO(perf): will probably change once we implement caching of partial likelihoods at internal nodes
    virtual double computeLikelihoodBranch(PhyloNeighbor *dad_branch, PhyloNode *dad,
                                           bool save_log_value = true) override;

    /**
    * Currently we just call a fresh computeLikelihood because there is no SIMD kernel for SciPhy.
    */
    virtual double computeLikelihoodFromBuffer() override;

    /**
     * Fill per-pattern log-likelihoods from the SciPhy kernel instead of from theta buffers.
     *
     */
    virtual void computePatternLikelihood(double *pattern_lh, double *cur_logl = nullptr,
                                          double *pattern_lh_cat = nullptr,
                                          SiteLoglType wsl = WSL_RATECAT) override;

    /**
     * Brent-only per-branch optimizer.  Mirrors PhyloTree::optimizeOneBranch
     * except we skip Newton-Raphson (no analytical derivatives for the SciPhy Poisson kernel yet)
     */
    virtual void optimizeOneBranch(PhyloNode *node1, PhyloNode *node2,
                                   bool clearLH = true,
                                   int maxNRStep = 100) override;

    /**
    * Outer loop over branches: sweep optimizeOneBranch across every edge,
    * iterate until tree log-likelihood change drops below tolerance.
    */
    virtual double optimizeAllBranches(int my_iterations = 100,
                                       double tolerance = TOL_LIKELIHOOD,
                                       int maxNRStep = 100) override;

protected:
    bool hasBarcodeAlignment() const;
    bool useExperimentalSciPhyLikelihood() const;

    double computeSciPhyLikelihood(double *pattern_lh, bool save_log_value);
    double computeSciPhyPatternLogLikelihood(size_t ptn) const;

    // Compute transition probability between two raw tape vectors over time t.
    double transProbVec(const std::vector<int> &parent_tape,
                        const std::vector<int> &child_tape,
                        double branch_length) const;

    // Compatibility wrapper: looks up tape vectors from compound state indices and forwards to transProbVec.
    double transProb(size_t parent_state, size_t child_state, double branch_length) const;

    /**
     * Enumerate all possible ancestral tape vectors of a given tape, by sequentially removing edits from the right end until the tape is fully unedited.  
     * Equivalent to SciPhyTreeLikelihood.getPossibleAncestors.
     */
    static std::vector<std::vector<int>> getPossibleAncestors(const std::vector<int> &tape_vec);

    // Recursive post-order build of (ancestor list, partial likelihood vector) at each node
    //   - leaf: ancestors = getPossibleAncestors(observed tape);
    //           partial_lh = [1, 0, 0, ...]   (mass on the observed value)
    //   - internal: ancestors = intersection of children's ancestor lists;
    //               partial_lh[s] = prod_child sum_c transProbVec(s, c, t_c)
    //                                                * child.partial_lh[c]
    SciPhyAncestorPartials computeSubtreeAncestorPartials(
        Node *node, Node *parent, size_t ptn,
        std::map<std::pair<int, int>, SciPhyAncestorPartials> &memo) const;

    int getObservedLeafState(Node *leaf, size_t ptn) const;

    // MARK - model parameter optimization helpers

    /**
     * Synchronize the branch length of the origin node with the current tree structure. 
     * Ensures that the origin branch length is consistent with any updates to the tree.
     */
    void syncOriginBranchLength();

    /**
     * Height of the SciPhy tree, defined as the distance frothe root to the furthest leaf
     */
    double sciphyTreeHeight() const;

};

#endif
