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

// SciPhyAncestorPartials is defined in phylonode.h (so PhyloNeighbor can own a
// per-directed-edge cache of it, mirroring partial_lh), and is visible here via
// the iqtree.h -> phylotree.h -> phylonode.h include chain.

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

    /**
     * No-op for SciPhy. 
     */
    virtual double optimizeRootPosition(int root_dist, bool write_info,
                                        double logl_epsilon) override;

    /**
     * Brent objective for branch-length optimization. 
     */
    virtual double computeFunction(double value) override;


    virtual NNIMove getBestNNIForBran(PhyloNode *node1, PhyloNode *node2,
                                      NNIMove *nniMoves = nullptr) override;

    /**
     * Invalidate the SciPhy up/down snapshot after a real topology change, then
     * defer to the base swap.
     */
    virtual void doNNI(NNIMove &move, bool clearLH = true) override;

    /**
     * Apply only the central branch length recorded by getBestNNIForBran.
     */
    virtual void changeNNIBrans(NNIMove &nnimove) override;

protected:
    bool hasBarcodeAlignment() const;

    double computeSciPhyLikelihood(double *pattern_lh, bool save_log_value);

    // Fill all toward-root (upward) partials from a valid downward snapshot.
    void computeSciPhyFullUpward();

    // Optimize one branch's length against the current up/down snapshot
    void optimizeOneBranchCombine(PhyloNode *node1, PhyloNode *node2);

    // Branch-sweep state consumed by computeFunction() when in combine mode.
    bool sciphy_combine_active = false;
    PhyloNode *sciphy_combine_parent = nullptr;
    PhyloNode *sciphy_combine_child  = nullptr;

    // NNI-scoring state consumed by computeFunction() when sciphy_nni_active
    bool sciphy_nni_active = false;
    const std::vector<SciPhyAncestorPartials> *sciphy_nni_up   = nullptr;
    const std::vector<SciPhyAncestorPartials> *sciphy_nni_down = nullptr;

    // True when the neighbor-cached downward + upward partials form a consistent
    // snapshot of the CURRENT tree. 
    bool sciphy_snapshot_valid = false;

    // Compute transition probability between two raw tape vectors over time t.
    double transProbVec(const std::vector<int> &parent_tape,
                        const std::vector<int> &child_tape,
                        double branch_length) const;

    /**
     * Enumerate all possible ancestral tape vectors of a given tape, by sequentially removing edits from the right end until the tape is fully unedited.  
     * Equivalent to SciPhyTreeLikelihood.getPossibleAncestors.
     */
    static std::vector<std::vector<int>> getPossibleAncestors(const std::vector<int> &tape_vec);

    // Cache-gated post-order (downward) partial for the subtree on the far side
    // of `dad_branch` (away from `dad`), for ALL patterns, stored on the
    // neighbor (dad_branch->sciphy_partial) and gated by dad_branch->
    // partial_lh_computed
    //   - leaf: ancestors = getPossibleAncestors(observed tape); mass on observed
    //   - internal: ancestors = intersection of children's ancestor lists;
    //               partial_lh[s] = prod_child sum_c transProbVec(s,c,t_c)*child_lh[c]
    void computeSciPhyDownward(PhyloNeighbor *dad_branch, PhyloNode *dad);

    // Pre-order (upward / toward-root) pass.
    void computeSciPhyUpward(PhyloNode *node, PhyloNode *dad);

    // Whole-tree log-likelihood obtained by combining at the edge (parent,child):
    //   L = sum_p log( sum_{s_par,s_child} UpToC(parent)[s_par]
    //                    * transProbVec(s_par -> s_child, len) * Down(child)[s_child] )
    //       + scales, weighted by pattern frequency.
    double combineAtBranch(PhyloNode *parent, PhyloNode *child);

    // whole-tree log-likelihood obtained by joining an upward partial `up`
    // (function of the parent state) and a downward partial `down` (function of
    // the child state) across a branch of length `len`:
    //   L = sum_p ptn_freq[p] * ( log sum_{s_par,s_child} up[p][s_par]
    //         * transProbVec(s_par -> s_child, len) * down[p][s_child]
    //       + up[p].log_scale + down[p].log_scale )
    double combineVectors(const std::vector<SciPhyAncestorPartials> &up,
                          const std::vector<SciPhyAncestorPartials> &down,
                          double len) const;

    // Ensure a consistent downward+upward snapshot of the CURRENT tree exists
    void ensureSciPhySnapshot();

    // Build Down(child) for `child` given its CURRENT neighbor set, reading the
    // (cached) downward partials of its children (all neighbors except `parent`).
    void buildDownwardLocal(PhyloNode *child, PhyloNode *parent,
                            std::vector<SciPhyAncestorPartials> &out) const;

    // Build UpToC(parent) as a function of parent's state, for the specific
    // `child` edge, from parent's CURRENT neighbor set
    void buildUpToCLocal(PhyloNode *parent, PhyloNode *child,
                         std::vector<SciPhyAncestorPartials> &out) const;

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
