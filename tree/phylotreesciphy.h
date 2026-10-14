/*
 * phylotreesciphy.h
 *
 *  Created on: Jul 6, 2026
 *      Author: Maggie Lam
 */

#ifndef PHYLOTREESCIPHY_H
#define PHYLOTREESCIPHY_H

#include "iqtree.h"

#include <vector>

class BarcodeAlignment;

// SciPhyAncestorPartials is defined in phylonode.h (cached per half-edge).

/**
 * SciPhy likelihood tree
 */
class PhyloTreeSciPhy : public IQTree {
public:
    PhyloTreeSciPhy();
    explicit PhyloTreeSciPhy(Alignment *aln);
    virtual ~PhyloTreeSciPhy();

    virtual double computeLikelihood(double *pattern_lh = nullptr, bool save_log_value = true) override;

    // Full-tree likelihood (no per-branch shortcut).
    virtual double computeLikelihoodBranch(PhyloNeighbor *dad_branch, PhyloNode *dad,
                                           bool save_log_value = true) override;

    // No SIMD buffer: recomputes the likelihood.
    virtual double computeLikelihoodFromBuffer() override;

    virtual void computePatternLikelihood(double *pattern_lh, double *cur_logl = nullptr,
                                          double *pattern_lh_cat = nullptr,
                                          SiteLoglType wsl = WSL_RATECAT) override;

    // Brent only (no analytical derivatives).
    virtual void optimizeOneBranch(PhyloNode *node1, PhyloNode *node2,
                                   bool clearLH = true,
                                   int maxNRStep = 100) override;

    // Jacobi sweeps over all branches against one up/down snapshot, until convergence.
    virtual double optimizeAllBranches(int my_iterations = 100,
                                       double tolerance = TOL_LIKELIHOOD,
                                       int maxNRStep = 100) override;

    // No-op: the root is fixed.
    virtual double optimizeRootPosition(int root_dist, bool write_info,
                                        double logl_epsilon) override;

    // Brent objective for branch-length optimization.
    virtual double computeFunction(double value) override;

    virtual NNIMove getBestNNIForBran(PhyloNode *node1, PhyloNode *node2,
                                      NNIMove *nniMoves = nullptr) override;

    // Base swap + invalidate the up/down snapshot.
    virtual void doNNI(NNIMove &move, bool clearLH = true) override;

    // Apply only the central branch length recorded by getBestNNIForBran.
    virtual void changeNNIBrans(NNIMove &nnimove) override;

    // Memory: the base class's dense buffers are sized by num_states (= #distinct tape
    // vectors, ~1e4 on real data); barcode data allocates only per-pattern arrays.
    using IQTree::getMemoryRequired;
    using IQTree::initializeAllPartialLh;
    virtual uint64_t getMemoryRequired(size_t ncategory = 1, bool full_mem = false) override;
    virtual void initializeAllPartialLh() override;
    // No-op for barcode data (no base partial_lh buffers to move).
    virtual void reorientPartialLh(PhyloNeighbor* dad_branch, Node *dad) override;

    // Keep identical cells: de-duplication rebuilds a plain Alignment, losing the
    // BarcodeAlignment type (the model would fall back to ModelMorphology).
    virtual void removeIdenticalSeqs(Params &params) override;

    // For a user starting tree, set unknown branch lengths directly instead of
    // allocating the parsimony buffers (O(N*S) memory).
    virtual int wrapperFixNegativeBranch(bool force_change) override;

protected:
    bool hasBarcodeAlignment() const;

    double computeSciPhyLikelihood(double *pattern_lh, bool save_log_value);

    // Founder -> root_int over the origin branch.
    double originClosure(const std::vector<SciPhyAncestorPartials> &bio,
                         double t, double *pattern_lh_out);

    void optimizeOriginBranch();
    bool sciphy_origin_active = false;   // computeFunction mode: origin branch

    // Fill all upward partials from a valid downward pass.
    void computeSciPhyFullUpward();

    // Optimize one branch length against the current up/down snapshot.
    void optimizeOneBranchCombine(PhyloNode *node1, PhyloNode *node2);

    // computeFunction mode: branch sweep
    bool sciphy_combine_active = false;
    PhyloNode *sciphy_combine_parent = nullptr;
    PhyloNode *sciphy_combine_child  = nullptr;

    // computeFunction mode: NNI candidate scoring
    bool sciphy_nni_active = false;
    const std::vector<SciPhyAncestorPartials> *sciphy_nni_up   = nullptr;
    const std::vector<SciPhyAncestorPartials> *sciphy_nni_down = nullptr;

    // Cached downward + upward partials match the current tree.
    bool sciphy_snapshot_valid = false;

    // log P(parent tape -> child tape) over branch_length (-inf if impossible).
    // Edits arrive as a Poisson process (mean clock * length), fill sites left to
    // right, and draw the insert from edit_probs.
    double logTransProbVec(const std::vector<int> &parent_tape,
                           const std::vector<int> &child_tape,
                           double branch_length) const;

    // All ancestors of a tape: remove edits from the right until unedited.
    static std::vector<std::vector<int>> getPossibleAncestors(const std::vector<int> &tape_vec);

    // Downward partial of the subtree below dad_branch, cached on the half-edge.
    void computeSciPhyDownward(PhyloNeighbor *dad_branch, PhyloNode *dad);

    // Upward (pre-order) pass.
    void computeSciPhyUpward(PhyloNode *node, PhyloNode *dad);

    // Whole-tree logL from the cached partials at edge (parent, child).
    double combineAtBranch(PhyloNode *parent, PhyloNode *child);

    // Whole-tree logL from an up partial (parent state) and a down partial
    // (child state) joined by a branch of length len.
    double combineVectors(const std::vector<SciPhyAncestorPartials> &up,
                          const std::vector<SciPhyAncestorPartials> &down,
                          double len) const;

    void ensureSciPhySnapshot();

    // NNI scoring: rebuild Down(child) / UpToC(parent) for the current (swapped)
    // neighbors from cached partials, without touching the cache.
    void buildDownwardLocal(PhyloNode *child, PhyloNode *parent,
                            std::vector<SciPhyAncestorPartials> &out) const;
    void buildUpToCLocal(PhyloNode *parent, PhyloNode *child,
                         std::vector<SciPhyAncestorPartials> &out) const;

    int getObservedLeafState(Node *leaf, size_t ptn) const;
};

#endif
