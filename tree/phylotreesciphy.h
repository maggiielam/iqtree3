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
#include <string>
#include <vector>

class BarcodeAlignment;
class ModelSciPhy;

// SciPhyAncestorPartials and SciPhyTapePartials are defined in phylonode.h (cached per half-edge).

/**
 * SciPhy likelihood tree
 *
 * Kernel (env IQTREE_SCIPHY_KERNEL):
 *   unset  prefix-depth kernel (SciPhyTapePartials)
 *   old    original kernel (SciPhyAncestorPartials, logTransProbVec)
 *   check  prefix-depth kernel drives the run; the old kernel recomputes every
 *          evaluation and per-tape differences are reported (correctness oracle).
 *          The oracle sums its absorbing tail directly (see logTransProbVec TODO)
 *   check-raw  same, against the unmodified old kernel
 * Other debug variables: IQTREE_SCIPHY_PRINTLL=1 prints every logL;
 * IQTREE_SCIPHY_CHECK=1 compares each NNI candidate score with the exact logL.
 * All three are read once, in the constructor.
 */
class PhyloTreeSciPhy : public IQTree {
public:
    // longest supported tape (sites); sizes the per-branch Poisson table
    static const int MAX_TAPE_SITES = 32;

    // Poisson pmf and upper tails of one branch, shared by all tapes (see phylotreesciphy.cpp)
    struct BranchTable {
        double pois[MAX_TAPE_SITES + 1];
        double tail[MAX_TAPE_SITES + 1];   // tail[r] = P(K >= r), summed directly
        bool log_space;                    // entries are logs
    };

    enum SciPhyKernel { SCIPHY_KERNEL_TAPE, SCIPHY_KERNEL_OLD, SCIPHY_KERNEL_CHECK,
                        SCIPHY_KERNEL_CHECK_RAW };

    PhyloTreeSciPhy();
    explicit PhyloTreeSciPhy(Alignment *aln);
    virtual ~PhyloTreeSciPhy();

    // check mode: print per-call-site statistics, then clear them
    void reportKernelCheck();

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

    // parallel NNI scoring (virtual swaps)
    virtual void evaluateNNIs(Branches &nniBranches, vector<NNIMove> &positiveNNIs) override;

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

    // ML pairwise distances are never computed for barcode data (compute_ml_dist is off), so
    // skip the per-thread distance buffers (4 x num_states^2 doubles each; num_states = #tapes)
    virtual void prepareToComputeDistances() override;

    // Strict-clock dating of the current topology (--clock-date): node ages and clock rate by ML,
    // tips at age 0, founder at age --origin-time. Writes <prefix>.timetree.nwk and
    // <prefix>.clockdate.txt; the tree's free branch lengths and clock rate are restored afterwards.
    void runClockDating(const std::string &prefix);

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
    void computeAncestorFullUpward();

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
                          double len, double *pattern_lh_out = nullptr) const;

    void ensureSciPhySnapshot();

    // logL over the origin branch with the active kernel (compares kernels in check mode)
    double originLikelihood(PhyloNeighbor *origin_edge, double t);

    bool buildNNILocalPartials(PhyloNode *node1, PhyloNode *node2,
                               std::vector<SciPhyAncestorPartials> &downChild,
                               std::vector<SciPhyAncestorPartials> &upParent);
    double scoreNNICandidate(PhyloNode *node1, PhyloNode *node2, int cnt,
                             double orig_len, double &optlen);

    // NNI scoring: rebuild Down(child) / UpToC(parent) for the current (swapped)
    // neighbors from cached partials, without touching the cache.
    void buildDownwardLocal(PhyloNode *child, PhyloNode *parent,
                            std::vector<SciPhyAncestorPartials> &out) const;
    // child_down: Down(child) of the swapped tree (from buildDownwardLocal); the
    // cached one on (parent -> child) still describes the subtree before the swap
    void buildUpToCLocal(PhyloNode *parent, PhyloNode *child,
                         const std::vector<SciPhyAncestorPartials> &child_down,
                         std::vector<SciPhyAncestorPartials> &out) const;

    int getObservedLeafState(Node *leaf, size_t ptn) const;

    // <----- clock dating ----->
    std::vector<double> clock_age;              // node id -> age (time before sampling)
    std::vector<PhyloNode*> clock_parent;       // node id -> parent (founder for root_int)
    std::vector<PhyloNode*> clock_post;         // internal nodes, children before parents
    double clock_T = 25.0;                      // founder age
    // computeFunction modes
    bool clock_age_active = false, clock_age_exact = false;
    PhyloNode *clock_age_node = nullptr;
    bool clock_rate_active = false;
    bool clock_scale_active = false;
    std::vector<double> clock_scale_base;       // ages before the scale move
    double clock_scale_rate = 1.0;              // clock rate before the scale move

    void clockCollect();
    void clockInitAges(double root_frac);
    void clockSetEdge(PhyloNode *a, PhyloNode *b, double len);
    void clockApplyAge(PhyloNode *v);           // lengths of the edges around v from the ages
    void clockApplyAllAges();
    void clockSetRate(double r);
    double clockRate() const;
    double clockAgeObjective(double a);
    double clockSweep(bool exact);
    double clockOptimizeRate();
    double clockOptimizeScale();
    double clockOptimize(int max_sweeps, double tol);
    // joint L-BFGS-B over relative ages r_v = a_v / a_parent (all in (0,1)) and log clock rate,
    // with gradients from one up/down snapshot (exact edge-length derivatives)
    bool clock_lbfgs_active = false;
    std::vector<PhyloNode*> clock_pre;          // internal nodes, parents before children
    double clockLBFGS();
    void clockSetVars(const double *x);
    double clockEdgeDeriv(PhyloNode *p, PhyloNode *c);
public:
    virtual double optimFunc(int nvar, double *vars) override;
    virtual double optimGradient(int nvar, double *vars, double *gradient) override;
protected:

    // <----- kernel selection and check mode ----->

    void readEnv();
    SciPhyKernel sciphy_kernel = SCIPHY_KERNEL_TAPE;
    bool print_ll = false;
    bool nni_cache_check = false;
    // IQTREE_SCIPHY_BRLEN=brent: Brent instead of Newton; IQTREE_SCIPHY_CHECK_BRLEN=1: checks
    bool brlen_brent = false;
    bool brlen_check = false;
    // Newton branch lengths in use
    bool closedFormBrlen() const { return !useOldKernel() && !checking() && !brlen_brent; }
    bool useOldKernel() const { return sciphy_kernel == SCIPHY_KERNEL_OLD; }
    bool checking() const { return sciphy_kernel >= SCIPHY_KERNEL_CHECK; }
    // old kernel's tail as a direct sum (check mode only)
    bool oracleDirectTail() const { return sciphy_kernel == SCIPHY_KERNEL_CHECK; }
    bool hasPartial(PhyloNeighbor *nei) const;

    // check mode: compare per-tape logL (new vs old), keep counts per call site
    void checkTapes(const char *where, const double *ptn_new, const double *ptn_old);
    struct CheckStats { size_t n = 0, fail = 0; double max_rel = 0.0; };
    std::map<std::string, CheckStats> check_stats;
    void checkAllBranches();

    // <----- prefix-depth kernel ----->

    // Per pattern: tape length and offset into SciPhyTapePartials::lh (L+1 slots).
    std::vector<int> tape_len;
    std::vector<size_t> tape_lh_offset;
    int max_tape_len = 0;

    // Per tip tape state s (BarcodeAlignment::getTapeVec): prefix products of insert
    // probabilities F[0..d_s] over nonzero factors, and the count of zero factors Z[k].
    // T(i -> j) = Pois(j-i) * F[j] / F[i] if Z[j] == Z[i], else 0.
    std::vector<size_t> prefix_offset;
    std::vector<double> prefix_prod;
    std::vector<unsigned char> prefix_zeros;

    // Builds the tables above once; the alignment and model are fixed afterwards.
    void initTapeKernel();
    // tape length (sites) of pattern p, from its first observed cell
    int patternTapeLength(size_t p) const;
    const ModelSciPhy &sciphyModel() const;
    const BarcodeAlignment &barcodeAlignment() const;
    int stateDepth(int s) const { return (int)(prefix_offset[s + 1] - prefix_offset[s]) - 1; }
    const double *prefixProd(int s) const { return &prefix_prod[prefix_offset[s]]; }
    const unsigned char *prefixZeros(int s) const { return &prefix_zeros[prefix_offset[s]]; }

    // depth of the longest common prefix of tape states a and b, at most bound
    int commonDepth(int a, int b, int bound) const;

    // table for a branch of the given length (mean edits = clock * length)
    void fillBranchTable(double len, BranchTable &bt) const;

    // Closed-form branch likelihood: per-tape Poisson coefficients, independent of the length
    struct BranchCoeffs {
        std::vector<double> c, e;            // per tape, tape_lh_offset layout (L+1 slots)
        std::vector<double> scale;           // per tape: log_scale terms (constant in len)
        std::vector<unsigned char> active;   // tape depends on len
        double const_logl = 0.0;             // sum over constant tapes of w * logL
    };
    // U = UpToC(parent) seen from the child, D = Down(child)
    void buildBranchCoeffs(const SciPhyTapePartials &U, const SciPhyTapePartials &D, BranchCoeffs &bc) const;
    // origin edge: the parent is the unedited founder
    void buildOriginCoeffs(const SciPhyTapePartials &bio, BranchCoeffs &bc) const;
    // logL and first/second derivatives at len; ok = false if Brent is needed
    double branchLogL(const BranchCoeffs &bc, double len, double *d1, double *d2, bool &ok) const;
    // safeguarded Newton; false = use Brent
    bool newtonBranchLength(const BranchCoeffs &bc, double x, double f, double g, double h,
                            double &best_len, double &best_logl) const;
    // optimal length of one branch; used_newton = false means fall back to Brent
    double optimizeBranchClosedForm(const BranchCoeffs &bc, double cur, double &new_len,
                                    bool &used_newton) const;
    mutable BranchCoeffs brlen_bc;           // scratch for the sequential callers
    // IQTREE_SCIPHY_CHECK_BRLEN: closed form, derivative and optimum vs the exact likelihood
    template <class Exact>
    void checkBranchClosedForm(const char *where, const BranchCoeffs &bc, double cur,
                               double newton_len, double newton_logl, Exact exact);
    void recordCheck(const std::string &key, double err, double thresh);
    void sizeTapePartials(SciPhyTapePartials &tp) const;
    SciPhyTapePartials &tapePartials(PhyloNeighbor *nei) const;

    // per-thread scratch, so the const kernel functions below can run in parallel
    struct Workspace {
        std::vector<char> redo;              // per tape: recompute in log space
        SciPhyTapePartials upbar;            // UpBar(node) in the upward pass
        SciPhyTapePartials down, up;         // local partials of an NNI candidate
        BranchCoeffs bc;                     // closed-form branch coefficients
    };
    mutable std::vector<Workspace> workspaces;
    int numWorkThreads() const { return std::max(1, num_threads); }
    Workspace &workspace(int t = 0) const;
    Workspace &threadWorkspace() const;      // the calling OpenMP thread's workspace

    // lh[i] *= sum_j T(i,j) child[j] for one child branch; underflow flags tape p in redo
    void multiplyChild(const BranchTable &bt, int p, const SciPhyTapeHead &c,
                       const double *child_lh, SciPhyTapeHead &h, double *lh,
                       std::vector<char> &redo) const;
    // support (rep, depth) of the observed kids' common prefix; headOf(kid) gives its head
    template <class HeadOf>
    void commonSupport(const std::vector<PhyloNeighbor*> &kids, HeadOf headOf,
                       int &rep, int &depth) const;

    // Down partials below dad_branch; parallel over the nodes of each height level
    void computeTapeDownward(PhyloNeighbor *dad_branch, PhyloNode *dad);
    // Down partial on dad_branch from its children's (already computed) Down partials
    void computeTapeDownNode(PhyloNeighbor *dad_branch, PhyloNode *dad, Workspace &ws) const;
    // Down partial of a node from its children's cached Down partials.
    void combineTapeChildren(const std::vector<PhyloNeighbor*> &kids, SciPhyTapePartials &out,
                             Workspace &ws) const;
    // UpToC partials on every edge; parallel over the nodes of each depth level
    void computeTapeFullUpward();
    // UpToC(node) on every (child -> node) half-edge, given UpToC(dad) on (node -> dad)
    void computeTapeUpwardNode(PhyloNode *node, PhyloNode *dad, Workspace &ws) const;
    // one branch of the parallel sweep: 0 = skipped, 1 = new_len set, 2 = needs Brent
    int sweepBranchClosedForm(PhyloNode *node1, PhyloNode *node2, double &new_len, Workspace &ws) const;
    // scores both NNI candidates of a branch without changing the tree; fallback[c] = 1 if
    // candidate c needs the sequential path
    void scoreBranchNNIsVirtual(PhyloNode *node1, PhyloNode *node2, NNIMove *moves, char *fallback,
                                Workspace &ws) const;
    // swap / unswap a candidate on the tree (sequential fallback path)
    void nniRawSwap(NNIMove &m) const;
    void nniUndoSwap(NNIMove &m) const;
    // parallel paths enabled (off in check/debug modes)
    bool parallelPathEnabled() const;
    double combineTape(const SciPhyTapePartials &up, const SciPhyTapePartials &down,
                       double len, double *pattern_lh_out = nullptr) const;
    double originClosureTape(const SciPhyTapePartials &bio, double t, double *pattern_lh_out);
    void buildTapeDownwardLocal(PhyloNode *child, PhyloNode *parent, SciPhyTapePartials &out);
    // child_down: as in buildUpToCLocal
    void buildTapeUpToCLocal(PhyloNode *parent, PhyloNode *child,
                             const SciPhyTapePartials &child_down, SciPhyTapePartials &out);
    // same, from explicit neighbor lists (scores a candidate topology without rewiring the tree)
    void buildTapeDownwardKids(const std::vector<PhyloNeighbor*> &kids, SciPhyTapePartials &out,
                               Workspace &ws) const;
    void buildTapeUpToCKids(PhyloNeighbor *pnei, const std::vector<PhyloNeighbor*> &kids,
                            PhyloNeighbor *child_nei, const SciPhyTapePartials &child_down,
                            SciPhyTapePartials &out, Workspace &ws) const;

    // One tape of a node partial in log space (fallback when the linear result underflows):
    // log lh[s] = log sum_i up[i] T(i,s) (if up, over up_len) + sum_{kids != skip} log sum_j T(s,j) kid[j].
    // Normalizes lh[0..depth] to max 1 and returns log(max); constants (log_scale) are not included.
    double tapeLogs(int p, int rep, int depth, const SciPhyTapePartials *up, int up_depth, double up_len,
                    const std::vector<PhyloNeighbor*> &kids, const PhyloNeighbor *skip, double *lh) const;
    SciPhyTapePartials tape_nni_up, tape_nni_down;      // NNI candidate partials (sequential path)
    const SciPhyTapePartials *sciphy_nni_up_tape   = nullptr;
    const SciPhyTapePartials *sciphy_nni_down_tape = nullptr;
};

#endif
