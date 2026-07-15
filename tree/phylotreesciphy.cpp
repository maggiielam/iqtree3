#include "phylotreesciphy.h"

#include "alignment/barcodealignment.h"
#include "model/modelsciphy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>

PhyloTreeSciPhy::PhyloTreeSciPhy() : IQTree() {} 

PhyloTreeSciPhy::PhyloTreeSciPhy(Alignment *aln) : IQTree(aln) {}

PhyloTreeSciPhy::~PhyloTreeSciPhy() {}

bool PhyloTreeSciPhy::hasBarcodeAlignment() const {
    return dynamic_cast<BarcodeAlignment*>(this->aln) != nullptr;
}

bool PhyloTreeSciPhy::useExperimentalSciPhyLikelihood() const {
    const char *flag = std::getenv("IQTREE_SCIPHY_EXPERIMENT");
    return flag && std::string(flag) == "1";
}

int PhyloTreeSciPhy::getObservedLeafState(Node *leaf, size_t ptn) const {
    if (!leaf || !leaf->isLeaf()) {
        return aln->STATE_UNKNOWN;
    }
    if (leaf->id < 0 || (size_t)leaf->id >= aln->getNSeq()) {
        return aln->STATE_UNKNOWN;
    }
    return (int)aln->at(ptn)[leaf->id];
}

namespace {

// Poisson PMF P(K = k | mean = lambda)
double poissonPMF(int k, double mean) {
    if (k < 0 || mean < 0.0) return 0.0;
    if (mean == 0.0) return (k == 0) ? 1.0 : 0.0;
    const double log_p = (double)k * std::log(mean) - mean
                         - std::lgamma((double)k + 1.0);
    return std::exp(log_p);
}

double maxDistToLeafHelper(Node *node, Node *parent) {
    if (node->isLeaf()) return 0.0;
    double max_dist = 0.0;
    FOR_NEIGHBOR_IT(node, parent, it) {
        double child_dist = maxDistToLeafHelper((*it)->node, node) + (*it)->length;
        if (child_dist > max_dist) max_dist = child_dist;
    }
    return max_dist;
}
}  // anonymous namespace

double PhyloTreeSciPhy::sciphyTreeHeight() const {
    if (!rooted || !root) return 0.0;
    if (root->neighbors.empty()) return 0.0;
    Node *root_int = root->neighbors[0]->node;
    return maxDistToLeafHelper(root_int, root);
}

void PhyloTreeSciPhy::syncOriginBranchLength() {
    if (!rooted || !root || root->neighbors.empty()) return;

    const ModelSciPhy *sm = dynamic_cast<const ModelSciPhy*>(model);
    if (!sm) return;
    const double lambda = sm->getClockRate();

    double slack = std::max(0.0, sm->getOriginTime() - sciphyTreeHeight());

    PhyloNeighbor *edge0 = (PhyloNeighbor*) root->neighbors[0];
    if (!edge0) return;
    edge0->length = slack;

    PhyloNeighbor *edge0_back = (PhyloNeighbor*) edge0->node->findNeighbor(root);
    if (edge0_back) edge0_back->length = slack;
}

double PhyloTreeSciPhy::transProbVec(const std::vector<int> &parent,
                                     const std::vector<int> &child,
                                     double branch_length) const {
    // SciPhySubstitutionModel.getSequenceTransitionProbability
    // formula:
    //   - the whole tape accumulates edits as a single Poisson process whose
    //     mean over this branch is `distance` (= sciphy_lambda * branch_length);
    //   - edits fill the leftmost unedited slot in order;
    //   - each new edit independently picks a scar type from `editProbs`
    //     (uniform 1/M here as a placeholder for a future free distribution).

    const ModelSciPhy *sm = dynamic_cast<const ModelSciPhy*>(model);
    if (!sm) return 0.0;
    const double lambda = sm->getClockRate();
    const auto &edit_probs = sm->getEditProbs();


    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) {
        return 0.0;
    }
    if (parent.size() != child.size() || parent.empty()) {
        return 0.0;
    }

    const int L = (int)parent.size();
    const int M = std::max(1, barcode_aln->getMaxInsertionType());

    // Extract the non-zero scars from each tape, in left-to-right order.
    std::vector<int> parent_scars;
    std::vector<int> child_scars;
    parent_scars.reserve((size_t)L);
    child_scars.reserve((size_t)L);
    for (int x : parent) if (x != 0) parent_scars.push_back(x);
    for (int x : child)  if (x != 0) child_scars.push_back(x);

    // parent must carry no more edits than child.
    if (parent_scars.size() > child_scars.size()) return 0.0;

    std::vector<int> new_inserts = child_scars;
    for (int p_scar : parent_scars) {
        auto it = std::find(new_inserts.begin(), new_inserts.end(), p_scar);
        if (it == new_inserts.end()) {
            return 0.0;
        }
        new_inserts.erase(it);
    }

    const int nr_possible = L - (int)parent_scars.size();
    const int k = (int)new_inserts.size();
    if (k > nr_possible) return 0.0;  // logically unreachable, defensive.

    // Scar-type factor.  Use the model's edit_probs when available
    // (populated from --edit-probs or defaulted to uniform 1/M by
    // ModelFactory).  Fall back to uniform 1/M defensively if the vector
    // is somehow shorter than M -- this shouldn't happen in normal use,
    // since ModelFactory enforces edit_probs.size() == M at model
    // construction time.
    const double per_scar_uniform = 1.0 / (double)M;
    const int n_edit_probs = (int)edit_probs.size();
    double insert_factor = 1.0;
    for (int scar : new_inserts) {
        if (scar < 1 || scar > M) return 0.0;
        insert_factor *= (scar - 1 < n_edit_probs)
                          ? edit_probs[scar - 1]
                          : per_scar_uniform;
    }

    const double distance = std::max(0.0, lambda * branch_length);

    if (k == nr_possible) {
        // Absorbing case: tape is now fully edited.  Probability is the
        // tail mass P(>= nr_possible events) under Poisson(distance).
        double absorb_prob = 1.0;
        for (int i = 0; i < nr_possible; ++i) {
            absorb_prob -= poissonPMF(i, distance);
        }
        if (absorb_prob < 0.0) absorb_prob = 0.0;
        return absorb_prob * insert_factor;
    }
    return poissonPMF(k, distance) * insert_factor;
}

double PhyloTreeSciPhy::transProb(size_t parent_state, size_t child_state, double branch_length) const {
    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) {
        return 0.0;
    }
    return transProbVec(barcode_aln->getTapeVec((int)parent_state),
                        barcode_aln->getTapeVec((int)child_state),
                        branch_length);
}

std::vector<std::vector<int>>
PhyloTreeSciPhy::getPossibleAncestors(const std::vector<int> &tape_vec) {
    // Walk the tape from right to left, clearing edits one at a time.  Every
    // partial sequence we visit is a valid ancestral state under the SciPhy
    // model (edits fill from the front of the tape and are absorbing, so any
    // strict prefix of edits is a legal ancestor).
    std::vector<std::vector<int>> ancestors;
    if (tape_vec.empty()) {
        return ancestors;
    }
    ancestors.push_back(tape_vec);
    std::vector<int> ancestor = tape_vec;
    for (int i = (int)tape_vec.size() - 1; i >= 0; --i) {
        if (tape_vec[i] != 0) {
            ancestor[i] = 0;
            ancestors.push_back(ancestor);
        }
    }
    return ancestors;
}

std::vector<double> PhyloTreeSciPhy::getRootPrior() const {
    // Retained for backwards compatibility.  The per-node ancestor list path
    // uses an inline "is all zeros" test on the candidate ancestor instead.
    const int nstates = std::max(2, model->num_states);
    std::vector<double> prior((size_t)nstates, 0.0);

    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) {
        const double uniform = 1.0 / (double)nstates;
        std::fill(prior.begin(), prior.end(), uniform);
        return prior;
    }

    int zero_state = -1;
    for (int s = 0; s < barcode_aln->getNumTapeStates(); ++s) {
        const std::vector<int> &vec = barcode_aln->getTapeVec(s);
        bool all_zero = true;
        for (int v : vec) {
            if (v != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            zero_state = s;
            break;
        }
    }

    if (zero_state >= 0 && zero_state < nstates) {
        prior[(size_t)zero_state] = 1.0;
        return prior;
    }

    const double uniform = 1.0 / (double)nstates;
    std::fill(prior.begin(), prior.end(), uniform);
    return prior;
}

namespace {

// Vector intersection that preserves the order of `a` and removes any element
// not present in `b`.  O(|a|*|b|) which is fine for the small per-node
// ancestor lists we expect (typically <= number of edits in any descendant
// tape, so single digits).
std::vector<std::vector<int>> intersectAncestorLists(
        const std::vector<std::vector<int>> &a,
        const std::vector<std::vector<int>> &b) {
    std::vector<std::vector<int>> out;
    out.reserve(std::min(a.size(), b.size()));
    for (const auto &x : a) {
        for (const auto &y : b) {
            if (x == y) {
                out.push_back(x);
                break;
            }
        }
    }
    return out;
}

bool tapeIsAllZero(const std::vector<int> &v) {
    if (v.empty()) return false;
    for (int x : v) {
        if (x != 0) return false;
    }
    return true;
}

}  // anonymous namespace

SciPhyAncestorPartials PhyloTreeSciPhy::computeSubtreeAncestorPartials(
        Node *node, Node *parent, size_t ptn,
        std::map<std::pair<int, int>, SciPhyAncestorPartials> &memo) const {
    std::pair<int, int> key(node ? node->id : -999999,
                            parent ? parent->id : -999999);
    auto it_mem = memo.find(key);
    if (it_mem != memo.end()) {
        return it_mem->second;
    }

    SciPhyAncestorPartials result;
    if (!node) {
        memo[key] = result;
        return result;
    }

    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) {
        memo[key] = result;
        return result;
    }

    if (node->isLeaf()) {
        // ----------------------------------------------------------------
        // Phase 2: synthetic __root__ leaf = SciPhy founder-state delta.
        // ----------------------------------------------------------------
        // The synthetic leaf inserted by convertToRooted() is not a real
        // observed taxon -- it stands in for the unobserved biological
        // founder at the origin of the experiment.  Under SciPhy the
        // founder state is always the all-zeros tape (no scars accumulated
        // yet).  So we hard-wire this leaf's:
        //   ancestor list  = a single element, the all-zeros tape
        //   partial_lh     = mass 1.0 on that single state
        //
        // This is the "delta prior on (0,...,0) at the origin" that the
        // rewrite comment on the OLD combination loop was waiting for.
        // Once this is set, the ancestor-set intersection at root_int
        // (the internal node just below the synthetic root) will
        // automatically be constrained to states reachable from the
        // all-zeros founder via a prefix-of-edits trail.  And the
        // combination loop in computeSciPhyPatternLogLikelihood collapses
        // to the SciPhy origin-branch closure:
        //     P(tree) = sum_s transProbVec((0,...,0), s, t_origin) * partial(s)
        //
        // This matches the BEAST SciPhy reference implementation exactly
        // (see calculateOriginPartial in the Java code).
        if (node->name == ROOT_NAME) {
            int site_count = 0;
            const auto &tsc = barcode_aln->getTapeSiteCounts();
            if (!tsc.empty()) site_count = tsc[0];
            result.ancestors.emplace_back((size_t)site_count, 0);
            result.partial_lh.push_back(1.0);
            memo[key] = result;
            return result;
        }

        // Regular observed leaf.
        //
        // Leaf ancestor list = prefix-of-edits trail of the observed tape.
        // Partial likelihood places all mass on the observed value (slot 0
        // of getPossibleAncestors output); the rest of the list exists so
        // that the parent's intersection sees the correct possibilities.
        const int obs = getObservedLeafState(node, ptn);
        if (obs < 0 || obs >= barcode_aln->getNumTapeStates()) {
            // Unknown/ambiguous leaf: fall back to "no edits", a single
            // ancestor state with full mass.  This branch is defensive --
            // buildTapeAlignment always assigns a valid compound state to
            // every leaf, so we should not normally reach here.
            int site_count = 0;
            const auto &tsc = barcode_aln->getTapeSiteCounts();
            if (!tsc.empty()) site_count = tsc[0];
            result.ancestors.emplace_back((size_t)site_count, 0);
            result.partial_lh.push_back(1.0);
            memo[key] = result;
            return result;
        }
        const std::vector<int> &leaf_tape = barcode_aln->getTapeVec(obs);
        result.ancestors = getPossibleAncestors(leaf_tape);
        result.partial_lh.assign(result.ancestors.size(), 0.0);
        if (!result.partial_lh.empty()) {
            result.partial_lh[0] = 1.0;
        }
        memo[key] = result;
        return result;
    }

    // Internal node: gather child neighbors (everything except the edge back
    // toward `parent`).
    std::vector<const Neighbor*> children;
    for (NeighborVec::const_iterator nit = node->neighbors.begin();
         nit != node->neighbors.end(); ++nit) {
        if ((*nit)->node != parent) {
            children.push_back(*nit);
        }
    }
    if (children.empty()) {
        memo[key] = result;
        return result;
    }

    // Recursively compute each child's (ancestor list, partial_lh).
    std::vector<SciPhyAncestorPartials> child_data;
    child_data.reserve(children.size());
    for (const Neighbor *child_nei : children) {
        child_data.push_back(
            computeSubtreeAncestorPartials(child_nei->node, node, ptn, memo));
    }

    // Ancestor list at this node = intersection of children's ancestor lists.
    // If the intersection is empty, the data is inconsistent with the model
    // at this internal node and the partial likelihood there must be zero;
    // we simply return an empty (ancestors, partial_lh) pair and let the
    // caller propagate the zero upward.
    result.ancestors = child_data[0].ancestors;
    for (size_t k = 1; k < child_data.size(); ++k) {
        result.ancestors = intersectAncestorLists(result.ancestors,
                                                  child_data[k].ancestors);
        if (result.ancestors.empty()) break;
    }
    result.partial_lh.assign(result.ancestors.size(), 0.0);

    // For each candidate ancestor s at this node compute
    //   partial_lh[s] = prod_{child}
    //                     sum_{c in child.ancestors}
    //                       transProbVec(s, c, t_child) * child.partial_lh[c]
    for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx) {
        const std::vector<int> &s_vec = result.ancestors[s_idx];
        double lh_s = 1.0;
        for (size_t k = 0; k < children.size(); ++k) {
            const Neighbor *child_nei = children[k];
            const SciPhyAncestorPartials &cd = child_data[k];
            double sum_child = 0.0;
            for (size_t c_idx = 0; c_idx < cd.ancestors.size(); ++c_idx) {
                if (cd.partial_lh[c_idx] <= 0.0) continue;
                const double tr = transProbVec(s_vec, cd.ancestors[c_idx],
                                               child_nei->length);
                if (tr > 0.0) {
                    sum_child += tr * cd.partial_lh[c_idx];
                }
            }
            lh_s *= sum_child;
            if (lh_s <= 0.0) break;
        }
        result.partial_lh[s_idx] = lh_s;
    }

    memo[key] = result;
    return result;
}

double PhyloTreeSciPhy::computeSciPhyPatternLogLikelihood(size_t ptn) const {
    // ------------------------------------------------------------------
    // Phase 3 + 4: deterministic combination edge and origin-branch closure.
    // ------------------------------------------------------------------
    // The combination edge is ALWAYS the origin pseudo-branch: the unique
    // edge between the synthetic __root__ leaf (root) and its sole
    // neighbor (root_int, an internal node inserted by convertToRooted()).
    //
    // Why not use current_it / current_it_back like the OLD code?
    //   Because current_it is owned by the branch-length optimizer.  Brent
    //   installs the branch it is currently optimizing there so that its
    //   computeFunction(x) callback can update lengths in place.  Reading
    //   current_it for LL combination coupled two independent concerns:
    //   "which branch is being optimized" and "which edge closes the tree
    //   likelihood".  Under a non-reversible model those must be
    //   independent -- the closure edge has to be fixed for the tree LL to
    //   be well-defined, regardless of what Brent is doing.
    //
    // Why is this correct?
    //   The synthetic root leaf's ancestor list is by construction the
    //   singleton {(0,...,0)} with mass 1.0 (see Phase 2 in
    //   computeSubtreeAncestorPartials).  So the outer sum below collapses
    //   to a single term and the whole closure becomes:
    //     P(pattern) = sum_s transProbVec((0,...,0), s, t_origin) * partial(s)
    //   which is exactly the SciPhy origin-branch closure formula used by
    //   BEAST's SciPhyTreeLikelihood.calculateOriginPartial.
    //
    // Defensive invariants that must hold before the recursion runs.  These
    // are guaranteed by ModelSciPhy's constructor, which triggers
    // convertToRooted() on the tree via ModelMarkov's adapt_tree path.
    if (!rooted || !root || root->name != ROOT_NAME) {
        return -std::numeric_limits<double>::infinity();
    }
    if (root->neighbors.empty()) {
        return -std::numeric_limits<double>::infinity();
    }

    // The synthetic root leaf has exactly one neighbor (root_int) by
    // construction.  That neighbor's edge is the origin pseudo-branch.
    PhyloNeighbor *origin_edge = (PhyloNeighbor*)root->neighbors[0];
    Node *root_like  = root;                // synthetic __root__ leaf: founder delta
    Node *other_side = origin_edge->node;   // root_int: top of the biological tree

    // Recurse into both halves at the origin pseudo-branch.  Because
    // root_like is a leaf (with the special-cased founder partials), the
    // first recursion returns immediately with the singleton ancestor
    // list.  The second recursion walks the entire biological tree below
    // root_int, doing the ancestor-set intersection / child-product
    // computation that was validated against the SciPhy Python reference
    // to 3.7e-11 on the cherry and 3-leaf tests.
    std::map<std::pair<int, int>, SciPhyAncestorPartials> memo;
    const SciPhyAncestorPartials root_data =
        computeSubtreeAncestorPartials(root_like, other_side, ptn, memo);
    const SciPhyAncestorPartials other_data =
        computeSubtreeAncestorPartials(other_side, root_like, ptn, memo);

    // t = length of the origin pseudo-branch.  convertToRooted() sets
    // this to 0 by default.  At t=0 the transProbVec kernel returns 1
    // only for identity transitions (state s -> state s) and 0 otherwise,
    // so the closure with the founder delta becomes:
    //     P(pattern) = partial_at_root_int[(0,...,0)]
    // which is the "SciPhy without origin partial" formulation matching
    // the -49.16 ground truth we computed for the 5-taxon demo.
    //
    // Once ModelSciPhy adds a free origin_time parameter, the origin
    // branch length will become (origin_time - tree_height) * clock_rate
    // and this same closure will produce the "SciPhy with origin partial"
    // value.  No code change needed here.
    const double t = std::max(0.0, origin_edge->length);
    double pattern_prob = 0.0;
    const double eps = 1e-300;

    // Founder-delta closure: sum over root_data's ancestors (a singleton),
    // and for each, transition through the origin pseudo-branch to every
    // possible state at root_int.
    for (size_t s_idx = 0; s_idx < root_data.ancestors.size(); ++s_idx) {
        if (root_data.partial_lh[s_idx] <= 0.0) continue;

        const std::vector<int> &s_vec = root_data.ancestors[s_idx];
        double branch_sum = 0.0;
        for (size_t x_idx = 0; x_idx < other_data.ancestors.size(); ++x_idx) {
            if (other_data.partial_lh[x_idx] <= 0.0) continue;
            const double tr = transProbVec(s_vec, other_data.ancestors[x_idx], t);
            if (tr > 0.0) {
                branch_sum += tr * other_data.partial_lh[x_idx];
            }
        }
        pattern_prob += root_data.partial_lh[s_idx] * branch_sum;
    }

    if (!(pattern_prob > 0.0) || !std::isfinite(pattern_prob)) {
        pattern_prob = eps;
    }
    return std::log(pattern_prob);
}

double PhyloTreeSciPhy::computeSciPhyLikelihood(double *pattern_lh, bool save_log_value) {
    (void)save_log_value;

    // The tree must already be rooted by the time we get here. Rooting is
    // performed once, up front, by ModelSciPhy's constructor -- which calls
    // ModelMarkov(..., /*reversible=*/false, /*adapt_tree=*/true), which in
    // turn calls phylo_tree->convertToRooted().
    //
    // We only need to keep the origin pseudo-branch length in sync with
    // (origin_time - tree_height) each call, since tree_height changes as
    // branches / NNI swaps modify the tree.
    syncOriginBranchLength();

    if (!_pattern_lh) {
        initializeAllPartialLh();
    }
    computePtnFreq();

    const size_t nptn = aln->getNPattern();
    double tree_lh = 0.0;
    for (size_t ptn = 0; ptn < nptn; ++ptn) {
        _pattern_lh[ptn] = computeSciPhyPatternLogLikelihood(ptn);
        tree_lh += _pattern_lh[ptn] * ptn_freq[ptn];
    }

    if (pattern_lh) {
        memmove(pattern_lh, _pattern_lh, nptn * sizeof(double));
    }

    // Full-precision debug print (opt-in). Set IQTREE_SCIPHY_PRINTLL=1 to emit
    // the tree log-likelihood at 15 significant digits on every evaluation --
    // the last line printed for a fixed-tree (-te -blfix) run is the final
    // value, for comparison against the SciPhy/BEAST reference to ~1e-12.
    if (std::getenv("IQTREE_SCIPHY_PRINTLL")) {
        std::cout << std::setprecision(15)
                  << "SciPhy tree logL = " << tree_lh << std::endl;
    }

    curScore = tree_lh;
    return tree_lh;
}

double PhyloTreeSciPhy::computeLikelihood(double *pattern_lh, bool save_log_value) {
    if (!hasBarcodeAlignment() || !useExperimentalSciPhyLikelihood()) {
        return IQTree::computeLikelihood(pattern_lh, save_log_value);
    }
    return computeSciPhyLikelihood(pattern_lh, save_log_value);
}

double PhyloTreeSciPhy::computeLikelihoodBranch(PhyloNeighbor *dad_branch,
                                                PhyloNode *dad,
                                                bool save_log_value) {
    if (!hasBarcodeAlignment() || !useExperimentalSciPhyLikelihood()) {
        return IQTree::computeLikelihoodBranch(dad_branch, dad, save_log_value);
    }

    // Phase 3: the LL combination edge is always the origin pseudo-branch
    // (see computeSciPhyPatternLogLikelihood), independent of whichever
    // branch Brent is currently optimizing.
    //
    // We deliberately do NOT install (dad_branch, dad) as current_it any
    // more.  That was the OLD behavior and it was the direct cause of
    // the "same tree, different LL" bouncing: every Brent call, every
    // NNI candidate evaluation, would silently reinstall a new
    // combination edge and produce a different LL for logically the
    // same tree state.
    //
    // Brent still uses current_it / current_it_back for its own
    // bookkeeping (updating branch lengths via computeFunction(x)), but
    // those pointers are installed by optimizeOneBranch, not by us.
    // The branch-length update reaches the LL through the tree
    // structure itself: transProbVec reads child_nei->length inside
    // computeSubtreeAncestorPartials, and that length was written by
    // Brent onto the neighbor object.  No shared mutable state needed.
    (void)dad_branch;
    (void)dad;
    return computeSciPhyLikelihood(nullptr, save_log_value);
}

double PhyloTreeSciPhy::computeLikelihoodFromBuffer() {
    // The base-class version reads a SIMD theta buffer that we never populate
    // in the SciPhy code path -- it asserts on `theta_all && theta_computed`
    // at phylokernelnew.h:3291 and crashes the moment NNI evaluation invokes
    // it.  Redirect through our own per-node ancestor-list recursion using
    // the active (current_it, current_it_back) rooting edge.  This is slower
    // than a cached re-evaluation would be, but correct.
    if (!hasBarcodeAlignment() || !useExperimentalSciPhyLikelihood()) {
        return IQTree::computeLikelihoodFromBuffer();
    }
    return computeSciPhyLikelihood(nullptr, true);
}

void PhyloTreeSciPhy::computePatternLikelihood(double *pattern_lh, double *cur_logl,
                                              double *pattern_lh_cat, SiteLoglType wsl) {
    if (!hasBarcodeAlignment() || !useExperimentalSciPhyLikelihood()) {
        IQTree::computePatternLikelihood(pattern_lh, cur_logl, pattern_lh_cat, wsl);
        return;
    }

    // computeSciPhyLikelihood fills _pattern_lh[ptn] with the per-pattern log
    // likelihood and (when pattern_lh != nullptr) memmoves it into pattern_lh.
    const double tree_lh = computeSciPhyLikelihood(pattern_lh, true);

    if (cur_logl) {
        *cur_logl = tree_lh;
    }

    // SciPhy has a single implicit rate category, so the per-category output
    // is just a copy of the per-pattern log likelihoods.
    if (pattern_lh_cat) {
        const size_t nptn = aln->getNPattern();
        const double *src = pattern_lh ? pattern_lh : _pattern_lh;
        if (src) {
            memmove(pattern_lh_cat, src, nptn * sizeof(double));
        }
    }
    (void)wsl;
}

void PhyloTreeSciPhy::optimizeOneBranch(PhyloNode *node1, PhyloNode *node2,
                                        bool clearLH, int maxNRStep) {
    if (!hasBarcodeAlignment() || !useExperimentalSciPhyLikelihood()) {
        IQTree::optimizeOneBranch(node1, node2, clearLH, maxNRStep);
        return;
    }

    // Skip the synthetic root edge in rooted-tree storage mode (matches
    // PhyloTree::optimizeOneBranch behavior).
    if (rooted && (node1 == root || node2 == root)) {
        return;
    }

    current_it      = (PhyloNeighbor*) node1->findNeighbor(node2);
    current_it_back = (PhyloNeighbor*) node2->findNeighbor(node1);
    ASSERT(current_it && current_it_back);

    const double current_len = current_it->length;
    double negative_lh = 0.0;
    double ferror = 0.0;

    const double initial_neg_lh =
        -computeLikelihoodBranch(current_it,
                                 (PhyloNode*) current_it_back->node);

    // Brent on this branch.
    double optx = minimizeOneDimen(params->min_branch_length,
                                   current_len,
                                   params->max_branch_length,
                                   params->min_branch_length,
                                   &negative_lh,
                                   &ferror);

    if (negative_lh > initial_neg_lh + 1e-8) {
        optx        = current_len;
        negative_lh = initial_neg_lh;
    }

    current_it->length      = optx;
    current_it_back->length = optx;

    if (clearLH && current_len != optx) {
        node1->clearReversePartialLh(node2);
        node2->clearReversePartialLh(node1);
    }

    (void)maxNRStep;  // unused: only matters for the Newton path we skipped
}

double PhyloTreeSciPhy::optimizeAllBranches(int my_iterations, double tolerance, int maxNRStep) {
    if (!hasBarcodeAlignment() || !useExperimentalSciPhyLikelihood()) {
        return IQTree::optimizeAllBranches(my_iterations, tolerance, maxNRStep);
    }
    NodeVector nodes, nodes2;
    getBranches(nodes, nodes2);
    ASSERT(nodes.size() == nodes2.size());

    double tree_lh = computeLikelihood(nullptr, true);
    DoubleVector saved_lengths;

    for (int iter = 0; iter < my_iterations; ++iter) {
        const double prev_lh = tree_lh;

        // Save branch lengths before the sweep so we can roll back exactly
        // if the sweep regresses.  Without this, reporting `tree_lh =
        // prev_lh` on regression is dishonest -- the tree state would still
        // hold the worse branches and the next caller (NNI's assertion at
        // iqtree.cpp:3291) would see the mismatch and abort.
        saveBranchLengths(saved_lengths);

        for (size_t j = 0; j < nodes.size(); ++j) {
            optimizeOneBranch((PhyloNode*) nodes[j],
                              (PhyloNode*) nodes2[j],
                              /*clearLH=*/true,
                              maxNRStep);
        }

        tree_lh = computeLikelihood(nullptr, true);

        if (verbose_mode >= VB_MED) {
            std::cout << "SciPhy brlen pass " << (iter + 1)
                      << ": logL = " << tree_lh << std::endl;
        }

        // Converged: per-sweep change smaller than tolerance.
        if (std::fabs(tree_lh - prev_lh) < tolerance) {
            break;
        }

        // Regression: with the per-branch monotonicity guard in
        // optimizeOneBranch this should be extremely rare, but if it
        // happens we restore the pre-sweep branch lengths so the reported
        // tree_lh matches the actual tree state.
        if (tree_lh < prev_lh - tolerance * 0.1) {
            if (verbose_mode >= VB_MED) {
                std::cout << "SciPhy brlen optimization regression: "
                          << prev_lh << " -> " << tree_lh
                          << "; restoring previous branch lengths"
                          << std::endl;
            }
            restoreBranchLengths(saved_lengths);
            tree_lh = prev_lh;
            break;
        }
    }

    curScore = tree_lh;
    return tree_lh;
}
