/*
 * phylotreesciphy.cpp
 *
 *  Created on: Jul 6, 2026
 *      Author: Maggie Lam
 */

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

    // Scar-type factor.  Use the model's edit_probs when available (populated from --edit-probs or defaulted to uniform 1/M by ModelFactory).  
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
        // Absorbing case: tape is now fully edited.  Probability is the tail mass P(>= nr_possible events) under Poisson(distance).
        double absorb_prob = 1.0;
        for (int i = 0; i < nr_possible; ++i) {
            absorb_prob -= poissonPMF(i, distance);
        }
        if (absorb_prob < 0.0) absorb_prob = 0.0;
        return absorb_prob * insert_factor;
    }
    return poissonPMF(k, distance) * insert_factor;
}

std::vector<std::vector<int>>
PhyloTreeSciPhy::getPossibleAncestors(const std::vector<int> &tape_vec) {
    // Walk the tape from right to left, clearing edits one at a time.  
    // Every partial sequence we visit is a valid ancestral state.
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

namespace {

// Vector intersection that preserves the order of `a` and removes any element not present in `b`. 
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

}  // anonymous namespace

// <----- Partial Likelihood caching implementation ----->


/**
 * Compute downward partial likelihoods, caching them in each PhyloNeighbor's sciphy_partial vector.
 * Recursive post-order traversal of the tree.
 */
void PhyloTreeSciPhy::computeSciPhyDownward(PhyloNeighbor *dad_branch, PhyloNode *dad) {
    // Cache hit: partial lh bit is 1, so it is valid
    if (dad_branch->partial_lh_computed & 1) {
        return;
    }

    PhyloNode *node = (PhyloNode*) dad_branch->node;
    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    const size_t nptn = aln->getNPattern();

    if (!dad_branch->sciphy_partial) {
        dad_branch->sciphy_partial = new std::vector<SciPhyAncestorPartials>();
    }
    std::vector<SciPhyAncestorPartials> &out = *dad_branch->sciphy_partial;
    out.assign(nptn, SciPhyAncestorPartials());

    if (!barcode_aln) {
        dad_branch->partial_lh_computed |= 1;
        return;
    }

    if (node->isLeaf()) {
        for (size_t ptn = 0; ptn < nptn; ++ptn) {
            SciPhyAncestorPartials &result = out[ptn];

            // observed tape on leaf node; all probability on the observed value (idx 0 of ancestors).
            const int obs = getObservedLeafState(node, ptn);
            if (obs < 0 || obs >= barcode_aln->getNumTapeStates()) {
                // defensive fallback for unknown/ambiguous leaf
                int site_count = 0;
                const auto &tsc = barcode_aln->getTapeSiteCounts();
                if (!tsc.empty()) site_count = tsc[0];
                result.ancestors.emplace_back((size_t)site_count, 0);
                result.partial_lh.push_back(1.0);
                continue;
            }
            const std::vector<int> &leaf_tape = barcode_aln->getTapeVec(obs);
            result.ancestors = getPossibleAncestors(leaf_tape);
            result.partial_lh.assign(result.ancestors.size(), 0.0);
            if (!result.partial_lh.empty()) {
                result.partial_lh[0] = 1.0;
            }
        }
        dad_branch->partial_lh_computed |= 1;
        return;
    }

    // Internal node: gather child neighbors (everything except the edge back
    // toward `dad`) and compute their (cached) downward partials first.
    std::vector<PhyloNeighbor*> child_nei;
    for (NeighborVec::iterator nit = node->neighbors.begin();
         nit != node->neighbors.end(); ++nit) {
        if ((*nit)->node != dad) {
            PhyloNeighbor *cb = (PhyloNeighbor*)(*nit);
            computeSciPhyDownward(cb, node);
            child_nei.push_back(cb);
        }
    }
    if (child_nei.empty()) {
        dad_branch->partial_lh_computed |= 1;
        return;
    }

    // Combine per pattern, parallelizable because tapes evolve independently.
    // (children are already computed serially above; each iteration writes a
    // distinct out[p] and only reads children's cached partials).
#pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1)
    for (int p = 0; p < (int)nptn; ++p) {
        SciPhyAncestorPartials &result = out[p];

        // ancestors = intersection of children's ancestor lists
        result.ancestors = (*child_nei[0]->sciphy_partial)[p].ancestors;
        for (size_t k = 1; k < child_nei.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, (*child_nei[k]->sciphy_partial)[p].ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0); // initialize vector of partial likelihoods size to number of ancestors, vals to 0

        // Felsentein's pruning algorithm without probability matrix
        // partial_lh[s] = prod_child sum_c transProbVec(s,c,t_c) * child_lh[c]
        for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx) {
            const std::vector<int> &s_vec = result.ancestors[s_idx]; // for each ancestor tape state s
            double lh_s = 1.0;
            for (size_t k = 0; k < child_nei.size(); ++k) {
                const SciPhyAncestorPartials &cd = (*child_nei[k]->sciphy_partial)[p];
                const double blen = child_nei[k]->length;
                double sum_child = 0.0;
                for (size_t c_idx = 0; c_idx < cd.ancestors.size(); ++c_idx) { // for each ancestor tape
                    if (cd.partial_lh[c_idx] <= 0.0) continue;
                    const double tr = transProbVec(s_vec, cd.ancestors[c_idx], blen);
                    if (tr > 0.0) sum_child += tr * cd.partial_lh[c_idx];
                }
                lh_s *= sum_child;
                if (lh_s <= 0.0) break;
            }
            result.partial_lh[s_idx] = lh_s;
        }

        // numerical scaling: log_scale = sum of children's, plus log(max) after
        // factoring the largest partial out so stored values stay O(1).
        double ls = 0.0;
        for (size_t k = 0; k < child_nei.size(); ++k) {
            ls += (*child_nei[k]->sciphy_partial)[p].log_scale;
        }
        double max_partial = 0.0;
        for (double v : result.partial_lh) if (v > max_partial) max_partial = v;
        if (max_partial > 0.0) {
            for (double &v : result.partial_lh) v /= max_partial;
            ls += std::log(max_partial);
        }
        result.log_scale = ls;
    }

    dad_branch->partial_lh_computed |= 1;
}

/**
 * Compute upward partial likelihoods, caching them in each PhyloNeighbor's sciphy_partial vector.
 * Recursive pre-order traversal of the tree.
 */
void PhyloTreeSciPhy::computeSciPhyUpward(PhyloNode *node, PhyloNode *dad) {
    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) return;
    const size_t nptn = aln->getNPattern();

    //   up_nei   = (node -> dad): holds UpToC(dad)   [function of dad's state]
    //   down_nei = (dad -> node): holds Down(node)   [function of node's state]
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) node->findNeighbor(dad);
    PhyloNeighbor *down_nei = (PhyloNeighbor*) dad->findNeighbor(node);
    if (!up_nei->sciphy_partial || !down_nei->sciphy_partial) return;
    const std::vector<SciPhyAncestorPartials> &upToDad = *up_nei->sciphy_partial;
    const std::vector<SciPhyAncestorPartials> &downNode = *down_nei->sciphy_partial;
    const double len_dad_node = down_nei->length;

    // UpBar(node)[s_node] = sum_{s_dad} UpToC(dad)[s_dad] * trans(s_dad -> s_node,
    //                       len(dad,node)), restricted to Down(node)'s support.
    std::vector<SciPhyAncestorPartials> upBar(nptn);
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyAncestorPartials &ud = upToDad[p];
        const SciPhyAncestorPartials &dn = downNode[p];
        SciPhyAncestorPartials &ub = upBar[p];
        ub.ancestors = dn.ancestors;                        // indexed by s_node
        ub.partial_lh.assign(ub.ancestors.size(), 0.0);
        for (size_t i = 0; i < ub.ancestors.size(); ++i) {
            const std::vector<int> &s_node = ub.ancestors[i];
            double acc = 0.0;
            for (size_t j = 0; j < ud.ancestors.size(); ++j) {
                if (ud.partial_lh[j] <= 0.0) continue;
                const double tr = transProbVec(ud.ancestors[j], s_node, len_dad_node);
                if (tr > 0.0) acc += ud.partial_lh[j] * tr;
            }
            ub.partial_lh[i] = acc;
        }
        double ls = ud.log_scale, mx = 0.0;
        for (double v : ub.partial_lh) if (v > mx) mx = v;
        if (mx > 0.0) { for (double &v : ub.partial_lh) v /= mx; ls += std::log(mx); }
        ub.log_scale = ls;
    }

    // children of node = all neighbors except dad
    std::vector<PhyloNeighbor*> childDown;
    std::vector<PhyloNode*> childNode;
    for (NeighborVec::iterator it = node->neighbors.begin(); it != node->neighbors.end(); ++it) {
        if ((*it)->node != dad) {
            childDown.push_back((PhyloNeighbor*)(*it));
            childNode.push_back((PhyloNode*)(*it)->node);
        }
    }

    // For each child c: UpToC(node)[s_node] = UpBar(node)[s_node] * prod over
    // siblings d (!= c) of ( sum_{s_d} trans(s_node -> s_d, len(node,d)) * Down(d)[s_d] );
    // store on (c -> node), then recurse into c.
    for (size_t ci = 0; ci < childDown.size(); ++ci) {
        PhyloNeighbor *c_up = (PhyloNeighbor*) childNode[ci]->findNeighbor(node);
        if (!c_up->sciphy_partial) c_up->sciphy_partial = new std::vector<SciPhyAncestorPartials>();
        std::vector<SciPhyAncestorPartials> &out = *c_up->sciphy_partial;
        out.assign(nptn, SciPhyAncestorPartials());
        for (size_t p = 0; p < nptn; ++p) {
            SciPhyAncestorPartials &result = out[p];
            result.ancestors  = upBar[p].ancestors;
            result.partial_lh = upBar[p].partial_lh;
            double ls = upBar[p].log_scale;
            for (size_t di = 0; di < childDown.size(); ++di) {
                if (di == ci) continue;
                const SciPhyAncestorPartials &dd = (*childDown[di]->sciphy_partial)[p];
                const double blen = childDown[di]->length;
                for (size_t i = 0; i < result.ancestors.size(); ++i) {
                    if (result.partial_lh[i] <= 0.0) continue;
                    const std::vector<int> &s_node = result.ancestors[i];
                    double sm = 0.0;
                    for (size_t j = 0; j < dd.ancestors.size(); ++j) {
                        if (dd.partial_lh[j] <= 0.0) continue;
                        const double tr = transProbVec(s_node, dd.ancestors[j], blen);
                        if (tr > 0.0) sm += tr * dd.partial_lh[j];
                    }
                    result.partial_lh[i] *= sm;
                }
                ls += dd.log_scale;
            }
            double mx = 0.0;
            for (double v : result.partial_lh) if (v > mx) mx = v;
            if (mx > 0.0) { for (double &v : result.partial_lh) v /= mx; ls += std::log(mx); }
            result.log_scale = ls;
        }
        c_up->partial_lh_computed |= 1;
        computeSciPhyUpward(childNode[ci], node);
    }
}

/**
 * Combine upward and downward partial likelihoods at a branch, returning the log-likelihood contribution of that branch.
 */
double PhyloTreeSciPhy::combineVectors(const std::vector<SciPhyAncestorPartials> &up,
                                      const std::vector<SciPhyAncestorPartials> &down,
                                      double len) const {
    const size_t nptn = aln->getNPattern();
    const double eps = 1e-300;
    if (up.size() != nptn || down.size() != nptn) {
        return -std::numeric_limits<double>::infinity();
    }
    double tree_lh = 0.0;
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyAncestorPartials &dc = down[p];  // s_child
        const SciPhyAncestorPartials &uc = up[p];    // s_parent
        double prob = 0.0;
        for (size_t j = 0; j < uc.ancestors.size(); ++j) {
            if (uc.partial_lh[j] <= 0.0) continue;
            const std::vector<int> &s_par = uc.ancestors[j];
            double s = 0.0;
            for (size_t i = 0; i < dc.ancestors.size(); ++i) {
                if (dc.partial_lh[i] <= 0.0) continue;
                const double tr = transProbVec(s_par, dc.ancestors[i], len);
                if (tr > 0.0) s += tr * dc.partial_lh[i];
            }
            prob += uc.partial_lh[j] * s;
        }
        double lp = (prob > 0.0 && std::isfinite(prob))
                        ? (std::log(prob) + dc.log_scale + uc.log_scale)
                        : std::log(eps);
        tree_lh += lp * ptn_freq[p];
    }
    return tree_lh;
}

/**
 * Combine upward and downward partial likelihoods at a branch, returning the log-likelihood contribution of that branch.
 */
double PhyloTreeSciPhy::combineAtBranch(PhyloNode *parent, PhyloNode *child) {
    PhyloNeighbor *down_nei = (PhyloNeighbor*) parent->findNeighbor(child); // Down(child)
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) child->findNeighbor(parent); // UpToC(parent)
    if (!down_nei->sciphy_partial || !up_nei->sciphy_partial) {
        return -std::numeric_limits<double>::infinity();
    }
    return combineVectors(*up_nei->sciphy_partial, *down_nei->sciphy_partial,
                          down_nei->length);
}

/**
 * Compute the log-likelihood of the entire tree.
 * 
 * @param pattern_lh Optional output array to store per-pattern log-likelihoods.
 * @param save_log_value Unused parameter for compatibility with base class.
 */
double PhyloTreeSciPhy::computeSciPhyLikelihood(double *pattern_lh, bool save_log_value) {
    (void)save_log_value;

    // Keep the origin pseudo-branch length in sync with (origin_time - tree_height) each call, 
    //      since tree_height changes as branches / NNI swaps modify the tree.
    syncOriginBranchLength();

    if (!_pattern_lh) {
        initializeAllPartialLh();
    }
    computePtnFreq();

    PhyloNeighbor *saved_current_it      = current_it;
    PhyloNeighbor *saved_current_it_back = current_it_back;
    clearAllPartialLH(); // TODO: we currently just clear it because we don't have a way to know which branches are dirty and need recomputation.  This is a potential optimization for the future.
    current_it      = saved_current_it;
    current_it_back = saved_current_it_back;

    const size_t nptn = aln->getNPattern();
    const double eps = 1e-300;
    double tree_lh = 0.0;

    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    const bool ok = rooted && root && root->name == ROOT_NAME &&
                    !root->neighbors.empty() && barcode_aln;
    if (!ok) {
        const double ninf = -std::numeric_limits<double>::infinity();
        for (size_t ptn = 0; ptn < nptn; ++ptn) _pattern_lh[ptn] = ninf;
        if (pattern_lh) memmove(pattern_lh, _pattern_lh, nptn * sizeof(double));
        curScore = ninf;
        return ninf;
    }

    // Compute (and cache) the downward partial of the whole biological subtree at root_int, over all patterns.
    PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];
    PhyloNode *root_node = (PhyloNode*) root;
    computeSciPhyDownward(origin_edge, root_node);
    const std::vector<SciPhyAncestorPartials> &bio = *origin_edge->sciphy_partial;

    // founder = all-zeros, transition through the origin branch (length t) to every possible state at root_int,
    // then add back the subtree's log_scale.
    const double t = std::max(0.0, origin_edge->length);
#pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1) reduction(+: tree_lh)
    for (int p = 0; p < (int)nptn; ++p) {
        const SciPhyAncestorPartials &b = bio[p];
        double pattern_prob = 0.0;
        if (!b.ancestors.empty()) {
            const std::vector<int> founder(b.ancestors[0].size(), 0);
            for (size_t x = 0; x < b.ancestors.size(); ++x) {
                if (b.partial_lh[x] <= 0.0) continue;
                const double tr = transProbVec(founder, b.ancestors[x], t);
                if (tr > 0.0) pattern_prob += tr * b.partial_lh[x];
            }
        }
        double lp;
        if (!(pattern_prob > 0.0) || !std::isfinite(pattern_prob)) {
            lp = std::log(eps);
        } else {
            lp = std::log(pattern_prob) + b.log_scale;
        }
        _pattern_lh[p] = lp;
        tree_lh += lp * ptn_freq[p];
    }

    if (pattern_lh) {
        memmove(pattern_lh, _pattern_lh, nptn * sizeof(double));
    }

    // Full-precision debug print (opt-in). Set IQTREE_SCIPHY_PRINTLL=1 to print log-likelihood at 15 sigfigs
    if (std::getenv("IQTREE_SCIPHY_PRINTLL")) {
        std::cout << std::setprecision(15)
                  << "SciPhy tree logL = " << tree_lh << std::endl;
    }

    curScore = tree_lh;

    // Downward partials were just cleared and rebuilt; any previously computed upward partials no longer match, so the up/down snapshot is now stale.
    // (getBestNNIForBran rebuilds the upward pass when it needs a snapshot.)
    sciphy_snapshot_valid = false;

    return tree_lh;
}

double PhyloTreeSciPhy::computeLikelihood(double *pattern_lh, bool save_log_value) {
    if (!hasBarcodeAlignment()) {
        return IQTree::computeLikelihood(pattern_lh, save_log_value);
    }
    return computeSciPhyLikelihood(pattern_lh, save_log_value);
}

double PhyloTreeSciPhy::computeLikelihoodBranch(PhyloNeighbor *dad_branch,
                                                PhyloNode *dad,
                                                bool save_log_value) {
    if (!hasBarcodeAlignment()) {
        return IQTree::computeLikelihoodBranch(dad_branch, dad, save_log_value);
    }

    (void)dad_branch;
    (void)dad;
    return computeSciPhyLikelihood(nullptr, save_log_value);
}

double PhyloTreeSciPhy::computeLikelihoodFromBuffer() {
    // we do not have a SIMD kernel, so we ignore the buffer and just call computeSciPhyLikelihood directly
    if (!hasBarcodeAlignment()) {
        return IQTree::computeLikelihoodFromBuffer();
    }
    return computeSciPhyLikelihood(nullptr, true);
}

void PhyloTreeSciPhy::computePatternLikelihood(double *pattern_lh, double *cur_logl,
                                              double *pattern_lh_cat, SiteLoglType wsl) {
    if (!hasBarcodeAlignment()) {
        IQTree::computePatternLikelihood(pattern_lh, cur_logl, pattern_lh_cat, wsl);
        return;
    }

    // computeSciPhyLikelihood fills _pattern_lh[ptn] with the per-pattern log likelihood and (when pattern_lh != nullptr) moves it into pattern_lh.
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
    if (!hasBarcodeAlignment()) {
        IQTree::optimizeOneBranch(node1, node2, clearLH, maxNRStep);
        return;
    }

    // Skip the synthetic root edge in rooted-tree storage mode
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


double PhyloTreeSciPhy::computeFunction(double value) {
    // the central branch length of a *swapped*
    // topology is optimized against two local vectors that were built without
    // mutating the tree. 
    if (sciphy_nni_active) {
        return -combineVectors(*sciphy_nni_up, *sciphy_nni_down, value);
    }
    if (!sciphy_combine_active) {
        return PhyloTree::computeFunction(value);
    }
    // Fast branch-length combine mode: set the branch length on both directed
    // neighbors and evaluate the whole-tree log-likelihood by combining at
    // (parent, child) using the cached snapshot partials.
    current_it->length = value;
    current_it_back->length = value;
    return -combineAtBranch(sciphy_combine_parent, sciphy_combine_child);
}

void PhyloTreeSciPhy::computeSciPhyFullUpward() {
    if (!rooted || !root || root->neighbors.empty()) return;
    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) return;
    const size_t nptn = aln->getNPattern();

    PhyloNode *root_node = (PhyloNode*) root;
    PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];   // Down(root_int)
    PhyloNode *root_int = (PhyloNode*) origin_edge->node;
    if (!origin_edge->sciphy_partial) return;                          // downward must exist
    const std::vector<SciPhyAncestorPartials> &downRoot = *origin_edge->sciphy_partial;

    // Seed UpToC(founder) on (root_int -> root): founder delta = all-zeros, mass 1.
    PhyloNeighbor *up0 = (PhyloNeighbor*) root_int->findNeighbor(root_node);
    if (!up0->sciphy_partial) up0->sciphy_partial = new std::vector<SciPhyAncestorPartials>();
    std::vector<SciPhyAncestorPartials> &seed = *up0->sciphy_partial;
    seed.assign(nptn, SciPhyAncestorPartials());
    for (size_t p = 0; p < nptn; ++p) {
        const size_t L = downRoot[p].ancestors.empty() ? 0 : downRoot[p].ancestors[0].size();
        seed[p].ancestors.emplace_back(L, 0);
        seed[p].partial_lh.push_back(1.0);
        seed[p].log_scale = 0.0;
    }
    computeSciPhyUpward(root_int, root_node);

    // Downward (from the caller's computeSciPhyLikelihood) + upward (just filled)
    // now form a consistent snapshot of the current tree.
    sciphy_snapshot_valid = true;
}

void PhyloTreeSciPhy::optimizeOneBranchCombine(PhyloNode *node1, PhyloNode *node2) {
    if (rooted && (node1 == root || node2 == root)) return;   // skip origin edge

    current_it      = (PhyloNeighbor*) node1->findNeighbor(node2);
    current_it_back = (PhyloNeighbor*) node2->findNeighbor(node1);
    if (!current_it || !current_it_back) return;

    // parent = the endpoint toward the root.
    if (current_it->direction == TOWARD_ROOT) {
        sciphy_combine_parent = node2; sciphy_combine_child = node1;
    } else {
        sciphy_combine_parent = node1; sciphy_combine_child = node2;
    }

    // Both snapshot partials must be present; otherwise leave this branch as-is.
    PhyloNeighbor *down_nei = (PhyloNeighbor*) sciphy_combine_parent->findNeighbor(sciphy_combine_child);
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) sciphy_combine_child->findNeighbor(sciphy_combine_parent);
    if (!down_nei->sciphy_partial || !up_nei->sciphy_partial) return;

    const double current_len = current_it->length;
    double negative_lh = 0.0, ferror = 0.0;

    sciphy_combine_active = true;
    const double initial_neg_lh = -combineAtBranch(sciphy_combine_parent, sciphy_combine_child);
    double optx = minimizeOneDimen(params->min_branch_length, current_len,
                                   params->max_branch_length, params->min_branch_length,
                                   &negative_lh, &ferror);
    sciphy_combine_active = false;

    if (negative_lh > initial_neg_lh + 1e-8) optx = current_len;   // keep if no improvement
    current_it->length      = optx;
    current_it_back->length = optx;
    // No invalidation here: the whole sweep runs against one snapshot (Jacobi);
    // optimizeAllBranches recomputes the exact likelihood afterwards.
}

double PhyloTreeSciPhy::optimizeAllBranches(int my_iterations, double tolerance, int maxNRStep) {
    if (!hasBarcodeAlignment()) {
        return IQTree::optimizeAllBranches(my_iterations, tolerance, maxNRStep);
    }
    (void)maxNRStep;

    // Branch directions (parent vs child) are needed to orient each combine.
    if (rooted) computeBranchDirection();

    NodeVector nodes, nodes2;
    getBranches(nodes, nodes2);
    ASSERT(nodes.size() == nodes2.size());

    double tree_lh = computeLikelihood(nullptr, true);   // exact
    DoubleVector saved_lengths;

    for (int iter = 0; iter < my_iterations; ++iter) {
        const double prev_lh = tree_lh;
        saveBranchLengths(saved_lengths);

        // Build one consistent up/down snapshot for this sweep. 
        computeSciPhyLikelihood(nullptr, true);
        computeSciPhyFullUpward();

        for (size_t j = 0; j < nodes.size(); ++j) {
            optimizeOneBranchCombine((PhyloNode*) nodes[j], (PhyloNode*) nodes2[j]);
        }

        // Exact likelihood with the new branch lengths (full recompute).
        tree_lh = computeLikelihood(nullptr, true);

        if (verbose_mode >= VB_MED) {
            std::cout << "SciPhy brlen pass " << (iter + 1)
                      << ": logL = " << tree_lh << std::endl;
        }

        if (std::fabs(tree_lh - prev_lh) < tolerance) {
            break;   // converged
        }
        if (tree_lh < prev_lh - tolerance * 0.1) {
            if (verbose_mode >= VB_MED) {
                std::cout << "SciPhy brlen optimization regression: "
                          << prev_lh << " -> " << tree_lh
                          << "; restoring previous branch lengths" << std::endl;
            }
            restoreBranchLengths(saved_lengths);
            tree_lh = prev_lh;
            break;
        }
    }

    curScore = tree_lh;
    return tree_lh;
}

double PhyloTreeSciPhy::optimizeRootPosition(int root_dist, bool write_info,
                                             double logl_epsilon) {
    if (!hasBarcodeAlignment()) {
        return IQTree::optimizeRootPosition(root_dist, write_info, logl_epsilon);
    }
    // no-op, root is fixed
    (void)root_dist;
    (void)write_info;
    (void)logl_epsilon;
    curScore = computeLikelihood();
    return curScore;
}

void PhyloTreeSciPhy::ensureSciPhySnapshot() {
    if (sciphy_snapshot_valid) return;
    if (rooted) computeBranchDirection();
    computeSciPhyLikelihood(nullptr, true);
    computeSciPhyFullUpward();
}

void PhyloTreeSciPhy::buildDownwardLocal(PhyloNode *child, PhyloNode *parent,
                                         std::vector<SciPhyAncestorPartials> &out) const {
    const size_t nptn = aln->getNPattern();
    out.assign(nptn, SciPhyAncestorPartials());

    // child's children = all neighbors except the edge back to `parent`.
    std::vector<PhyloNeighbor*> kids;
    for (NeighborVec::iterator nit = child->neighbors.begin();
         nit != child->neighbors.end(); ++nit) {
        if ((*nit)->node != parent) kids.push_back((PhyloNeighbor*)(*nit));
    }
    if (kids.empty()) { out.clear(); return; }
    for (PhyloNeighbor *k : kids)
        if (!k->sciphy_partial) { out.clear(); return; }   // cache miss => bail

    for (size_t p = 0; p < nptn; ++p) {
        SciPhyAncestorPartials &result = out[p];
        result.ancestors = (*kids[0]->sciphy_partial)[p].ancestors;
        for (size_t k = 1; k < kids.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, (*kids[k]->sciphy_partial)[p].ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0);
        for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx) {
            const std::vector<int> &s_vec = result.ancestors[s_idx];
            double lh_s = 1.0;
            for (size_t k = 0; k < kids.size(); ++k) {
                const SciPhyAncestorPartials &cd = (*kids[k]->sciphy_partial)[p];
                const double blen = kids[k]->length;
                double sum_child = 0.0;
                for (size_t c_idx = 0; c_idx < cd.ancestors.size(); ++c_idx) {
                    if (cd.partial_lh[c_idx] <= 0.0) continue;
                    const double tr = transProbVec(s_vec, cd.ancestors[c_idx], blen);
                    if (tr > 0.0) sum_child += tr * cd.partial_lh[c_idx];
                }
                lh_s *= sum_child;
                if (lh_s <= 0.0) break;
            }
            result.partial_lh[s_idx] = lh_s;
        }
        double ls = 0.0;
        for (size_t k = 0; k < kids.size(); ++k)
            ls += (*kids[k]->sciphy_partial)[p].log_scale;
        double mx = 0.0;
        for (double v : result.partial_lh) if (v > mx) mx = v;
        if (mx > 0.0) { for (double &v : result.partial_lh) v /= mx; ls += std::log(mx); }
        result.log_scale = ls;
    }
}

void PhyloTreeSciPhy::buildUpToCLocal(PhyloNode *parent, PhyloNode *child,
                                      std::vector<SciPhyAncestorPartials> &out) const {
    const size_t nptn = aln->getNPattern();
    out.assign(nptn, SciPhyAncestorPartials());

    // parent's toward-root neighbor holds UpToC(P) (P = parent's parent, or the
    // founder seed when parent == root_int). Its length is len(P, parent).
    PhyloNeighbor *pnei = nullptr;
    for (NeighborVec::iterator nit = parent->neighbors.begin();
         nit != parent->neighbors.end(); ++nit) {
        if (((PhyloNeighbor*)(*nit))->direction == TOWARD_ROOT) {
            pnei = (PhyloNeighbor*)(*nit);
            break;
        }
    }
    if (!pnei || !pnei->sciphy_partial) { out.clear(); return; }
    const std::vector<SciPhyAncestorPartials> &upP = *pnei->sciphy_partial;
    const double len_Pp = pnei->length;

    // parent's children = all neighbors except the toward-root one; one of them
    // is `child` (the target edge), the rest are siblings.
    std::vector<PhyloNeighbor*> kids;
    PhyloNeighbor *child_nei = nullptr;
    for (NeighborVec::iterator nit = parent->neighbors.begin();
         nit != parent->neighbors.end(); ++nit) {
        if ((PhyloNeighbor*)(*nit) == pnei) continue;
        PhyloNeighbor *k = (PhyloNeighbor*)(*nit);
        kids.push_back(k);
        if (k->node == child) child_nei = k;
    }
    if (kids.empty() || !child_nei) { out.clear(); return; }
    for (PhyloNeighbor *k : kids)
        if (!k->sciphy_partial) { out.clear(); return; }

    for (size_t p = 0; p < nptn; ++p) {
        SciPhyAncestorPartials &result = out[p];
        result.ancestors = (*kids[0]->sciphy_partial)[p].ancestors;
        for (size_t k = 1; k < kids.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, (*kids[k]->sciphy_partial)[p].ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0);

        // UpBar(parent)[s] = sum_{sP} UpToC(P)[sP] * trans(sP -> s, len_Pp)
        const SciPhyAncestorPartials &uP = upP[p];
        double ls = uP.log_scale;
        for (size_t i = 0; i < result.ancestors.size(); ++i) {
            const std::vector<int> &s = result.ancestors[i];
            double acc = 0.0;
            for (size_t j = 0; j < uP.ancestors.size(); ++j) {
                if (uP.partial_lh[j] <= 0.0) continue;
                const double tr = transProbVec(uP.ancestors[j], s, len_Pp);
                if (tr > 0.0) acc += uP.partial_lh[j] * tr;
            }
            result.partial_lh[i] = acc;
        }

        // multiply by every sibling's down contribution (children except `child`)
        for (size_t k = 0; k < kids.size(); ++k) {
            if (kids[k] == child_nei) continue;
            const SciPhyAncestorPartials &dd = (*kids[k]->sciphy_partial)[p];
            const double blen = kids[k]->length;
            for (size_t i = 0; i < result.ancestors.size(); ++i) {
                if (result.partial_lh[i] <= 0.0) continue;
                const std::vector<int> &s = result.ancestors[i];
                double sm = 0.0;
                for (size_t jj = 0; jj < dd.ancestors.size(); ++jj) {
                    if (dd.partial_lh[jj] <= 0.0) continue;
                    const double tr = transProbVec(s, dd.ancestors[jj], blen);
                    if (tr > 0.0) sm += tr * dd.partial_lh[jj];
                }
                result.partial_lh[i] *= sm;
            }
            ls += dd.log_scale;
        }
        double mx = 0.0;
        for (double v : result.partial_lh) if (v > mx) mx = v;
        if (mx > 0.0) { for (double &v : result.partial_lh) v /= mx; ls += std::log(mx); }
        result.log_scale = ls;
    }
}

void PhyloTreeSciPhy::doNNI(NNIMove &move, bool clearLH) {
    PhyloTree::doNNI(move, clearLH);
    // Topology changed: the cached up/down snapshot no longer matches the tree.
    sciphy_snapshot_valid = false;
}

void PhyloTreeSciPhy::changeNNIBrans(NNIMove &nnimove) {
    if (!hasBarcodeAlignment()) {
        PhyloTree::changeNNIBrans(nnimove);
        return;
    }
    // Apply only the central branch length we optimized during scoring.
    if (nnimove.newLen[0].empty()) return;
    PhyloNeighbor *n12 = (PhyloNeighbor*) nnimove.node1->findNeighbor(nnimove.node2);
    PhyloNeighbor *n21 = (PhyloNeighbor*) nnimove.node2->findNeighbor(nnimove.node1);
    n12->setLength(nnimove.newLen[0]);
    n21->setLength(nnimove.newLen[0]);
}

NNIMove PhyloTreeSciPhy::getBestNNIForBran(PhyloNode *node1, PhyloNode *node2,
                                           NNIMove *nniMoves) {
    if (!hasBarcodeAlignment()) {
        return IQTree::getBestNNIForBran(node1, node2, nniMoves);
    }
    if (nniMoves && (nniMoves[0].ptnlh || nniMoves[1].ptnlh)) {
        return IQTree::getBestNNIForBran(node1, node2, nniMoves);
    }

    ASSERT(!node1->isLeaf() && !node2->isLeaf());
    ASSERT(node1->degree() == 3 && node2->degree() == 3);

    // One valid snapshot for the whole evaluateNNIs pass (rebuilt only if stale).
    ensureSciPhySnapshot();

    // Orient so node1 is the parent (node1 -> node2 points away from the root).
    if (((PhyloNeighbor*)node1->findNeighbor(node2))->direction == TOWARD_ROOT) {
        PhyloNode *tmp = node1; node1 = node2; node2 = tmp;
    }

    const double NEG = -std::numeric_limits<double>::max();

    bool newNNIMoves = false;
    if (nniMoves == nullptr) {
        newNNIMoves = true;
        nniMoves = new NNIMove[2];
        nniMoves[0].ptnlh = nniMoves[1].ptnlh = nullptr;
        nniMoves[0].node1 = nullptr;
    }

    // Set up the two candidate swaps exactly as the base does: node1Nei is
    // node1's non-parent child; node2Nei ranges over node2's two children.
    if (!nniMoves[0].node1) {
        int cnt = 0;
        FOR_NEIGHBOR_IT(node1, node2, node1_it)
            if (((PhyloNeighbor*)*node1_it)->direction != TOWARD_ROOT) {
                cnt = 0;
                FOR_NEIGHBOR_IT(node2, node1, node2_it) {
                    nniMoves[cnt].node1Nei_it = node1_it;
                    nniMoves[cnt].node2Nei_it = node2_it;
                    cnt++;
                }
                break;
            }
        ASSERT(cnt == 2);
    }
    nniMoves[0].node1 = nniMoves[1].node1 = node1;
    nniMoves[0].node2 = nniMoves[1].node2 = node2;
    nniMoves[0].newloglh = nniMoves[1].newloglh = NEG;

    const double backupScore = curScore;
    PhyloNeighbor *central      = (PhyloNeighbor*) node1->findNeighbor(node2);
    PhyloNeighbor *central_back = (PhyloNeighbor*) node2->findNeighbor(node1);
    const double orig_central_len = central->length;
    const size_t nptn = aln->getNPattern();
    double optlen_arr[2] = {orig_central_len, orig_central_len};

    auto rawSwap = [&](int cnt) {
        NeighborVec::iterator n1it = nniMoves[cnt].node1Nei_it;
        NeighborVec::iterator n2it = nniMoves[cnt].node2Nei_it;
        Neighbor *n1nei = *n1it, *n2nei = *n2it;
        node1->updateNeighbor(n1it, n2nei);
        n2nei->node->updateNeighbor(node2, node1);
        node2->updateNeighbor(n2it, n1nei);
        n1nei->node->updateNeighbor(node1, node2);
    };
    auto undoSwap = [&](int cnt) {
        NeighborVec::iterator n1it = nniMoves[cnt].node1Nei_it;
        NeighborVec::iterator n2it = nniMoves[cnt].node2Nei_it;
        // after rawSwap, *n1it holds node2's old child, *n2it holds node1's old
        // child; restore each to its original neighbor object.
        Neighbor *cur_n1 = *n1it, *cur_n2 = *n2it;
        node1->updateNeighbor(n1it, cur_n2);
        cur_n2->node->updateNeighbor(node2, node1);
        node2->updateNeighbor(n2it, cur_n1);
        cur_n1->node->updateNeighbor(node1, node2);
    };

    // PASS 1: fast local-combine optimization 
    for (int cnt = 0; cnt < 2; cnt++) if (constraintTree.isCompatible(nniMoves[cnt])) {
        rawSwap(cnt);
        std::vector<SciPhyAncestorPartials> downChild, upParent;
        buildDownwardLocal(node2, node1, downChild);
        buildUpToCLocal(node1, node2, upParent);
        if (downChild.size() == nptn && upParent.size() == nptn) {
            sciphy_nni_up = &upParent;
            sciphy_nni_down = &downChild;
            sciphy_nni_active = true;
            double neg = 0.0, ferr = 0.0;
            const double initial_neg = -combineVectors(upParent, downChild, orig_central_len);
            double optlen = minimizeOneDimen(params->min_branch_length, orig_central_len,
                                             params->max_branch_length, params->min_branch_length,
                                             &neg, &ferr);
            sciphy_nni_active = false;
            sciphy_nni_up = sciphy_nni_down = nullptr;
            if (neg > initial_neg + 1e-8) optlen = orig_central_len;
            optlen_arr[cnt] = optlen;
        }
        undoSwap(cnt);
    }

    // PASS 2: EXACT likelihood of each swapped topology
    for (int cnt = 0; cnt < 2; cnt++) if (constraintTree.isCompatible(nniMoves[cnt])) {
        rawSwap(cnt);
        central->length      = optlen_arr[cnt];
        central_back->length = optlen_arr[cnt];
        const double exact = computeLikelihood();
        nniMoves[cnt].newloglh = exact;
        nniMoves[cnt].newLen[0].assign(1, optlen_arr[cnt]);
        central->length      = orig_central_len;
        central_back->length = orig_central_len;
        undoSwap(cnt);
    }

    // Restore the original snapshot + score for the next branch in this pass.
    ensureSciPhySnapshot();
    curScore = backupScore;

    NNIMove res = (nniMoves[0].newloglh > nniMoves[1].newloglh) ? nniMoves[0] : nniMoves[1];
    if (newNNIMoves) delete[] nniMoves;
    return res;
}
