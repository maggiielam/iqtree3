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

// log Poisson PMF; stays finite where poissonPMF would underflow to 0.
double logPoissonPMF(int k, double mean) {
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    if (k < 0 || mean < 0.0) return NEG_INF;
    if (mean == 0.0) return (k == 0) ? 0.0 : NEG_INF;
    return (double)k * std::log(mean) - mean - std::lgamma((double)k + 1.0);
}

// Numerically stable log(exp(a) + exp(b)).
double logAdd(double a, double b) {
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    const double hi = (a > b) ? a : b;
    const double lo = (a > b) ? b : a;
    return hi + std::log1p(std::exp(lo - hi));
}
}  // anonymous namespace

// log P(parent tape -> child tape) over branch_length; -inf if impossible.
double PhyloTreeSciPhy::logTransProbVec(const std::vector<int> &parent,
                                        const std::vector<int> &child,
                                        double branch_length) const {
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    const ModelSciPhy *sm = dynamic_cast<const ModelSciPhy*>(model);
    if (!sm) return NEG_INF;
    const double lambda = sm->getClockRate();
    const auto &edit_probs = sm->getEditProbs();

    const BarcodeAlignment *barcode_aln = dynamic_cast<const BarcodeAlignment*>(aln);
    if (!barcode_aln) return NEG_INF;
    if (parent.size() != child.size() || parent.empty()) return NEG_INF;

    const int L = (int)parent.size();
    const int M = std::max(1, barcode_aln->getMaxInsertionType());

    std::vector<int> parent_scars, child_scars;
    parent_scars.reserve((size_t)L);
    child_scars.reserve((size_t)L);
    for (int x : parent) if (x != 0) parent_scars.push_back(x);
    for (int x : child)  if (x != 0) child_scars.push_back(x);

    if (parent_scars.size() > child_scars.size()) return NEG_INF;

    std::vector<int> new_inserts = child_scars;
    for (int p_scar : parent_scars) {
        auto it = std::find(new_inserts.begin(), new_inserts.end(), p_scar);
        if (it == new_inserts.end()) return NEG_INF;
        new_inserts.erase(it);
    }

    const int nr_possible = L - (int)parent_scars.size();
    const int k = (int)new_inserts.size();
    if (k > nr_possible) return NEG_INF;

    // Scar-type factor, in log space.
    const double per_scar_uniform = 1.0 / (double)M;
    const int n_edit_probs = (int)edit_probs.size();
    double log_insert = 0.0;
    for (int scar : new_inserts) {
        if (scar < 1 || scar > M) return NEG_INF;
        const double pe = (scar - 1 < n_edit_probs) ? edit_probs[scar - 1] : per_scar_uniform;
        if (!(pe > 0.0)) return NEG_INF;
        log_insert += std::log(pe);
    }

    const double distance = std::max(0.0, lambda * branch_length);

    if (k == nr_possible) {
        // Absorbing tail P(K >= nr_possible) under Poisson(distance).
        double tail = 1.0;
        for (int i = 0; i < nr_possible; ++i) tail -= poissonPMF(i, distance);
        if (tail > 0.0) return std::log(tail) + log_insert;
        // tail underflowed: use its leading term P(K = nr_possible)
        return logPoissonPMF(nr_possible, distance) + log_insert;
    }
    return logPoissonPMF(k, distance) + log_insert;
}

std::vector<std::vector<int>>
PhyloTreeSciPhy::getPossibleAncestors(const std::vector<int> &tape_vec) {
    // clear edits right to left; every prefix visited is a possible ancestor
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

// elements of `a` also in `b`, in the order of `a`
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

// log(sum_i partial_lh[i]) + log_scale (just log_scale for a wildcard partial)
double totalLogMass(const SciPhyAncestorPartials &v) {
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    if (v.wildcard) return v.log_scale;
    double s = 0.0;
    for (double x : v.partial_lh) if (x > 0.0) s += x;
    return (s > 0.0) ? (std::log(s) + v.log_scale) : NEG_INF;
}

}  // anonymous namespace

// <----- Partial likelihood caching ----->

// Post-order: cache Down(subtree) on dad_branch->sciphy_partial.
void PhyloTreeSciPhy::computeSciPhyDownward(PhyloNeighbor *dad_branch, PhyloNode *dad) {
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
        // Dropout tip weights (constants): dropped -> q, observed -> 1-q; 1 if q unset.
        const ModelSciPhy *sm_leaf = dynamic_cast<const ModelSciPhy*>(model);
        const bool use_q = sm_leaf && sm_leaf->hasDropoutProb();
        const double log_q     = use_q ? std::log(sm_leaf->getDropoutProb()) : 0.0;
        const double log_1mq   = use_q ? std::log(1.0 - sm_leaf->getDropoutProb()) : 0.0;

        for (size_t ptn = 0; ptn < nptn; ++ptn) {
            SciPhyAncestorPartials &result = out[ptn];

            const int obs = getObservedLeafState(node, ptn);
            if (obs == aln->STATE_UNKNOWN || obs < 0 ||
                obs >= barcode_aln->getNumTapeStates()) {
                // dropped tape: wildcard (present, identity unknown)
                result.wildcard = true;
                result.ancestors.clear();
                result.partial_lh.clear();
                result.log_scale = log_q;
                continue;
            }
            const std::vector<int> &leaf_tape = barcode_aln->getTapeVec(obs);
            result.ancestors = getPossibleAncestors(leaf_tape);
            result.partial_lh.assign(result.ancestors.size(), 0.0);
            if (!result.partial_lh.empty()) {
                result.partial_lh[0] = 1.0;
            }
            result.log_scale = log_1mq;
        }
        dad_branch->partial_lh_computed |= 1;
        return;
    }

    // internal node: compute children first
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

    // patterns (tapes) are independent
#pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1)
    for (int p = 0; p < (int)nptn; ++p) {
        SciPhyAncestorPartials &result = out[p];

        // wildcard children don't constrain the ancestor set; they add only a constant
        std::vector<PhyloNeighbor*> obs_nei;
        double wild_scale = 0.0;
        for (size_t k = 0; k < child_nei.size(); ++k) {
            const SciPhyAncestorPartials &cp = (*child_nei[k]->sciphy_partial)[p];
            if (cp.wildcard) wild_scale += cp.log_scale;
            else             obs_nei.push_back(child_nei[k]);
        }

        // all children missing this tape -> wildcard
        if (obs_nei.empty()) {
            result.wildcard = true;
            result.ancestors.clear();
            result.partial_lh.clear();
            result.log_scale = wild_scale;
            continue;
        }

        // possible states = intersection of the observed children's ancestor lists
        result.ancestors = (*obs_nei[0]->sciphy_partial)[p].ancestors;
        for (size_t k = 1; k < obs_nei.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, (*obs_nei[k]->sciphy_partial)[p].ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0);

        // Pruning in log space:
        //   log partial_lh[s] = sum_child logsumexp_c( logTrans(s -> c, t_c) + log child_lh[c] )
        // stored relative to its max; the rest goes into log_scale.
        const double NEG_INF = -std::numeric_limits<double>::infinity();
        double base_scale = wild_scale;
        for (size_t k = 0; k < obs_nei.size(); ++k)
            base_scale += (*obs_nei[k]->sciphy_partial)[p].log_scale;

        std::vector<std::vector<double>> log_cp(obs_nei.size());
        for (size_t k = 0; k < obs_nei.size(); ++k) {
            const SciPhyAncestorPartials &cd = (*obs_nei[k]->sciphy_partial)[p];
            log_cp[k].resize(cd.ancestors.size());
            for (size_t c = 0; c < cd.ancestors.size(); ++c)
                log_cp[k][c] = (cd.partial_lh[c] > 0.0) ? std::log(cd.partial_lh[c]) : NEG_INF;
        }

        std::vector<double> log_rel(result.ancestors.size(), NEG_INF);
        double max_log_rel = NEG_INF;
        for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx) {
            const std::vector<int> &s_vec = result.ancestors[s_idx];
            double log_lh_s = 0.0;
            for (size_t k = 0; k < obs_nei.size(); ++k) {
                const SciPhyAncestorPartials &cd = (*obs_nei[k]->sciphy_partial)[p];
                const double blen = obs_nei[k]->length;
                double child_lse = NEG_INF;
                for (size_t c_idx = 0; c_idx < cd.ancestors.size(); ++c_idx) {
                    if (log_cp[k][c_idx] == NEG_INF) continue;
                    const double lt = logTransProbVec(s_vec, cd.ancestors[c_idx], blen);
                    if (lt == NEG_INF) continue;
                    child_lse = logAdd(child_lse, lt + log_cp[k][c_idx]);
                }
                log_lh_s += child_lse;
                if (log_lh_s == NEG_INF) break;
            }
            log_rel[s_idx] = log_lh_s;
            if (log_lh_s > max_log_rel) max_log_rel = log_lh_s;
        }

        if (max_log_rel == NEG_INF) {
            result.log_scale = NEG_INF;   // no possible ancestral state
        } else {
            for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx)
                result.partial_lh[s_idx] = std::exp(log_rel[s_idx] - max_log_rel);
            result.log_scale = base_scale + max_log_rel;
        }
    }

    dad_branch->partial_lh_computed |= 1;
}

// Pre-order: cache UpToC(node) on each (child -> node) half-edge below `node`.
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

    // UpBar(node)[s] = sum_{s_dad} UpToC(dad)[s_dad] * trans(s_dad -> s), s in Down(node)'s support
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<SciPhyAncestorPartials> upBar(nptn);
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyAncestorPartials &ud = upToDad[p];
        const SciPhyAncestorPartials &dn = downNode[p];
        SciPhyAncestorPartials &ub = upBar[p];

        // all-missing subtree or uninformative up side: collapses to the total up mass
        if (dn.wildcard || ud.wildcard) {
            ub.wildcard = true;
            ub.ancestors.clear();
            ub.partial_lh.clear();
            ub.log_scale = totalLogMass(ud);
            continue;
        }

        ub.ancestors = dn.ancestors;                        // indexed by s_node
        ub.partial_lh.assign(ub.ancestors.size(), 0.0);

        std::vector<double> log_ud(ud.ancestors.size(), NEG_INF);
        for (size_t j = 0; j < ud.ancestors.size(); ++j)
            log_ud[j] = (ud.partial_lh[j] > 0.0) ? std::log(ud.partial_lh[j]) : NEG_INF;

        std::vector<double> log_rel(ub.ancestors.size(), NEG_INF);
        double max_log_rel = NEG_INF;
        for (size_t i = 0; i < ub.ancestors.size(); ++i) {
            const std::vector<int> &s_node = ub.ancestors[i];
            double lse = NEG_INF;
            for (size_t j = 0; j < ud.ancestors.size(); ++j) {
                if (log_ud[j] == NEG_INF) continue;
                const double lt = logTransProbVec(ud.ancestors[j], s_node, len_dad_node);
                if (lt == NEG_INF) continue;
                lse = logAdd(lse, log_ud[j] + lt);
            }
            log_rel[i] = lse;
            if (lse > max_log_rel) max_log_rel = lse;
        }
        if (max_log_rel == NEG_INF) {
            ub.log_scale = NEG_INF;
        } else {
            for (size_t i = 0; i < ub.ancestors.size(); ++i)
                ub.partial_lh[i] = std::exp(log_rel[i] - max_log_rel);
            ub.log_scale = ud.log_scale + max_log_rel;
        }
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

    // UpToC(node) for child c = UpBar(node) * prod_{siblings d} sum_{s_d} trans * Down(d);
    // stored on (c -> node), then recurse into c.
    for (size_t ci = 0; ci < childDown.size(); ++ci) {
        PhyloNeighbor *c_up = (PhyloNeighbor*) childNode[ci]->findNeighbor(node);
        if (!c_up->sciphy_partial) c_up->sciphy_partial = new std::vector<SciPhyAncestorPartials>();
        std::vector<SciPhyAncestorPartials> &out = *c_up->sciphy_partial;
        out.assign(nptn, SciPhyAncestorPartials());
        for (size_t p = 0; p < nptn; ++p) {
            SciPhyAncestorPartials &result = out[p];

            // constant UpBar -> constant UpToC (fold in sibling constants)
            if (upBar[p].wildcard) {
                double ls = upBar[p].log_scale;
                for (size_t di = 0; di < childDown.size(); ++di) {
                    if (di == ci) continue;
                    ls += (*childDown[di]->sciphy_partial)[p].log_scale;
                }
                result.wildcard = true;
                result.ancestors.clear();
                result.partial_lh.clear();
                result.log_scale = ls;
                continue;
            }

            result.ancestors = upBar[p].ancestors;
            result.partial_lh.assign(result.ancestors.size(), 0.0);

            double base_scale = upBar[p].log_scale;
            std::vector<double> log_rel(result.ancestors.size(), NEG_INF);
            for (size_t i = 0; i < result.ancestors.size(); ++i)
                log_rel[i] = (upBar[p].partial_lh[i] > 0.0) ? std::log(upBar[p].partial_lh[i]) : NEG_INF;

            for (size_t di = 0; di < childDown.size(); ++di) {
                if (di == ci) continue;
                const SciPhyAncestorPartials &dd = (*childDown[di]->sciphy_partial)[p];
                if (dd.wildcard) { base_scale += dd.log_scale; continue; }
                const double blen = childDown[di]->length;
                std::vector<double> log_dd(dd.ancestors.size(), NEG_INF);
                for (size_t j = 0; j < dd.ancestors.size(); ++j)
                    log_dd[j] = (dd.partial_lh[j] > 0.0) ? std::log(dd.partial_lh[j]) : NEG_INF;
                for (size_t i = 0; i < result.ancestors.size(); ++i) {
                    if (log_rel[i] == NEG_INF) continue;
                    const std::vector<int> &s_node = result.ancestors[i];
                    double sm = NEG_INF;
                    for (size_t j = 0; j < dd.ancestors.size(); ++j) {
                        if (log_dd[j] == NEG_INF) continue;
                        const double lt = logTransProbVec(s_node, dd.ancestors[j], blen);
                        if (lt == NEG_INF) continue;
                        sm = logAdd(sm, lt + log_dd[j]);
                    }
                    log_rel[i] += sm;
                }
                base_scale += dd.log_scale;
            }

            double max_log_rel = NEG_INF;
            for (double v : log_rel) if (v > max_log_rel) max_log_rel = v;
            if (max_log_rel == NEG_INF) {
                result.log_scale = NEG_INF;
            } else {
                for (size_t i = 0; i < result.ancestors.size(); ++i)
                    result.partial_lh[i] = std::exp(log_rel[i] - max_log_rel);
                result.log_scale = base_scale + max_log_rel;
            }
        }
        c_up->partial_lh_computed |= 1;
        computeSciPhyUpward(childNode[ci], node);
    }
}

// Whole-tree logL from an up partial and a down partial joined by a branch of length len.
double PhyloTreeSciPhy::combineVectors(const std::vector<SciPhyAncestorPartials> &up,
                                      const std::vector<SciPhyAncestorPartials> &down,
                                      double len) const {
    const size_t nptn = aln->getNPattern();
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    if (up.size() != nptn || down.size() != nptn) {
        return NEG_INF;
    }
    double tree_lh = 0.0;
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyAncestorPartials &dc = down[p];  // s_child
        const SciPhyAncestorPartials &uc = up[p];    // s_parent
        // log P(pattern) = logsumexp_{s_par}[ log up[s_par]
        //     + logsumexp_{s_child}( logTrans(s_par -> s_child, len) + log down[s_child] ) ] + scales
        double lp;
        if (uc.wildcard && dc.wildcard) {
            lp = uc.log_scale + dc.log_scale;
        } else if (dc.wildcard) {
            // child side all missing: marginalize the up side
            double lse = NEG_INF;
            for (size_t j = 0; j < uc.partial_lh.size(); ++j)
                if (uc.partial_lh[j] > 0.0) lse = logAdd(lse, std::log(uc.partial_lh[j]));
            lp = (lse == NEG_INF) ? NEG_INF : lse + uc.log_scale + dc.log_scale;
        } else if (uc.wildcard) {
            // up side uninformative (normally not reached): marginalize the down side
            double lse = NEG_INF;
            for (size_t i = 0; i < dc.partial_lh.size(); ++i)
                if (dc.partial_lh[i] > 0.0) lse = logAdd(lse, std::log(dc.partial_lh[i]));
            lp = (lse == NEG_INF) ? NEG_INF : lse + uc.log_scale + dc.log_scale;
        } else {
            double lse_par = NEG_INF;
            for (size_t j = 0; j < uc.ancestors.size(); ++j) {
                if (uc.partial_lh[j] <= 0.0) continue;
                const std::vector<int> &s_par = uc.ancestors[j];
                double lse_child = NEG_INF;
                for (size_t i = 0; i < dc.ancestors.size(); ++i) {
                    if (dc.partial_lh[i] <= 0.0) continue;
                    const double lt = logTransProbVec(s_par, dc.ancestors[i], len);
                    if (lt == NEG_INF) continue;
                    lse_child = logAdd(lse_child, lt + std::log(dc.partial_lh[i]));
                }
                if (lse_child == NEG_INF) continue;
                lse_par = logAdd(lse_par, std::log(uc.partial_lh[j]) + lse_child);
            }
            lp = (lse_par == NEG_INF) ? NEG_INF : lse_par + dc.log_scale + uc.log_scale;
        }
        tree_lh += lp * ptn_freq[p];
    }
    return tree_lh;
}

// combineVectors at the edge (parent, child) using the cached snapshot partials.
double PhyloTreeSciPhy::combineAtBranch(PhyloNode *parent, PhyloNode *child) {
    PhyloNeighbor *down_nei = (PhyloNeighbor*) parent->findNeighbor(child); // Down(child)
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) child->findNeighbor(parent); // UpToC(parent)
    if (!down_nei->sciphy_partial || !up_nei->sciphy_partial) {
        return -std::numeric_limits<double>::infinity();
    }
    return combineVectors(*up_nei->sciphy_partial, *down_nei->sciphy_partial,
                          down_nei->length);
}

// Founder (all-unedited) -> root_int over the origin branch of length t.
double PhyloTreeSciPhy::originClosure(const std::vector<SciPhyAncestorPartials> &bio,
                                      double t, double *pattern_lh_out) {
    const size_t nptn = aln->getNPattern();
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    const double tt = std::max(0.0, t);
    double tree_lh = 0.0;
#pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1) reduction(+: tree_lh)
    for (int p = 0; p < (int)nptn; ++p) {
        const SciPhyAncestorPartials &b = bio[p];
        if (b.wildcard) {
            // tape dropped in every cell: only the dropout constant
            const double lp = b.log_scale;
            if (pattern_lh_out) pattern_lh_out[p] = lp;
            tree_lh += lp * ptn_freq[p];
            continue;
        }
        double lse = NEG_INF;
        if (!b.ancestors.empty()) {
            const std::vector<int> founder(b.ancestors[0].size(), 0);
            for (size_t x = 0; x < b.ancestors.size(); ++x) {
                if (b.partial_lh[x] <= 0.0) continue;
                const double lt = logTransProbVec(founder, b.ancestors[x], tt);
                if (lt == NEG_INF) continue;
                lse = logAdd(lse, lt + std::log(b.partial_lh[x]));
            }
        }
        const double lp = (lse == NEG_INF) ? NEG_INF : lse + b.log_scale;
        if (pattern_lh_out) pattern_lh_out[p] = lp;
        tree_lh += lp * ptn_freq[p];
    }
    return tree_lh;
}

// Whole-tree log-likelihood (full downward pass); fills _pattern_lh and optionally pattern_lh.
double PhyloTreeSciPhy::computeSciPhyLikelihood(double *pattern_lh, bool save_log_value) {
    (void)save_log_value;

    if (!_pattern_lh) {
        initializeAllPartialLh();
    }
    computePtnFreq();

    PhyloNeighbor *saved_current_it      = current_it;
    PhyloNeighbor *saved_current_it_back = current_it_back;
    clearAllPartialLH();   // TODO: track dirty branches instead of clearing everything
    current_it      = saved_current_it;
    current_it_back = saved_current_it_back;

    const size_t nptn = aln->getNPattern();
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

    // Origin branch (founder -> root_int) is a free length; initialize only if unset.
    PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];
    PhyloNode *root_node = (PhyloNode*) root;
    if (origin_edge->length < params->min_branch_length) {
        const ModelSciPhy *sm = dynamic_cast<const ModelSciPhy*>(model);
        const double init = (sm ? 0.5 * sm->getOriginTime() : 1.0);
        origin_edge->length = init;
        PhyloNeighbor *ob = (PhyloNeighbor*) origin_edge->node->findNeighbor(root_node);
        if (ob) ob->length = init;
    }

    computeSciPhyDownward(origin_edge, root_node);
    const std::vector<SciPhyAncestorPartials> &bio = *origin_edge->sciphy_partial;
    tree_lh = originClosure(bio, origin_edge->length, _pattern_lh);

    if (pattern_lh) {
        memmove(pattern_lh, _pattern_lh, nptn * sizeof(double));
    }

    // IQTREE_SCIPHY_PRINTLL=1: print every logL at full precision (debugging)
    if (std::getenv("IQTREE_SCIPHY_PRINTLL")) {
        std::cout << std::setprecision(15)
                  << "SciPhy tree logL = " << tree_lh << std::endl;
    }

    curScore = tree_lh;

    sciphy_snapshot_valid = false;   // upward partials are now stale

    return tree_lh;
}

double PhyloTreeSciPhy::computeLikelihood(double *pattern_lh, bool save_log_value) {
    if (!hasBarcodeAlignment()) {
        return IQTree::computeLikelihood(pattern_lh, save_log_value);
    }
    return computeSciPhyLikelihood(pattern_lh, save_log_value);
}

// <----- Memory: skip the base class's num_states-sized buffers ----->
// num_states = #distinct tape vectors (~1e4 on real data); SciPhy uses sciphy_partial instead.

uint64_t PhyloTreeSciPhy::getMemoryRequired(size_t ncategory, bool full_mem) {
    if (!hasBarcodeAlignment()) {
        return IQTree::getMemoryRequired(ncategory, full_mem);
    }
    const uint64_t nptn  = aln->getNPattern();
    const uint64_t mem_size = get_safe_upper_limit(nptn) + get_safe_upper_limit(aln->num_states);
    uint64_t mem = 8 * mem_size * sizeof(double);           // per-pattern arrays
    // sciphy_partial upper bound: per half-edge, nptn x (L+1) states of L ints + 1 double
    const uint64_t L = 8;                                   // tapes have <= 5 sites
    const uint64_t n_halfedges = 2 * std::max<uint64_t>((uint64_t)nodeNum, 2 * (uint64_t)leafNum);
    mem += n_halfedges * nptn * ((L + 1) * (L * sizeof(int) + sizeof(double)) + 64);
    if (model) mem += model->getMemoryRequired();
    return mem;
}

void PhyloTreeSciPhy::initializeAllPartialLh() {
    if (!hasBarcodeAlignment()) {
        IQTree::initializeAllPartialLh();
        return;
    }
    const int numStates = model->num_states;
    // same per-pattern sizing as PhyloTree::initializeAllPartialLh
    const size_t mem_size = get_safe_upper_limit(getAlnNPattern()) +
        std::max(get_safe_upper_limit(numStates),
                 get_safe_upper_limit(model_factory->unobserved_ptns.size()));
    const size_t ncat_mix = site_rate->getNDiscreteRate() *
        ((model_factory->fused_mix_rate) ? 1 : model->getNMixtures());

    if (!gradient_vector)  gradient_vector  = aligned_alloc<double>(branchNum);
    if (!hessian_diagonal) hessian_diagonal = aligned_alloc<double>(branchNum);
    if (!_pattern_lh)      _pattern_lh      = aligned_alloc<double>(mem_size);
    if (!_pattern_lh_cat)  _pattern_lh_cat  = aligned_alloc<double>(mem_size * ncat_mix);
    if (!_site_lh && (params->robust_phy_keep < 1.0 || params->robust_median))
        _site_lh = aligned_alloc<double>(getAlnNSite());
    if (!_pattern_scaling) _pattern_scaling = aligned_alloc<double>(mem_size);
    if (!buffer_scale_all) buffer_scale_all = aligned_alloc<double>(mem_size);
    if (!ptn_freq) {
        ptn_freq = aligned_alloc<double>(mem_size);
        ptn_freq_computed = false;
    }
    if (!ptn_freq_pars)    ptn_freq_pars    = aligned_alloc<UINT>(mem_size);
    if (!ptn_invar)        ptn_invar        = aligned_alloc<double>(mem_size);

    // Not allocated (num_states-sized, unused): central/tip partial_lh, scale_num,
    // nni_partial_lh, theta_all, buffer_partial_lh, G_matrix.
    clearAllPartialLH();
}

int PhyloTreeSciPhy::wrapperFixNegativeBranch(bool force_change) {
    if (!hasBarcodeAlignment() || force_change) {
        return IQTree::wrapperFixNegativeBranch(force_change);
    }
    clearAllPartialLH();
    // unknown lengths -> 0.1 (ML branch optimization follows)
    int fixed = setNegativeBranch(false, 0.1, root, NULL);
    resetCurScore();
    return fixed;
}

void PhyloTreeSciPhy::removeIdenticalSeqs(Params &params) {
    if (!hasBarcodeAlignment()) {
        IQTree::removeIdenticalSeqs(params);
        return;
    }
    cout << "NOTE: SciPhy barcode alignment -- identical sequences are kept "
            "(de-duplication would drop the barcode alignment type)" << endl;
}

void PhyloTreeSciPhy::reorientPartialLh(PhyloNeighbor* dad_branch, Node *dad) {
    if (!hasBarcodeAlignment()) {
        IQTree::reorientPartialLh(dad_branch, dad);
        return;
    }
    // no base partial_lh buffers; sciphy_partial moves with the half-edges
    (void)dad_branch; (void)dad;
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

    const double tree_lh = computeSciPhyLikelihood(pattern_lh, true);

    if (cur_logl) {
        *cur_logl = tree_lh;
    }

    // single rate category: per-category = per-pattern
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

    // skip the synthetic root edge
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

    (void)maxNRStep;   // no Newton step
}


// Brent objective; the active mode picks which branch `value` sets.
double PhyloTreeSciPhy::computeFunction(double value) {
    if (sciphy_origin_active) {
        PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];
        origin_edge->length = value;
        PhyloNeighbor *ob = (PhyloNeighbor*) origin_edge->node->findNeighbor(root);
        if (ob) ob->length = value;
        return -originClosure(*origin_edge->sciphy_partial, value, nullptr);
    }
    if (sciphy_nni_active) {
        // central branch of a candidate NNI, against local partials
        return -combineVectors(*sciphy_nni_up, *sciphy_nni_down, value);
    }
    if (!sciphy_combine_active) {
        return PhyloTree::computeFunction(value);
    }
    // branch sweep: combine at (parent, child) using the snapshot
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
    sciphy_snapshot_valid = true;
}

void PhyloTreeSciPhy::optimizeOneBranchCombine(PhyloNode *node1, PhyloNode *node2) {
    if (rooted && (node1 == root || node2 == root)) return;   // skip origin edge

    current_it      = (PhyloNeighbor*) node1->findNeighbor(node2);
    current_it_back = (PhyloNeighbor*) node2->findNeighbor(node1);
    if (!current_it || !current_it_back) return;

    if (current_it->direction == TOWARD_ROOT) {
        sciphy_combine_parent = node2; sciphy_combine_child = node1;
    } else {
        sciphy_combine_parent = node1; sciphy_combine_child = node2;
    }

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
    // no invalidation: the whole sweep uses one snapshot (Jacobi)
}

void PhyloTreeSciPhy::optimizeOriginBranch() {
    if (!rooted || !root || root->neighbors.empty()) return;
    PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];
    if (!origin_edge->sciphy_partial) return;   // need cached Down(root_int)
    PhyloNeighbor *ob = (PhyloNeighbor*) origin_edge->node->findNeighbor(root);

    double cur = origin_edge->length;
    if (cur < params->min_branch_length) cur = params->min_branch_length;

    sciphy_origin_active = true;
    double neg = 0.0, ferr = 0.0;
    const double init_neg = -originClosure(*origin_edge->sciphy_partial, cur, nullptr);
    const double optx = minimizeOneDimen(params->min_branch_length, cur,
                                         params->max_branch_length, params->min_branch_length,
                                         &neg, &ferr);
    sciphy_origin_active = false;

    const double best = (neg > init_neg + 1e-8) ? cur : optx;
    origin_edge->length = best;
    if (ob) ob->length = best;
}

double PhyloTreeSciPhy::optimizeAllBranches(int my_iterations, double tolerance, int maxNRStep) {
    if (!hasBarcodeAlignment()) {
        return IQTree::optimizeAllBranches(my_iterations, tolerance, maxNRStep);
    }
    (void)maxNRStep;

    if (rooted) computeBranchDirection();

    NodeVector nodes, nodes2;
    getBranches(nodes, nodes2);
    ASSERT(nodes.size() == nodes2.size());

    double tree_lh = computeLikelihood(nullptr, true);   // exact
    DoubleVector saved_lengths;

    for (int iter = 0; iter < my_iterations; ++iter) {
        const double prev_lh = tree_lh;
        saveBranchLengths(saved_lengths);

        // one up/down snapshot per sweep
        computeSciPhyLikelihood(nullptr, true);
        computeSciPhyFullUpward();
        optimizeOriginBranch();

        for (size_t j = 0; j < nodes.size(); ++j) {
            optimizeOneBranchCombine((PhyloNode*) nodes[j], (PhyloNode*) nodes2[j]);
        }

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
    // root is fixed
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

    std::vector<PhyloNeighbor*> kids;
    for (NeighborVec::iterator nit = child->neighbors.begin();
         nit != child->neighbors.end(); ++nit) {
        if ((*nit)->node != parent) kids.push_back((PhyloNeighbor*)(*nit));
    }
    if (kids.empty()) { out.clear(); return; }
    for (PhyloNeighbor *k : kids)
        if (!k->sciphy_partial) { out.clear(); return; }   // cache miss

    // same combine as computeSciPhyDownward
    for (size_t p = 0; p < nptn; ++p) {
        SciPhyAncestorPartials &result = out[p];
        const double NEG_INF = -std::numeric_limits<double>::infinity();

        std::vector<PhyloNeighbor*> obs_kids;
        double wild_scale = 0.0;
        for (size_t k = 0; k < kids.size(); ++k) {
            const SciPhyAncestorPartials &cp = (*kids[k]->sciphy_partial)[p];
            if (cp.wildcard) wild_scale += cp.log_scale;
            else             obs_kids.push_back(kids[k]);
        }
        if (obs_kids.empty()) {
            result.wildcard = true;
            result.ancestors.clear();
            result.partial_lh.clear();
            result.log_scale = wild_scale;
            continue;
        }

        result.ancestors = (*obs_kids[0]->sciphy_partial)[p].ancestors;
        for (size_t k = 1; k < obs_kids.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, (*obs_kids[k]->sciphy_partial)[p].ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0);

        double base_scale = wild_scale;
        for (size_t k = 0; k < obs_kids.size(); ++k)
            base_scale += (*obs_kids[k]->sciphy_partial)[p].log_scale;

        std::vector<std::vector<double>> log_cp(obs_kids.size());
        for (size_t k = 0; k < obs_kids.size(); ++k) {
            const SciPhyAncestorPartials &cd = (*obs_kids[k]->sciphy_partial)[p];
            log_cp[k].resize(cd.ancestors.size());
            for (size_t c = 0; c < cd.ancestors.size(); ++c)
                log_cp[k][c] = (cd.partial_lh[c] > 0.0) ? std::log(cd.partial_lh[c]) : NEG_INF;
        }

        std::vector<double> log_rel(result.ancestors.size(), NEG_INF);
        double max_log_rel = NEG_INF;
        for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx) {
            const std::vector<int> &s_vec = result.ancestors[s_idx];
            double log_lh_s = 0.0;
            for (size_t k = 0; k < obs_kids.size(); ++k) {
                const SciPhyAncestorPartials &cd = (*obs_kids[k]->sciphy_partial)[p];
                const double blen = obs_kids[k]->length;
                double child_lse = NEG_INF;
                for (size_t c_idx = 0; c_idx < cd.ancestors.size(); ++c_idx) {
                    if (log_cp[k][c_idx] == NEG_INF) continue;
                    const double lt = logTransProbVec(s_vec, cd.ancestors[c_idx], blen);
                    if (lt == NEG_INF) continue;
                    child_lse = logAdd(child_lse, lt + log_cp[k][c_idx]);
                }
                log_lh_s += child_lse;
                if (log_lh_s == NEG_INF) break;
            }
            log_rel[s_idx] = log_lh_s;
            if (log_lh_s > max_log_rel) max_log_rel = log_lh_s;
        }
        if (max_log_rel == NEG_INF) {
            result.log_scale = NEG_INF;
        } else {
            for (size_t s_idx = 0; s_idx < result.ancestors.size(); ++s_idx)
                result.partial_lh[s_idx] = std::exp(log_rel[s_idx] - max_log_rel);
            result.log_scale = base_scale + max_log_rel;
        }
    }
}

void PhyloTreeSciPhy::buildUpToCLocal(PhyloNode *parent, PhyloNode *child,
                                      std::vector<SciPhyAncestorPartials> &out) const {
    const size_t nptn = aln->getNPattern();
    out.assign(nptn, SciPhyAncestorPartials());

    // parent's toward-root half-edge holds UpToC(grandparent or founder)
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

    // parent's children: `child` plus its siblings
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
        const double NEG_INF = -std::numeric_limits<double>::infinity();
        const SciPhyAncestorPartials &uP = upP[p];

        // observed kids define the support; all-missing -> constant
        std::vector<PhyloNeighbor*> obs_support;
        for (size_t k = 0; k < kids.size(); ++k)
            if (!(*kids[k]->sciphy_partial)[p].wildcard) obs_support.push_back(kids[k]);

        if (uP.wildcard || obs_support.empty()) {
            double ls = totalLogMass(uP);
            for (size_t k = 0; k < kids.size(); ++k) {
                if (kids[k] == child_nei) continue;
                const SciPhyAncestorPartials &dd = (*kids[k]->sciphy_partial)[p];
                if (dd.wildcard) ls += dd.log_scale;
            }
            result.wildcard = true;
            result.ancestors.clear();
            result.partial_lh.clear();
            result.log_scale = ls;
            continue;
        }

        result.ancestors = (*obs_support[0]->sciphy_partial)[p].ancestors;
        for (size_t k = 1; k < obs_support.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, (*obs_support[k]->sciphy_partial)[p].ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0);

        double base_scale = uP.log_scale;

        // UpBar(parent), then times each sibling's down contribution
        std::vector<double> log_uP(uP.ancestors.size(), NEG_INF);
        for (size_t j = 0; j < uP.ancestors.size(); ++j)
            log_uP[j] = (uP.partial_lh[j] > 0.0) ? std::log(uP.partial_lh[j]) : NEG_INF;

        std::vector<double> log_rel(result.ancestors.size(), NEG_INF);
        for (size_t i = 0; i < result.ancestors.size(); ++i) {
            const std::vector<int> &s = result.ancestors[i];
            double lse = NEG_INF;
            for (size_t j = 0; j < uP.ancestors.size(); ++j) {
                if (log_uP[j] == NEG_INF) continue;
                const double lt = logTransProbVec(uP.ancestors[j], s, len_Pp);
                if (lt == NEG_INF) continue;
                lse = logAdd(lse, log_uP[j] + lt);
            }
            log_rel[i] = lse;
        }

        for (size_t k = 0; k < kids.size(); ++k) {
            if (kids[k] == child_nei) continue;
            const SciPhyAncestorPartials &dd = (*kids[k]->sciphy_partial)[p];
            if (dd.wildcard) { base_scale += dd.log_scale; continue; }
            const double blen = kids[k]->length;
            std::vector<double> log_dd(dd.ancestors.size(), NEG_INF);
            for (size_t jj = 0; jj < dd.ancestors.size(); ++jj)
                log_dd[jj] = (dd.partial_lh[jj] > 0.0) ? std::log(dd.partial_lh[jj]) : NEG_INF;
            for (size_t i = 0; i < result.ancestors.size(); ++i) {
                if (log_rel[i] == NEG_INF) continue;
                const std::vector<int> &s = result.ancestors[i];
                double sm = NEG_INF;
                for (size_t jj = 0; jj < dd.ancestors.size(); ++jj) {
                    if (log_dd[jj] == NEG_INF) continue;
                    const double lt = logTransProbVec(s, dd.ancestors[jj], blen);
                    if (lt == NEG_INF) continue;
                    sm = logAdd(sm, lt + log_dd[jj]);
                }
                log_rel[i] += sm;
            }
            base_scale += dd.log_scale;
        }

        double max_log_rel = NEG_INF;
        for (double v : log_rel) if (v > max_log_rel) max_log_rel = v;
        if (max_log_rel == NEG_INF) {
            result.log_scale = NEG_INF;
        } else {
            for (size_t i = 0; i < result.ancestors.size(); ++i)
                result.partial_lh[i] = std::exp(log_rel[i] - max_log_rel);
            result.log_scale = base_scale + max_log_rel;
        }
    }
}

void PhyloTreeSciPhy::doNNI(NNIMove &move, bool clearLH) {
    PhyloTree::doNNI(move, clearLH);
    sciphy_snapshot_valid = false;
}

void PhyloTreeSciPhy::changeNNIBrans(NNIMove &nnimove) {
    if (!hasBarcodeAlignment()) {
        PhyloTree::changeNNIBrans(nnimove);
        return;
    }
    // only the central branch was optimized during scoring
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

    ensureSciPhySnapshot();

    // orient so node1 is the parent
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

    // two candidate swaps: node1's other child with each of node2's children
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
        Neighbor *cur_n1 = *n1it, *cur_n2 = *n2it;
        node1->updateNeighbor(n1it, cur_n2);
        cur_n2->node->updateNeighbor(node2, node1);
        node2->updateNeighbor(n2it, cur_n1);
        cur_n1->node->updateNeighbor(node1, node2);
    };

    // For each swap, rebuild only the two partials next to the central branch and
    // optimize its length: O(local) per candidate instead of O(N).
    // IQTREE_SCIPHY_CHECK=1 also computes the exact logL and reports differences.
    const bool dbg_cache = std::getenv("IQTREE_SCIPHY_CHECK") != nullptr;
    for (int cnt = 0; cnt < 2; cnt++) if (constraintTree.isCompatible(nniMoves[cnt])) {
        // check mode overwrites the snapshot, so rebuild it first
        if (dbg_cache) { sciphy_snapshot_valid = false; ensureSciPhySnapshot(); }

        rawSwap(cnt);
        std::vector<SciPhyAncestorPartials> downChild, upParent;
        buildDownwardLocal(node2, node1, downChild);
        buildUpToCLocal(node1, node2, upParent);

        double score;
        double optlen = orig_central_len;
        bool cached = (downChild.size() == nptn && upParent.size() == nptn);
        // TODO: local combine is not exact with wildcard partials; use the exact recompute
        if (cached)
            for (size_t p = 0; p < nptn; ++p)
                if (downChild[p].wildcard || upParent[p].wildcard) { cached = false; break; }
        if (cached) {
            sciphy_nni_up = &upParent;
            sciphy_nni_down = &downChild;
            sciphy_nni_active = true;
            double neg = 0.0, ferr = 0.0;
            const double initial_neg = -combineVectors(upParent, downChild, orig_central_len);
            const double optx = minimizeOneDimen(params->min_branch_length, orig_central_len,
                                                 params->max_branch_length, params->min_branch_length,
                                                 &neg, &ferr);
            sciphy_nni_active = false;
            sciphy_nni_up = sciphy_nni_down = nullptr;
            const bool keep_orig = (neg > initial_neg + 1e-8);
            optlen = keep_orig ? orig_central_len : optx;
            score  = -(keep_orig ? initial_neg : neg);
        } else {
            central->length = central_back->length = orig_central_len;
            score = computeLikelihood();
        }

        if (dbg_cache) {
            central->length = central_back->length = optlen;
            const double exact = computeLikelihood();
            if (std::fabs(exact - score) > 1e-6)
                std::cerr << std::setprecision(12)
                          << "[nni cache check] n1=" << node1->id << " n2=" << node2->id
                          << " cnt=" << cnt << " cached=" << score << " exact=" << exact
                          << " diff=" << (exact - score) << (cached ? "" : " (fallback)") << "\n";
            score = exact;
        }

        central->length = central_back->length = orig_central_len;
        nniMoves[cnt].newloglh = score;
        nniMoves[cnt].newLen[0].assign(1, optlen);
        undoSwap(cnt);
    }

    ensureSciPhySnapshot();
    curScore = backupScore;

    NNIMove res = (nniMoves[0].newloglh > nniMoves[1].newloglh) ? nniMoves[0] : nniMoves[1];
    if (newNNIMoves) delete[] nniMoves;
    return res;
}
