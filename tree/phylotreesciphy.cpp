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
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

PhyloTreeSciPhy::SciPhyKernel kernelFromEnv() {
    typedef PhyloTreeSciPhy PTS;
    const char *k = std::getenv("IQTREE_SCIPHY_KERNEL");
    if (!k) return PTS::SCIPHY_KERNEL_TAPE;
    if (!strcmp(k, "old")) return PTS::SCIPHY_KERNEL_OLD;
    if (!strcmp(k, "check")) return PTS::SCIPHY_KERNEL_CHECK;
    if (!strcmp(k, "check-raw")) return PTS::SCIPHY_KERNEL_CHECK_RAW;
    outError(std::string("IQTREE_SCIPHY_KERNEL must be 'old', 'check' or 'check-raw', not ") + k);
    return PTS::SCIPHY_KERNEL_TAPE;
}

}  // anonymous namespace

PhyloTreeSciPhy::PhyloTreeSciPhy() : IQTree() { readEnv(); }

PhyloTreeSciPhy::PhyloTreeSciPhy(Alignment *aln) : IQTree(aln) { readEnv(); }

void PhyloTreeSciPhy::readEnv() {
    sciphy_kernel = kernelFromEnv();
    print_ll = std::getenv("IQTREE_SCIPHY_PRINTLL") != nullptr;
    nni_cache_check = std::getenv("IQTREE_SCIPHY_CHECK") != nullptr;
    const char *bl = std::getenv("IQTREE_SCIPHY_BRLEN");
    brlen_brent = bl && !strcmp(bl, "brent");
    brlen_check = std::getenv("IQTREE_SCIPHY_CHECK_BRLEN") != nullptr;
}

PhyloTreeSciPhy::~PhyloTreeSciPhy() {
    reportKernelCheck();
}

void PhyloTreeSciPhy::reportKernelCheck() {
    for (auto &it : check_stats)
        std::cout << std::scientific << std::setprecision(2) << "[sciphy kernel check] " << it.first
                  << ": " << it.second.n << " tape comparisons, max relative |dlogL| = "
                  << it.second.max_rel << ", " << it.second.fail << " above threshold" << std::endl;
    check_stats.clear();
}

const ModelSciPhy &PhyloTreeSciPhy::sciphyModel() const {
    const ModelSciPhy *m = dynamic_cast<const ModelSciPhy*>(model);
    ASSERT(m);
    return *m;
}

const BarcodeAlignment &PhyloTreeSciPhy::barcodeAlignment() const {
    const BarcodeAlignment *a = dynamic_cast<const BarcodeAlignment*>(aln);
    ASSERT(a);
    return *a;
}

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

// P(K >= r) for K ~ Poisson(mean). direct = false is 1 - sum of the lower terms, which
// cancels below ~1e-16 (e.g. 2e-17 instead of 2e-45 on short branches); direct = true sums
// the upper terms, past the mode by TAIL_EXTRA_TERMS.
double poissonUpperTail(int r, double mean, bool direct) {
    const int TAIL_EXTRA_TERMS = 60;
    if (!direct) {
        double tail = 1.0;
        for (int i = 0; i < r; ++i) tail -= poissonPMF(i, mean);
        return tail;
    }
    double tail = (r == 0) ? 1.0 : 0.0;
    if (r > 0)
        for (int i = r; i < r + TAIL_EXTRA_TERMS || i < 2.0 * mean + TAIL_EXTRA_TERMS; ++i)
            tail += poissonPMF(i, mean);
    return tail;
}

// Probability of an insert of type scar (1..M); 0 if out of range. Types beyond
// edit_probs are uniform.
double insertProb(const std::vector<double> &edit_probs, int M, int scar) {
    if (scar < 1 || scar > M) return 0.0;
    return (scar - 1 < (int)edit_probs.size()) ? edit_probs[scar - 1] : 1.0 / M;
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

    // Sites fill left to right, so the parent's edits must sit at the same positions in the
    // child; the child's remaining edits are the new inserts.
    int n_parent = 0;
    std::vector<int> new_inserts;
    new_inserts.reserve((size_t)L);
    for (int i = 0; i < L; ++i) {
        if (parent[i] != 0) {
            if (child[i] != parent[i]) return NEG_INF;
            ++n_parent;
        } else if (child[i] != 0) {
            new_inserts.push_back(child[i]);
        }
    }

    const int nr_possible = L - n_parent;
    const int k = (int)new_inserts.size();
    if (k > nr_possible) return NEG_INF;

    // Scar-type factor, in log space.
    double log_insert = 0.0;
    for (int scar : new_inserts) {
        const double pe = insertProb(edit_probs, M, scar);
        if (!(pe > 0.0)) return NEG_INF;
        log_insert += std::log(pe);
    }

    const double distance = std::max(0.0, lambda * branch_length);

    if (k == nr_possible) {
        // Absorbing tail P(K >= nr_possible) under Poisson(distance). The check oracle
        // sums it directly; the old kernel itself cancels (see poissonUpperTail).
        const double tail = poissonUpperTail(nr_possible, distance, oracleDirectTail());
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

// An up partial as seen by an observed child: itself, or (for a wildcard up partial with an
// anchor) the anchor carried over extra_len more branch length, times exp(extra_scale).
struct UpView {
    const SciPhyAncestorPartials *u;
    double extra_len;
    double extra_scale;
};
UpView resolveUp(const SciPhyAncestorPartials &v) {
    if (v.wildcard && v.anchor) {
        const double m = totalLogMass(*v.anchor);
        if (m != -std::numeric_limits<double>::infinity())
            return {v.anchor.get(), v.anchor_len, v.log_scale - m};
    }
    return {&v, 0.0, 0.0};
}

// Wildcard up partial with constant log_scale that keeps the state dependence of `up`
// carried over a further branch of length len.
void makeDeferredUp(SciPhyAncestorPartials &out, const SciPhyAncestorPartials &up,
                    double len, double log_scale) {
    out.wildcard = true;
    out.ancestors.clear();
    out.partial_lh.clear();
    out.log_scale = log_scale;
    if (up.wildcard) {
        out.anchor = up.anchor;
        out.anchor_len = up.anchor ? up.anchor_len + len : 0.0;
    } else {
        auto a = std::make_shared<SciPhyAncestorPartials>();
        a->ancestors = up.ancestors;
        a->partial_lh = up.partial_lh;
        a->log_scale = up.log_scale;
        out.anchor = a;
        out.anchor_len = len;
    }
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

        // all-missing subtree: constant total up mass, state dependence kept as an anchor
        if (dn.wildcard) {
            makeDeferredUp(ub, ud, len_dad_node, totalLogMass(ud));
            continue;
        }
        const UpView uv = resolveUp(ud);
        if (uv.u->wildcard) {   // uninformative up side without an anchor
            ub.wildcard = true;
            ub.ancestors.clear();
            ub.partial_lh.clear();
            ub.log_scale = totalLogMass(ud);
            continue;
        }
        const SciPhyAncestorPartials &ue = *uv.u;
        const double len_eff = len_dad_node + uv.extra_len;

        ub.ancestors = dn.ancestors;                        // indexed by s_node
        ub.partial_lh.assign(ub.ancestors.size(), 0.0);

        std::vector<double> log_ud(ue.ancestors.size(), NEG_INF);
        for (size_t j = 0; j < ue.ancestors.size(); ++j)
            log_ud[j] = (ue.partial_lh[j] > 0.0) ? std::log(ue.partial_lh[j]) : NEG_INF;

        std::vector<double> log_rel(ub.ancestors.size(), NEG_INF);
        double max_log_rel = NEG_INF;
        for (size_t i = 0; i < ub.ancestors.size(); ++i) {
            const std::vector<int> &s_node = ub.ancestors[i];
            double lse = NEG_INF;
            for (size_t j = 0; j < ue.ancestors.size(); ++j) {
                if (log_ud[j] == NEG_INF) continue;
                const double lt = logTransProbVec(ue.ancestors[j], s_node, len_eff);
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
            ub.log_scale = ue.log_scale + uv.extra_scale + max_log_rel;
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
                result.anchor = upBar[p].anchor;          // same node state: same anchor
                result.anchor_len = upBar[p].anchor_len;
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
                                      double len, double *pattern_lh_out) const {
    const size_t nptn = aln->getNPattern();
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    if (up.size() != nptn || down.size() != nptn) {
        return NEG_INF;
    }
    double tree_lh = 0.0;
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyAncestorPartials &dc = down[p];  // s_child
        // an anchored wildcard up partial facing an observed child: use the anchor
        const UpView uv = dc.wildcard ? UpView{&up[p], 0.0, 0.0} : resolveUp(up[p]);
        const SciPhyAncestorPartials &uc = *uv.u;    // s_parent
        const double len_eff = len + uv.extra_len;
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
            // up side uninformative with no anchor (not reached for snapshot partials)
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
                    const double lt = logTransProbVec(s_par, dc.ancestors[i], len_eff);
                    if (lt == NEG_INF) continue;
                    lse_child = logAdd(lse_child, lt + std::log(dc.partial_lh[i]));
                }
                if (lse_child == NEG_INF) continue;
                lse_par = logAdd(lse_par, std::log(uc.partial_lh[j]) + lse_child);
            }
            lp = (lse_par == NEG_INF) ? NEG_INF
                                      : lse_par + dc.log_scale + uc.log_scale + uv.extra_scale;
        }
        if (pattern_lh_out) pattern_lh_out[p] = lp;
        tree_lh += lp * ptn_freq[p];
    }
    return tree_lh;
}

// combineVectors at the edge (parent, child) using the cached snapshot partials.
double PhyloTreeSciPhy::combineAtBranch(PhyloNode *parent, PhyloNode *child) {
    PhyloNeighbor *down_nei = (PhyloNeighbor*) parent->findNeighbor(child); // Down(child)
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) child->findNeighbor(parent); // UpToC(parent)
    if (!hasPartial(down_nei) || !hasPartial(up_nei)) {
        return -std::numeric_limits<double>::infinity();
    }
    if (useOldKernel())
        return combineVectors(*up_nei->sciphy_partial, *down_nei->sciphy_partial,
                              down_nei->length);
    if (checking()) {
        const size_t nptn = aln->getNPattern();
        std::vector<double> new_ptn(nptn), old_ptn(nptn);
        const double lh = combineTape(*up_nei->sciphy_tape_partial, *down_nei->sciphy_tape_partial,
                                      down_nei->length, new_ptn.data());
        combineVectors(*up_nei->sciphy_partial, *down_nei->sciphy_partial,
                       down_nei->length, old_ptn.data());
        checkTapes("branch", new_ptn.data(), old_ptn.data());
        return lh;
    }
    return combineTape(*up_nei->sciphy_tape_partial, *down_nei->sciphy_tape_partial,
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

    if (useOldKernel()) {
        computeSciPhyDownward(origin_edge, root_node);
        tree_lh = originClosure(*origin_edge->sciphy_partial, origin_edge->length, _pattern_lh);
    } else {
        initTapeKernel();
        computeTapeDownward(origin_edge, root_node);
        tree_lh = originClosureTape(*origin_edge->sciphy_tape_partial, origin_edge->length, _pattern_lh);
    }

    if (checking()) {
        // old kernel from scratch on the same tree; computed flags are shared
        clearAllPartialLH();
        current_it      = saved_current_it;
        current_it_back = saved_current_it_back;
        computeSciPhyDownward(origin_edge, root_node);
        std::vector<double> old_ptn(nptn);
        originClosure(*origin_edge->sciphy_partial, origin_edge->length, old_ptn.data());
        checkTapes("likelihood", _pattern_lh, old_ptn.data());
    }

    if (pattern_lh) {
        memmove(pattern_lh, _pattern_lh, nptn * sizeof(double));
    }

    // IQTREE_SCIPHY_PRINTLL=1: print every logL at full precision (debugging)
    if (print_ll) {
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
// num_states = #distinct tape vectors (~1e4 on real data); SciPhy caches per-edge tape partials instead.

uint64_t PhyloTreeSciPhy::getMemoryRequired(size_t ncategory, bool full_mem) {
    if (!hasBarcodeAlignment()) {
        return IQTree::getMemoryRequired(ncategory, full_mem);
    }
    const uint64_t nptn  = aln->getNPattern();
    const uint64_t mem_size = get_safe_upper_limit(nptn) + get_safe_upper_limit(aln->num_states);
    uint64_t mem = 8 * mem_size * sizeof(double);           // per-pattern arrays
    const uint64_t n_halfedges = 2 * std::max<uint64_t>((uint64_t)nodeNum, 2 * (uint64_t)leafNum);
    if (!useOldKernel()) {
        // per half-edge: a head and L+1 doubles per tape; plus 3 scratch partials
        uint64_t slots = 0;
        for (size_t p = 0; p < nptn; ++p) slots += patternTapeLength(p) + 1;
        const uint64_t per_partial = nptn * sizeof(SciPhyTapeHead) + slots * sizeof(double);
        mem += (n_halfedges + 3) * per_partial;
    }
    if (useOldKernel() || checking()) {
        // sciphy_partial upper bound: per half-edge, nptn x (L+1) states of L ints + 1 double
        const uint64_t L = 8;                               // tapes have <= 5 sites
        mem += n_halfedges * nptn * ((L + 1) * (L * sizeof(int) + sizeof(double)) + 64);
    }
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

void PhyloTreeSciPhy::prepareToComputeDistances() {
    if (!hasBarcodeAlignment()) IQTree::prepareToComputeDistances();
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


// logL over the origin branch (length t) with the active kernel; check mode also
// compares against the old kernel.
double PhyloTreeSciPhy::originLikelihood(PhyloNeighbor *origin_edge, double t) {
    if (useOldKernel())
        return originClosure(*origin_edge->sciphy_partial, t, nullptr);
    if (!checking())
        return originClosureTape(*origin_edge->sciphy_tape_partial, t, nullptr);
    const size_t nptn = aln->getNPattern();
    std::vector<double> new_ptn(nptn), old_ptn(nptn);
    const double lh = originClosureTape(*origin_edge->sciphy_tape_partial, t, new_ptn.data());
    originClosure(*origin_edge->sciphy_partial, t, old_ptn.data());
    checkTapes("origin", new_ptn.data(), old_ptn.data());
    return lh;
}

// Brent objective; the active mode picks which branch `value` sets.
double PhyloTreeSciPhy::computeFunction(double value) {
    if (clock_age_active) return -clockAgeObjective(value);
    if (clock_rate_active) { clockSetRate(value); return -computeLikelihood(); }
    if (clock_scale_active) {
        for (PhyloNode *v : clock_post) clock_age[v->id] = clock_scale_base[v->id] * value;
        clockApplyAllAges();
        clockSetRate(clock_scale_rate / value);
        return -computeLikelihood();
    }
    if (sciphy_origin_active) {
        PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];
        origin_edge->length = value;
        PhyloNeighbor *ob = (PhyloNeighbor*) origin_edge->node->findNeighbor(root);
        if (ob) ob->length = value;
        return -originLikelihood(origin_edge, value);
    }
    if (sciphy_nni_active) {
        // central branch of a candidate NNI, against local partials
        if (useOldKernel())
            return -combineVectors(*sciphy_nni_up, *sciphy_nni_down, value);
        return -combineTape(*sciphy_nni_up_tape, *sciphy_nni_down_tape, value);
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
    if (useOldKernel()) {
        computeAncestorFullUpward();
        return;
    }
    computeTapeFullUpward();
    if (checking()) {
        computeAncestorFullUpward();
        checkAllBranches();
    }
}

void PhyloTreeSciPhy::computeAncestorFullUpward() {
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
    if (!hasPartial(down_nei) || !hasPartial(up_nei)) return;

    const double current_len = current_it->length;
    double negative_lh = 0.0, ferror = 0.0;

    if (closedFormBrlen()) {
        buildBranchCoeffs(*up_nei->sciphy_tape_partial, *down_nei->sciphy_tape_partial, brlen_bc);
        double nl;
        bool used;
        const double fl = optimizeBranchClosedForm(brlen_bc, current_len, nl, used);
        if (brlen_check) recordCheck("brlen-sweep fallback to Brent", used ? 0.0 : 1.0, 0.5);
        if (used) {
            if (brlen_check) {
                const SciPhyTapePartials &U = *up_nei->sciphy_tape_partial, &D = *down_nei->sciphy_tape_partial;
                checkBranchClosedForm("brlen-sweep", brlen_bc, current_len, nl, fl,
                                      [&](double l) { return combineTape(U, D, l); });
            }
            current_it->length = current_it_back->length = nl;
            return;
        }
    }

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
    if (!hasPartial(origin_edge)) return;   // need cached Down(root_int)
    PhyloNeighbor *ob = (PhyloNeighbor*) origin_edge->node->findNeighbor(root);

    double cur = origin_edge->length;
    if (cur < params->min_branch_length) cur = params->min_branch_length;

    if (closedFormBrlen()) {
        buildOriginCoeffs(*origin_edge->sciphy_tape_partial, brlen_bc);
        double nl;
        bool used;
        const double fl = optimizeBranchClosedForm(brlen_bc, cur, nl, used);
        if (brlen_check) recordCheck("brlen-origin fallback to Brent", used ? 0.0 : 1.0, 0.5);
        if (used) {
            if (brlen_check) {
                const SciPhyTapePartials &bio = *origin_edge->sciphy_tape_partial;
                checkBranchClosedForm("brlen-origin", brlen_bc, cur, nl, fl,
                                      [&](double l) { return originClosureTape(bio, l, nullptr); });
            }
            origin_edge->length = nl;
            if (ob) ob->length = nl;
            return;
        }
    }

    sciphy_origin_active = true;
    double neg = 0.0, ferr = 0.0;
    const double init_neg = computeFunction(cur);
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

        if (parallelPathEnabled()) {
            // branches use the same snapshot, so they are independent; apply lengths afterwards
            const int nb = (int)nodes.size(), nt = numWorkThreads();
            std::vector<double> new_len(nb, 0.0);
            std::vector<char> status(nb, 0);
            workspace(nt - 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt) if (nt > 1)
#endif
            for (int j = 0; j < nb; ++j) {
                Workspace &ws = threadWorkspace();
                status[j] = (char) sweepBranchClosedForm((PhyloNode*) nodes[j], (PhyloNode*) nodes2[j],
                                                         new_len[j], ws);
            }
            for (int j = 0; j < nb; ++j) {
                PhyloNode *n1 = (PhyloNode*) nodes[j], *n2 = (PhyloNode*) nodes2[j];
                if (status[j] == 1) {
                    PhyloNeighbor *a = (PhyloNeighbor*) n1->findNeighbor(n2);
                    PhyloNeighbor *b = (PhyloNeighbor*) n2->findNeighbor(n1);
                    a->length = b->length = new_len[j];
                } else if (status[j] == 2) {
                    optimizeOneBranchCombine(n1, n2);   // Brent fallback
                }
                // current_it = last non-origin branch, as in the sequential loop
                if (!(rooted && (n1 == root || n2 == root))) {
                    current_it      = (PhyloNeighbor*) n1->findNeighbor(n2);
                    current_it_back = (PhyloNeighbor*) n2->findNeighbor(n1);
                }
            }
        } else {
            for (size_t j = 0; j < nodes.size(); ++j) {
                optimizeOneBranchCombine((PhyloNode*) nodes[j], (PhyloNode*) nodes2[j]);
            }
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
                                      const std::vector<SciPhyAncestorPartials> &child_down,
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
    if (kids.empty() || !child_nei || child_down.size() != nptn) { out.clear(); return; }
    for (PhyloNeighbor *k : kids)
        if (!k->sciphy_partial) { out.clear(); return; }
    auto kidPartial = [&](PhyloNeighbor *k, size_t p) -> const SciPhyAncestorPartials & {
        return (k == child_nei) ? child_down[p] : (*k->sciphy_partial)[p];
    };

    for (size_t p = 0; p < nptn; ++p) {
        SciPhyAncestorPartials &result = out[p];
        const double NEG_INF = -std::numeric_limits<double>::infinity();
        const SciPhyAncestorPartials &uP0 = upP[p];

        // observed kids define the support; all-missing -> constant (anchored)
        std::vector<PhyloNeighbor*> obs_support;
        for (size_t k = 0; k < kids.size(); ++k)
            if (!kidPartial(kids[k], p).wildcard) obs_support.push_back(kids[k]);

        const UpView uv = resolveUp(uP0);
        if (obs_support.empty() || uv.u->wildcard) {
            double ls = totalLogMass(uP0);
            for (size_t k = 0; k < kids.size(); ++k) {
                if (kids[k] == child_nei) continue;
                const SciPhyAncestorPartials &dd = (*kids[k]->sciphy_partial)[p];
                if (dd.wildcard) ls += dd.log_scale;
            }
            if (obs_support.empty()) {
                makeDeferredUp(result, uP0, len_Pp, ls);
            } else {   // up side without an anchor: legacy constant
                result.wildcard = true;
                result.ancestors.clear();
                result.partial_lh.clear();
                result.log_scale = ls;
            }
            continue;
        }
        const SciPhyAncestorPartials &uP = *uv.u;
        const double len_up = len_Pp + uv.extra_len;

        result.ancestors = kidPartial(obs_support[0], p).ancestors;
        for (size_t k = 1; k < obs_support.size(); ++k) {
            result.ancestors = intersectAncestorLists(
                result.ancestors, kidPartial(obs_support[k], p).ancestors);
            if (result.ancestors.empty()) break;
        }
        result.partial_lh.assign(result.ancestors.size(), 0.0);

        double base_scale = uP.log_scale + uv.extra_scale;

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
                const double lt = logTransProbVec(uP.ancestors[j], s, len_up);
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

// Down(node2) and UpToC(node1) of the current (swapped) topology, for the active kernel.
// False if a cached partial is missing, or (tape kernel) a wildcard tape makes the local combine inexact.
bool PhyloTreeSciPhy::buildNNILocalPartials(PhyloNode *node1, PhyloNode *node2,
                                            std::vector<SciPhyAncestorPartials> &downChild,
                                            std::vector<SciPhyAncestorPartials> &upParent) {
    const size_t nptn = aln->getNPattern();
    if (useOldKernel()) {
        buildDownwardLocal(node2, node1, downChild);
        buildUpToCLocal(node1, node2, downChild, upParent);
        // wildcard (dropout) partials are exact too: up partials keep an anchor
        return downChild.size() == nptn && upParent.size() == nptn;
    }
    buildTapeDownwardLocal(node2, node1, tape_nni_down);
    buildTapeUpToCLocal(node1, node2, tape_nni_down, tape_nni_up);
    if (tape_nni_down.head.size() != nptn || tape_nni_up.head.size() != nptn) return false;
    // tape partials keep no anchor: wildcard tapes use the exact recompute
    for (size_t p = 0; p < nptn; ++p)
        if (tape_nni_down.head[p].wildcard || tape_nni_up.head[p].wildcard) return false;
    return true;
}

// logL of the swapped topology around node1-node2, with the central branch length
// optimized (optlen). Falls back to the exact logL when local partials can't be used.
// IQTREE_SCIPHY_CHECK=1 also computes the exact logL and reports differences.
double PhyloTreeSciPhy::scoreNNICandidate(PhyloNode *node1, PhyloNode *node2, int cnt,
                                          double orig_len, double &optlen) {
    PhyloNeighbor *central      = (PhyloNeighbor*) node1->findNeighbor(node2);
    PhyloNeighbor *central_back = (PhyloNeighbor*) node2->findNeighbor(node1);
    std::vector<SciPhyAncestorPartials> downChild, upParent;
    const bool cached = buildNNILocalPartials(node1, node2, downChild, upParent);

    double score;
    optlen = orig_len;
    bool done_closed = false;
    if (cached && closedFormBrlen()) {
        buildBranchCoeffs(tape_nni_up, tape_nni_down, brlen_bc);
        bool used;
        double nl;
        const double fl = optimizeBranchClosedForm(brlen_bc, orig_len, nl, used);
        if (brlen_check) recordCheck("brlen-nni fallback to Brent", used ? 0.0 : 1.0, 0.5);
        if (used) {
            if (brlen_check)
                checkBranchClosedForm("brlen-nni", brlen_bc, orig_len, nl, fl,
                                      [&](double l) { return combineTape(tape_nni_up, tape_nni_down, l); });
            optlen = nl;
            score = fl;
            done_closed = true;
        }
    }
    if (!done_closed && cached) {   // Brent on the local partials
        if (useOldKernel()) {
            sciphy_nni_up = &upParent;
            sciphy_nni_down = &downChild;
        } else {
            sciphy_nni_up_tape = &tape_nni_up;
            sciphy_nni_down_tape = &tape_nni_down;
        }
        sciphy_nni_active = true;
        double neg = 0.0, ferr = 0.0;
        const double initial_neg = computeFunction(orig_len);
        const double optx = minimizeOneDimen(params->min_branch_length, orig_len,
                                             params->max_branch_length, params->min_branch_length,
                                             &neg, &ferr);
        sciphy_nni_active = false;
        sciphy_nni_up = sciphy_nni_down = nullptr;
        sciphy_nni_up_tape = sciphy_nni_down_tape = nullptr;
        const bool keep_orig = (neg > initial_neg + 1e-8);
        optlen = keep_orig ? orig_len : optx;
        score  = -(keep_orig ? initial_neg : neg);

        if (checking()) {
            // old local partials at the same candidate, combined at both lengths
            const size_t nptn = aln->getNPattern();
            buildDownwardLocal(node2, node1, downChild);
            buildUpToCLocal(node1, node2, downChild, upParent);
            std::vector<double> new_ptn(nptn), old_ptn(nptn);
            for (double len : {orig_len, optlen}) {
                combineTape(tape_nni_up, tape_nni_down, len, new_ptn.data());
                combineVectors(upParent, downChild, len, old_ptn.data());
                checkTapes("nni", new_ptn.data(), old_ptn.data());
            }
        }
    } else if (!done_closed) {      // no local partials: exact logL at the original length
        central->length = central_back->length = orig_len;
        score = computeLikelihood();
    }

    if (nni_cache_check) {
        central->length = central_back->length = optlen;
        const double exact = computeLikelihood();
        if (std::fabs(exact - score) > 1e-6)
            std::cerr << std::setprecision(12)
                      << "[nni cache check] n1=" << node1->id << " n2=" << node2->id
                      << " cnt=" << cnt << " cached=" << score << " exact=" << exact
                      << " diff=" << (exact - score) << (cached ? "" : " (fallback)") << "\n";
        score = exact;
    }
    return score;
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
    for (int cnt = 0; cnt < 2; cnt++) if (constraintTree.isCompatible(nniMoves[cnt])) {
        // check mode overwrites the snapshot, so rebuild it first
        if (nni_cache_check) { sciphy_snapshot_valid = false; ensureSciPhySnapshot(); }

        rawSwap(cnt);
        double optlen;
        const double score = scoreNNICandidate(node1, node2, cnt, orig_central_len, optlen);

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

// <----- Strict-clock dating (--clock-date) ----->
// Ages a(v) = time before sampling: tips 0, founder clock_T. Branch (p -> v) has length a(p) - a(v);
// the origin edge (founder -> root_int) has length clock_T - a(root_int).

double PhyloTreeSciPhy::clockRate() const {
    const ModelSciPhy *sm = dynamic_cast<const ModelSciPhy*>(model);
    return sm ? sm->getClockRate() : 1.0;
}

void PhyloTreeSciPhy::clockSetRate(double r) {
    ModelSciPhy *sm = dynamic_cast<ModelSciPhy*>(model);
    if (sm) sm->setClockRate(r);
}

void PhyloTreeSciPhy::clockSetEdge(PhyloNode *a, PhyloNode *b, double len) {
    PhyloNeighbor *ab = (PhyloNeighbor*) a->findNeighbor(b);
    PhyloNeighbor *ba = (PhyloNeighbor*) b->findNeighbor(a);
    ab->length = ba->length = len;
}

void PhyloTreeSciPhy::clockCollect() {
    clock_age.assign(nodeNum, 0.0);
    clock_parent.assign(nodeNum, nullptr);
    clock_post.clear();
    PhyloNode *founder = (PhyloNode*) root;
    PhyloNode *root_int = (PhyloNode*) root->neighbors[0]->node;
    clock_age[founder->id] = clock_T;
    // iterative postorder from root_int
    std::vector<std::pair<PhyloNode*, PhyloNode*>> stack{{root_int, founder}};
    std::vector<std::pair<PhyloNode*, PhyloNode*>> order;
    while (!stack.empty()) {
        auto [v, p] = stack.back(); stack.pop_back();
        clock_parent[v->id] = p;
        order.push_back({v, p});
        FOR_NEIGHBOR_IT(v, p, it) stack.push_back({(PhyloNode*)(*it)->node, v});
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if (!it->first->isLeaf()) clock_post.push_back(it->first);
    clock_pre.assign(clock_post.rbegin(), clock_post.rend());
}

// Ages from the current (free) branch lengths: mean distance to the tips below each node,
// scaled so the root sits at root_frac * T; the clock rate is set to keep expected edits fixed.
void PhyloTreeSciPhy::clockInitAges(double root_frac) {
    const double eps = params->min_branch_length;
    std::vector<double> d(nodeNum, 0.0), ntip(nodeNum, 0.0);
    for (PhyloNode *v : clock_post) {
        double sum = 0.0, cnt = 0.0;
        FOR_NEIGHBOR_IT(v, clock_parent[v->id], it) {
            PhyloNode *c = (PhyloNode*)(*it)->node;
            const double nc = c->isLeaf() ? 1.0 : ntip[c->id];
            sum += nc * (d[c->id] + (*it)->length);
            cnt += nc;
        }
        ntip[v->id] = cnt;
        d[v->id] = cnt > 0 ? sum / cnt : 0.0;
    }
    PhyloNode *root_int = (PhyloNode*) root->neighbors[0]->node;
    const double droot = std::max(d[root_int->id], eps);
    const double scale = root_frac * clock_T / droot;   // days per unit of free length
    for (PhyloNode *v : clock_post) {
        double lo = 0.0;
        FOR_NEIGHBOR_IT(v, clock_parent[v->id], it) lo = std::max(lo, clock_age[(*it)->node->id] + eps);
        clock_age[v->id] = std::min(std::max(d[v->id] * scale, lo), clock_T - eps);
    }
    clockSetRate(clockRate() / scale);
    clockApplyAllAges();
}

void PhyloTreeSciPhy::clockApplyAge(PhyloNode *v) {
    const double eps = params->min_branch_length;
    PhyloNode *p = clock_parent[v->id];
    clockSetEdge(p, v, std::max(clock_age[p->id] - clock_age[v->id], eps));
    FOR_NEIGHBOR_IT(v, p, it) {
        PhyloNode *c = (PhyloNode*)(*it)->node;
        clockSetEdge(v, c, std::max(clock_age[v->id] - clock_age[c->id], eps));
    }
}

void PhyloTreeSciPhy::clockApplyAllAges() {
    for (PhyloNode *v : clock_post) clockApplyAge(v);
}

// logL as a function of the age of clock_age_node. Exact: full recompute. Local: Down(v) rebuilt
// from the cached children partials, combined with the cached UpToC(parent) (snapshot).
double PhyloTreeSciPhy::clockAgeObjective(double a) {
    PhyloNode *v = clock_age_node;
    PhyloNode *p = clock_parent[v->id];
    clock_age[v->id] = a;
    clockApplyAge(v);
    if (clock_age_exact) return computeLikelihood();
    const double len = std::max(clock_age[p->id] - a, params->min_branch_length);
    PhyloNeighbor *up_nei = (PhyloNeighbor*) v->findNeighbor(p);   // UpToC(p), excluding v's subtree
    if (useOldKernel()) {
        std::vector<SciPhyAncestorPartials> down;
        buildDownwardLocal(v, p, down);
        if (down.empty() || !up_nei->sciphy_partial) return computeLikelihood();
        return combineVectors(*up_nei->sciphy_partial, down, len);
    }
    SciPhyTapePartials &down = tape_nni_down;   // scratch, as in NNI scoring
    buildTapeDownwardLocal(v, p, down);
    if (down.head.empty() || !up_nei->sciphy_tape_partial) return computeLikelihood();
    return combineTape(*up_nei->sciphy_tape_partial, down, len);
}

double PhyloTreeSciPhy::clockSweep(bool exact) {
    const double eps = params->min_branch_length;
    const bool dbg = std::getenv("IQTREE_SCIPHY_CHECK") != nullptr;
    if (!exact) { sciphy_snapshot_valid = false; ensureSciPhySnapshot(); }
    for (PhyloNode *v : clock_post) {
        PhyloNode *p = clock_parent[v->id];
        double lo = eps;
        FOR_NEIGHBOR_IT(v, p, it) lo = std::max(lo, clock_age[(*it)->node->id] + eps);
        const double hi = clock_age[p->id] - eps;
        const double cur = clock_age[v->id];
        if (hi <= lo) continue;
        if (dbg && !exact) {   // local objective vs full recompute on a fresh snapshot
            sciphy_snapshot_valid = false; ensureSciPhySnapshot();
            clock_age_node = v; clock_age_exact = false;
            const double loc = clockAgeObjective(cur);
            const double ex = computeLikelihood();
            if (std::fabs(loc - ex) > 1e-6)
                std::cerr << std::setprecision(12) << "[clock age check] node " << v->id
                          << " local " << loc << " exact " << ex << " diff " << (loc - ex) << "\n";
            sciphy_snapshot_valid = false; ensureSciPhySnapshot();
        }
        clock_age_node = v; clock_age_exact = exact; clock_age_active = true;
        const double init_neg = computeFunction(cur);
        double neg = 0.0, ferr = 0.0;
        double opt = minimizeOneDimen(lo, std::min(std::max(cur, lo), hi), hi, 1e-5, &neg, &ferr);
        clock_age_active = false;
        if (neg > init_neg) opt = cur;
        clock_age[v->id] = opt;
        clockApplyAge(v);
        if (!exact) {   // refresh Down(v) so v's parent sees its new subtree
            PhyloNeighbor *pv = (PhyloNeighbor*) p->findNeighbor(v);
            if (useOldKernel()) {
                std::vector<SciPhyAncestorPartials> down;
                buildDownwardLocal(v, p, down);
                if (!down.empty() && pv->sciphy_partial) *pv->sciphy_partial = down;
            } else {
                SciPhyTapePartials down;
                buildTapeDownwardLocal(v, p, down);
                if (!down.head.empty() && pv->sciphy_tape_partial) *pv->sciphy_tape_partial = down;
            }
        }
    }
    return computeLikelihood();
}

double PhyloTreeSciPhy::clockOptimizeRate() {
    const double cur = clockRate();
    clock_rate_active = true;
    const double init_neg = computeFunction(cur);
    double neg = 0.0, ferr = 0.0;
    double opt = minimizeOneDimen(1e-6, cur, 1e3, 1e-7, &neg, &ferr);
    clock_rate_active = false;
    if (neg > init_neg) opt = cur;
    clockSetRate(opt);
    return computeLikelihood();
}

// Move along the rate-age ridge: all internal ages x s, clock rate / s.
double PhyloTreeSciPhy::clockOptimizeScale() {
    const double eps = params->min_branch_length;
    PhyloNode *root_int = (PhyloNode*) root->neighbors[0]->node;
    clock_scale_base = clock_age;
    clock_scale_rate = clockRate();
    double smin = 0.0;   // keep every internal branch >= eps after scaling
    for (PhyloNode *v : clock_post) if (clock_scale_base[v->id] > 0) smin = std::max(smin, eps / clock_scale_base[v->id]);
    const double smax = (clock_T - eps) / clock_scale_base[root_int->id];
    clock_scale_active = true;
    const double init_neg = computeFunction(1.0);
    double neg = 0.0, ferr = 0.0;
    double opt = minimizeOneDimen(std::max(smin, 0.05), 1.0, smax, 1e-7, &neg, &ferr);
    if (neg > init_neg) opt = 1.0;
    computeFunction(opt);
    clock_scale_active = false;
    return computeLikelihood();
}

double PhyloTreeSciPhy::clockOptimize(int max_sweeps, double tol) {
    double ll = computeLikelihood();
    for (int round = 0; round < 20; ++round) {   // joint L-BFGS-B until it stops gaining
        const double t0 = getRealTime();
        const double l2 = clockLBFGS();
        if (verbose_mode >= VB_MED)
            std::cout << "Clock dating L-BFGS-B: logL " << l2 << " rate " << clockRate() << " root age "
                      << clock_age[root->neighbors[0]->node->id] << " (" << getRealTime() - t0 << " s)" << std::endl;
        const bool done = (l2 - ll < tol);
        ll = std::max(ll, l2);
        if (done) break;
    }
    for (int it = 0; it < max_sweeps; ++it) {   // coordinate polish
        const double prev = ll;
        const double t0 = getRealTime();
        ll = clockOptimizeScale();
        ll = clockOptimizeRate();
        const double t1 = getRealTime();
        const std::vector<double> saved = clock_age;
        double ls = clockSweep(false);
        if (verbose_mode >= VB_MED)
            std::cout << "  (scale+rate " << t1 - t0 << " s, ages " << getRealTime() - t1 << " s)" << std::endl;
        if (ls < ll - 1e-6) {   // the snapshot sweep went wrong: undo, redo exactly
            if (verbose_mode >= VB_MED)
                std::cout << "Clock dating: snapshot sweep lowered logL (" << ll << " -> " << ls
                          << "), redoing it exactly" << std::endl;
            clock_age = saved; clockApplyAllAges();
            ls = clockSweep(true);
        }
        ll = ls;
        if (verbose_mode >= VB_MED)
            std::cout << "Clock dating sweep " << it + 1 << ": logL " << ll << " rate " << clockRate()
                      << " root age " << clock_age[root->neighbors[0]->node->id] << std::endl;
        if (ll - prev < tol) break;
    }
    return ll;
}

void PhyloTreeSciPhy::runClockDating(const std::string &prefix) {
    if (!hasBarcodeAlignment() || !rooted || !root || root->neighbors.empty()) {
        outWarning("--clock-date needs a rooted SciPhy (barcode) tree; skipped");
        return;
    }
    const double t_start = getRealTime();
    const ModelSciPhy *sm = dynamic_cast<const ModelSciPhy*>(model);
    clock_T = sm ? sm->getOriginTime() : 25.0;
    if (rooted) computeBranchDirection();
    DoubleVector free_len;
    saveBranchLengths(free_len);
    const double free_rate = clockRate();
    const double ll_free = computeLikelihood();

    cout << endl << "Clock dating (strict clock, founder " << clock_T << " before sampling)..." << endl;
    clockCollect();
    // a few root-age starts, refined briefly; keep the best and finish it
    double best_ll = -std::numeric_limits<double>::infinity();
    std::vector<double> best_age; double best_rate = free_rate;
    for (double frac : {0.5, 0.8, 0.95}) {
        restoreBranchLengths(free_len); clockSetRate(free_rate);
        clockInitAges(frac);
        computeLikelihood();
        clockOptimizeScale();
        const double ll = clockOptimizeRate();
        if (ll > best_ll) { best_ll = ll; best_age = clock_age; best_rate = clockRate(); }
    }
    clock_age = best_age; clockSetRate(best_rate); clockApplyAllAges();
    const double ll_clock = clockOptimize(20, 1e-3);
    const double root_age = clock_age[root->neighbors[0]->node->id];

    const string tfile = prefix + ".timetree.nwk";
    printTree(tfile.c_str(), WT_BR_LEN | WT_BR_LEN_FIXED_WIDTH | WT_SORT_TAXA | WT_NEWLINE);
    ofstream out((prefix + ".clockdate.txt").c_str());
    out << std::setprecision(10) << "clock_rate\t" << clockRate() << "\nroot_age\t" << root_age
        << "\norigin_time\t" << clock_T << "\nlogL_clock\t" << ll_clock << "\nlogL_free\t" << ll_free
        << "\ndating_sec\t" << getRealTime() - t_start << "\n";
    out.close();
    cout << "Clock rate: " << clockRate() << "  root age: " << root_age << "  logL (clock): " << ll_clock
         << "  logL (free branch lengths): " << ll_free << endl;
    cout << "Time tree (branch lengths in time units) written to " << tfile << endl;
    cout << "Clock dating wall-clock time: " << getRealTime() - t_start << " sec" << endl;

    restoreBranchLengths(free_len);
    clockSetRate(free_rate);
    computeLikelihood();
}

void PhyloTreeSciPhy::clockSetVars(const double *x) {
    const size_t K = clock_pre.size();
    for (size_t i = 0; i < K; ++i) {
        PhyloNode *v = clock_pre[i];
        clock_age[v->id] = x[i] * clock_age[clock_parent[v->id]->id];
    }
    clockSetRate(std::exp(x[K]));
    clockApplyAllAges();
}

double PhyloTreeSciPhy::optimFunc(int nvar, double *vars) {
    if (!clock_lbfgs_active) return IQTree::optimFunc(nvar, vars);
    clockSetVars(vars);
    return -computeLikelihood();
}

// d logL / d length of edge (p -> c), central difference on the snapshot combine
double PhyloTreeSciPhy::clockEdgeDeriv(PhyloNode *p, PhyloNode *c) {
    PhyloNeighbor *down = (PhyloNeighbor*) p->findNeighbor(c);
    PhyloNeighbor *up = (PhyloNeighbor*) c->findNeighbor(p);
    const bool old = useOldKernel();
    if (old ? (!down->sciphy_partial || !up->sciphy_partial)
            : (!down->sciphy_tape_partial || !up->sciphy_tape_partial)) return 0.0;
    const double len = down->length;
    if (!old && !brlen_brent) {   // analytic derivative from the closed form
        buildBranchCoeffs(*up->sciphy_tape_partial, *down->sciphy_tape_partial, brlen_bc);
        bool ok;
        double d1;
        branchLogL(brlen_bc, len, &d1, nullptr, ok);
        if (ok) return d1;
    }
    const double h = std::max(1e-7, 1e-4 * len);
    const double lo = std::max(len - h, 0.0);
    auto f = [&](double l) {
        return old ? combineVectors(*up->sciphy_partial, *down->sciphy_partial, l)
                   : combineTape(*up->sciphy_tape_partial, *down->sciphy_tape_partial, l);
    };
    const double fp = f(len + h);
    const double fm = f(lo);
    return (fp - fm) / (len + h - lo);
}

double PhyloTreeSciPhy::optimGradient(int nvar, double *x, double *g) {
    if (!clock_lbfgs_active) return IQTree::optimGradient(nvar, x, g);
    clockSetVars(x);
    sciphy_snapshot_valid = false;
    ensureSciPhySnapshot();
    const double ll = curScore;
    // dL/da_u = sum_children dL/dlen(u,c) - dL/dlen(parent,u); dL/dlograte = sum_edges len * dL/dlen
    std::vector<double> dA(nodeNum, 0.0);
    double dlog_rate = 0.0;
    for (PhyloNode *u : clock_post) {
        PhyloNode *p = clock_parent[u->id];
        const double dpe = clockEdgeDeriv(p, u);
        dA[u->id] -= dpe;
        dlog_rate += ((PhyloNeighbor*) p->findNeighbor(u))->length * dpe;
        FOR_NEIGHBOR_IT(u, p, it) {
            PhyloNode *c = (PhyloNode*)(*it)->node;
            if (!c->isLeaf()) continue;   // internal children are handled as u themselves
            const double dc = clockEdgeDeriv(u, c);
            dA[u->id] += dc;
            dlog_rate += (*it)->length * dc;
        }
    }
    // internal child edges: dL/dlen(u,c) adds to dA[u] (subtracted from dA[c] above)
    for (PhyloNode *c : clock_post) {
        PhyloNode *u = clock_parent[c->id];
        if (u == root) continue;
        dA[u->id] += clockEdgeDeriv(u, c);
    }
    // dL/dr_v = (1/r_v) * sum over internal u in subtree(v) of a_u * dL/da_u
    std::vector<double> S(nodeNum, 0.0);
    for (PhyloNode *u : clock_post) {
        S[u->id] += clock_age[u->id] * dA[u->id];
        PhyloNode *p = clock_parent[u->id];
        if (p != root) S[p->id] += S[u->id];
    }
    const size_t K = clock_pre.size();
    for (size_t i = 0; i < K; ++i) g[i] = -S[clock_pre[i]->id] / x[i];
    g[K] = -dlog_rate;
    return -ll;
}

double PhyloTreeSciPhy::clockLBFGS() {
    const size_t K = clock_pre.size();
    std::vector<double> x(K + 1), lo(K + 1), hi(K + 1);
    for (size_t i = 0; i < K; ++i) {
        PhyloNode *v = clock_pre[i];
        x[i] = clock_age[v->id] / clock_age[clock_parent[v->id]->id];
        lo[i] = 1e-6; hi[i] = 1.0 - 1e-6;
        x[i] = std::min(std::max(x[i], lo[i]), hi[i]);
    }
    x[K] = std::log(clockRate()); lo[K] = std::log(1e-6); hi[K] = std::log(1e3);
    clock_lbfgs_active = true;
    if (std::getenv("IQTREE_SCIPHY_CHECK")) {   // analytic vs numerical gradient
        std::vector<double> g(K + 1), xx = x;
        optimGradient((int)K + 1, xx.data(), g.data());
        for (size_t i = 0; i <= K; i += std::max<size_t>(1, K / 8)) {
            const double h = 1e-6, xi = xx[i];
            xx[i] = xi + h; const double fp = optimFunc((int)K + 1, xx.data());
            xx[i] = xi - h; const double fm = optimFunc((int)K + 1, xx.data());
            xx[i] = xi;
            const double num = (fp - fm) / (2 * h);
            if (std::fabs(num - g[i]) > 1e-3 * std::max(1.0, std::fabs(num)))
                std::cerr << "[clock grad check] var " << i << " analytic " << g[i] << " numeric " << num << "\n";
        }
        { const double h = 1e-6, xi = xx[K];
          xx[K] = xi + h; const double fp = optimFunc((int)K + 1, xx.data());
          xx[K] = xi - h; const double fm = optimFunc((int)K + 1, xx.data()); xx[K] = xi;
          const double num = (fp - fm) / (2 * h);
          if (std::fabs(num - g[K]) > 1e-3 * std::max(1.0, std::fabs(num)))
              std::cerr << "[clock grad check] log rate analytic " << g[K] << " numeric " << num << "\n"; }
        std::cerr << "[clock grad check] done (" << K + 1 << " variables)\n";
    }
    L_BFGS_B((int)K + 1, x.data(), lo.data(), hi.data(), 1e-6, 2000);
    clock_lbfgs_active = false;
    clockSetVars(x.data());
    return computeLikelihood();
}

// <----- Prefix-depth kernel ----->
//
// Every possible state of a tape at a node is a prefix of the longest common prefix of
// the tip tapes below it, so a partial is (representative tip tape, depth d, d+1 numbers).
// The transition from depth i to depth j over a branch with mean mu = clock * length is
//   T(i, j) = Pois(j-i; mu) * F[j] / F[i]          (child not full)
//   T(i, L) = P(K >= L-i; mu) * F[L] / F[i]        (child full: absorbing)
// with F the prefix products of the representative's insert probabilities. The Poisson
// table is built once per branch and shared by all tapes.

namespace {

typedef PhyloTreeSciPhy PTS;

// Poisson(mu) pmf and upper tails for 0..L. Linear unless an entry would underflow
// (extreme mu, e.g. Brent probes of the clock rate); then all entries are logs.
typedef PTS::BranchTable BranchTable;

// Linear results below LH_UNDERFLOW are recomputed in log space (exact, like the old
// kernel); a tape is renormalised once its max drops below RENORM_BELOW.
const double LH_UNDERFLOW = 1e-290;
const double RENORM_BELOW = 1e-150;
// Poisson table: linear while mu < MU_LINEAR_MAX and both ends of the pmf exceed
// POIS_TINY (exp(-mu) stays representable); the tail sum stops at relative TAIL_REL_EPS
// (linear) or TAIL_LOG_GAP (log space).
const double MU_LINEAR_MAX = 600.0;
const double POIS_TINY = 1e-280;
const double TAIL_REL_EPS = 1e-17;
const double TAIL_LOG_GAP = 40.0;

void makeBranchTable(double mu, int L, BranchTable &bt) {
    bt.log_space = false;
    if (!(mu > 0.0)) {
        for (int m = 0; m <= L; ++m) bt.pois[m] = bt.tail[m] = 0.0;
        bt.pois[0] = bt.tail[0] = 1.0;
        return;
    }
    if (mu < MU_LINEAR_MAX) {
        bt.pois[0] = std::exp(-mu);
        for (int m = 1; m <= L; ++m) bt.pois[m] = bt.pois[m - 1] * mu / m;
        // pmf is unimodal: its smallest entry is at an end
        if (bt.pois[0] > POIS_TINY && bt.pois[L] > POIS_TINY) {
            double term = bt.pois[L], far = 0.0;
            for (int m = L + 1; ; ++m) {
                term *= mu / m;
                far += term;
                if (m > mu && term <= TAIL_REL_EPS * far) break;
            }
            bt.tail[L] = far + bt.pois[L];
            for (int r = L - 1; r >= 1; --r) bt.tail[r] = bt.tail[r + 1] + bt.pois[r];
            bt.tail[0] = 1.0;
            return;
        }
    }
    bt.log_space = true;
    const double lmu = std::log(mu);
    for (int m = 0; m <= L; ++m) bt.pois[m] = m * lmu - mu - std::lgamma(m + 1.0);
    double lterm = bt.pois[L], far = -std::numeric_limits<double>::infinity();
    for (int m = L + 1; ; ++m) {
        lterm += lmu - std::log((double)m);
        far = logAdd(far, lterm);
        if (m > mu && lterm < far - TAIL_LOG_GAP) break;
    }
    bt.tail[L] = logAdd(far, bt.pois[L]);
    for (int r = L - 1; r >= 1; --r) bt.tail[r] = logAdd(bt.tail[r + 1], bt.pois[r]);
    bt.tail[0] = 0.0;
}

// log of T(i, j) without the insert factor
inline double logPoisTerm(const BranchTable &bt, int L, int i, int j) {
    const double x = (j == L) ? bt.tail[L - i] : bt.pois[j - i];
    return bt.log_space ? x : std::log(x);
}

// exp(lv - max) into out; returns max (-inf, and out = 0, if every entry is -inf)
double fromLogs(const double *lv, int n, double *out) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < n; ++k) if (lv[k] > mx) mx = lv[k];
    for (int k = 0; k < n; ++k)
        out[k] = (mx == -std::numeric_limits<double>::infinity()) ? 0.0 : std::exp(lv[k] - mx);
    return mx;
}

// lv[i] = log sum_{j=i..dc} T(i,j) child[j], i = 0..dp
void transDownLogs(const BranchTable &bt, int L, const double *F, const unsigned char *Z,
                   const double *child, int dc, int dp, double *lv) {
    for (int i = 0; i <= dp; ++i) {
        double lse = -std::numeric_limits<double>::infinity();
        for (int j = i; j <= dc && Z[j] == Z[i]; ++j) {
            if (child[j] <= 0.0) continue;
            lse = logAdd(lse, logPoisTerm(bt, L, i, j) + std::log(F[j]) - std::log(F[i])
                              + std::log(child[j]));
        }
        lv[i] = lse;
    }
}

// lv[j] = log sum_{i=0..min(j,du)} up[i] T(i,j), j = 0..dc
void transUpLogs(const BranchTable &bt, int L, const double *F, const unsigned char *Z,
                 const double *up, int du, int dc, double *lv) {
    for (int j = 0; j <= dc; ++j) {
        double lse = -std::numeric_limits<double>::infinity();
        for (int i = std::min(j, du); i >= 0 && Z[i] == Z[j]; --i) {
            if (up[i] <= 0.0) continue;
            lse = logAdd(lse, logPoisTerm(bt, L, i, j) + std::log(F[j]) - std::log(F[i])
                              + std::log(up[i]));
        }
        lv[j] = lse;
    }
}

// out[i] = sum_{j=i..dc} T(i,j) child[j] for i = 0..dp (dp <= dc), times exp(-returned shift).
// F, Z: prefix products / zero counts of the child's representative tape of length L.
double transDown(const BranchTable &bt, int L, const double *F, const unsigned char *Z,
                 const double *child, int dc, int dp, double *out) {
    if (bt.log_space) {
        double lv[PTS::MAX_TAPE_SITES + 1];
        transDownLogs(bt, L, F, Z, child, dc, dp, lv);
        return fromLogs(lv, dp + 1, out);
    }
    double g[PTS::MAX_TAPE_SITES + 1];
    for (int j = 0; j <= dc; ++j) g[j] = F[j] * child[j];
    const int jp = (dc == L) ? L - 1 : dc;   // last non-full child state
    for (int i = 0; i <= dp; ++i) {
        double s = 0.0;
        int j = i;
        for (; j <= jp && Z[j] == Z[i]; ++j) s += bt.pois[j - i] * g[j];
        if (dc == L && j == L && Z[L] == Z[i]) s += bt.tail[L - i] * g[L];
        out[i] = s / F[i];
    }
    return 0.0;
}

// out[j] = sum_{i=0..min(j,du)} up[i] T(i,j) for j = 0..dc, times exp(-returned shift).
double transUp(const BranchTable &bt, int L, const double *F, const unsigned char *Z,
               const double *up, int du, int dc, double *out) {
    if (bt.log_space) {
        double lv[PTS::MAX_TAPE_SITES + 1];
        transUpLogs(bt, L, F, Z, up, du, dc, lv);
        return fromLogs(lv, dc + 1, out);
    }
    double h[PTS::MAX_TAPE_SITES + 1];
    for (int i = 0; i <= du; ++i) h[i] = up[i] / F[i];
    for (int j = 0; j <= dc; ++j) {
        const double *P = (j == L) ? bt.tail + L : bt.pois + j;   // P[-i]
        double s = 0.0;
        for (int i = std::min(j, du); i >= 0 && Z[i] == Z[j]; --i) s += h[i] * P[-i];
        out[j] = s * F[j];
    }
    return 0.0;
}

// lh[0..d] /= max; returns log(max), -inf if all zero
double normalizeTape(double *lh, int d) {
    double mx = 0.0;
    for (int k = 0; k <= d; ++k) if (lh[k] > mx) mx = lh[k];
    if (!(mx > 0.0)) return -std::numeric_limits<double>::infinity();
    const double inv = 1.0 / mx;
    for (int k = 0; k <= d; ++k) lh[k] *= inv;
    return std::log(mx);
}

// log(sum lh) + log_scale (just log_scale for a wildcard)
double totalTapeMass(const SciPhyTapeHead &h, const double *lh) {
    if (h.wildcard) return h.log_scale;
    double s = 0.0;
    for (int k = 0; k <= h.depth; ++k) if (lh[k] > 0.0) s += lh[k];
    return (s > 0.0) ? std::log(s) + h.log_scale : -std::numeric_limits<double>::infinity();
}

void setWildcard(SciPhyTapeHead &h, double log_scale) {
    h.wildcard = true;
    h.rep = -1;
    h.depth = -1;
    h.log_scale = log_scale;
}

// Normalises every non-wildcard tape of out; tapes flagged in redo are handed to
// redoTape(p, head, lh) instead, to be recomputed in log space.
template <class RedoTape>
void finishTapes(SciPhyTapePartials &out, const std::vector<size_t> &offset,
                 const std::vector<char> &redo, RedoTape redoTape) {
    for (size_t p = 0; p < out.head.size(); ++p) {
        SciPhyTapeHead &h = out.head[p];
        if (h.wildcard) continue;
        double *lh = &out.lh[offset[p]];
        if (redo[p]) redoTape((int)p, h, lh);
        else h.log_scale += normalizeTape(lh, h.depth);
    }
}

}  // anonymous namespace

bool PhyloTreeSciPhy::hasPartial(PhyloNeighbor *nei) const {
    return useOldKernel() ? nei->sciphy_partial != nullptr : nei->sciphy_tape_partial != nullptr;
}

int PhyloTreeSciPhy::patternTapeLength(size_t p) const {
    const BarcodeAlignment &barcode_aln = barcodeAlignment();
    for (size_t c = 0; c < aln->getNSeq(); ++c) {
        const int obs = (int)aln->at(p)[c];
        if (obs >= 0 && obs < barcode_aln.getNumTapeStates())
            return (int)barcode_aln.getTapeVec(obs).size();
    }
    return 0;
}

void PhyloTreeSciPhy::initTapeKernel() {
    if (!tape_lh_offset.empty()) return;
    const BarcodeAlignment &barcode_aln = barcodeAlignment();
    const std::vector<double> &edit_probs = sciphyModel().getEditProbs();
    const size_t nptn = aln->getNPattern();

    tape_len.assign(nptn, 0);
    tape_lh_offset.assign(nptn + 1, 0);
    max_tape_len = 0;
    for (size_t p = 0; p < nptn; ++p) {
        tape_len[p] = patternTapeLength(p);
        if (tape_len[p] > MAX_TAPE_SITES)
            outError("SciPhy: tapes longer than " + convertIntToString(MAX_TAPE_SITES) +
                     " sites are not supported");
        max_tape_len = std::max(max_tape_len, tape_len[p]);
        tape_lh_offset[p + 1] = tape_lh_offset[p] + tape_len[p] + 1;
    }

    // insert-probability prefix products per tip tape state (same factor as logTransProbVec)
    const int nstates = barcode_aln.getNumTapeStates();
    const int M = std::max(1, barcode_aln.getMaxInsertionType());
    prefix_offset.assign(nstates + 1, 0);
    prefix_prod.clear();
    prefix_zeros.clear();
    for (int s = 0; s < nstates; ++s) {
        prefix_prod.push_back(1.0);
        prefix_zeros.push_back(0);
        // the kernel assumes edits fill sites left to right: scars, then zeros
        const std::vector<int> &tape = barcode_aln.getTapeVec(s);
        bool seen_zero = false;
        for (int scar : tape) {
            if (scar == 0) seen_zero = true;
            else if (seen_zero)
                outError("SciPhy: a tape has an unedited site before an edited one; "
                         "the prefix-depth kernel needs scars to fill sites left to right");
        }
        for (int scar : tape) {
            if (scar == 0) continue;
            const double pe = insertProb(edit_probs, M, scar);
            const bool zero = !(pe > 0.0);
            prefix_prod.push_back(prefix_prod.back() * (zero ? 1.0 : pe));
            prefix_zeros.push_back(prefix_zeros.back() + (zero ? 1 : 0));
        }
        prefix_offset[s + 1] = prefix_prod.size();
    }
}

int PhyloTreeSciPhy::commonDepth(int a, int b, int bound) const {
    if (a == b) return bound;
    const std::vector<int> &u = barcodeAlignment().getTapeVec(a);
    const std::vector<int> &v = barcodeAlignment().getTapeVec(b);
    ASSERT(u.size() == v.size());
    // prefixes agree up to the first site where the tapes differ
    int k = 0;
    for (size_t q = 0; q < u.size() && k < bound; ++q) {
        if (u[q] != v[q]) break;
        if (u[q] != 0) ++k;
    }
    return k;
}

void PhyloTreeSciPhy::fillBranchTable(double len, BranchTable &bt) const {
    makeBranchTable(std::max(0.0, sciphyModel().getClockRate() * len), max_tape_len, bt);
}

void PhyloTreeSciPhy::sizeTapePartials(SciPhyTapePartials &tp) const {
    tp.head.resize(aln->getNPattern());
    tp.lh.resize(tape_lh_offset.back());
}

SciPhyTapePartials &PhyloTreeSciPhy::tapePartials(PhyloNeighbor *nei) const {
    if (!nei->sciphy_tape_partial) nei->sciphy_tape_partial = new SciPhyTapePartials();
    sizeTapePartials(*nei->sciphy_tape_partial);
    return *nei->sciphy_tape_partial;
}

// lh[i] *= sum_j T(i,j) child[j] over one child branch; a product that underflows
// flags tape p for a log-space redo.
void PhyloTreeSciPhy::multiplyChild(const BranchTable &bt, int p, const SciPhyTapeHead &c,
                                    const double *child_lh, SciPhyTapeHead &h, double *lh,
                                    std::vector<char> &redo) const {
    double v[MAX_TAPE_SITES + 1];
    h.log_scale += transDown(bt, tape_len[p], prefixProd(c.rep), prefixZeros(c.rep),
                             child_lh, c.depth, h.depth, v);
    double mx = 0.0;
    for (int i = 0; i <= h.depth; ++i) { lh[i] *= v[i]; mx = std::max(mx, lh[i]); }
    if (mx < LH_UNDERFLOW) redo[p] = 1;
    else if (mx < RENORM_BELOW) h.log_scale += normalizeTape(lh, h.depth);
}

// Support = prefixes of the common prefix of the observed (non-wildcard) kids' tapes;
// rep = -1 if every kid is a wildcard.
template <class HeadOf>
void PhyloTreeSciPhy::commonSupport(const std::vector<PhyloNeighbor*> &kids, HeadOf headOf,
                                    int &rep, int &depth) const {
    rep = -1;
    depth = -1;
    for (PhyloNeighbor *k : kids) {
        const SciPhyTapeHead &c = headOf(k);
        if (c.wildcard) continue;
        if (rep < 0) { rep = c.rep; depth = c.depth; }
        else depth = commonDepth(rep, c.rep, std::min(depth, (int)c.depth));
    }
}

// Down partials below dad_branch, by height level; nodes in a level run in parallel.
void PhyloTreeSciPhy::computeTapeDownward(PhyloNeighbor *dad_branch, PhyloNode *dad) {
    struct Item { PhyloNeighbor *edge; PhyloNode *dad; int height; };
    std::vector<Item> order;                       // pre-order
    std::vector<std::pair<PhyloNeighbor*, PhyloNode*>> stack{{dad_branch, dad}};
    std::vector<int> parent_of;                    // index of the parent item
    std::vector<int> pstack{-1};
    while (!stack.empty()) {
        auto [e, d] = stack.back(); stack.pop_back();
        const int par = pstack.back(); pstack.pop_back();
        order.push_back({e, d, 0});
        parent_of.push_back(par);
        const int me = (int)order.size() - 1;
        PhyloNode *node = (PhyloNode*) e->node;
        if (e->partial_lh_computed & 1) continue;  // cached: nothing to recompute below
        FOR_NEIGHBOR_IT(node, d, it) { stack.push_back({(PhyloNeighbor*) *it, node}); pstack.push_back(me); }
    }
    for (int i = (int)order.size() - 1; i >= 0; --i)      // children before parents
        if (parent_of[i] >= 0)
            order[parent_of[i]].height = std::max(order[parent_of[i]].height, order[i].height + 1);
    int max_h = 0;
    for (const Item &it : order) max_h = std::max(max_h, it.height);
    std::vector<std::vector<int>> levels(max_h + 1);
    for (int i = 0; i < (int)order.size(); ++i) levels[order[i].height].push_back(i);
    const int nt = numWorkThreads();
    workspace(nt - 1);
    for (const std::vector<int> &lev : levels) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt) if (nt > 1 && lev.size() > 16)
#endif
        for (int j = 0; j < (int)lev.size(); ++j) {
            const Item &it = order[lev[j]];
            if (it.edge->partial_lh_computed & 1) continue;
            Workspace &ws = threadWorkspace();
            computeTapeDownNode(it.edge, it.dad, ws);
        }
    }
}

void PhyloTreeSciPhy::computeTapeDownNode(PhyloNeighbor *dad_branch, PhyloNode *dad,
                                          Workspace &ws) const {
    if (dad_branch->partial_lh_computed & 1) {
        return;
    }
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    const int nptn = (int)aln->getNPattern();
    SciPhyTapePartials &out = tapePartials(dad_branch);

    if (node->isLeaf()) {
        // Dropout tip weights (constants): dropped -> q, observed -> 1-q; 1 if q unset.
        const ModelSciPhy &sm = sciphyModel();
        const bool use_q = sm.hasDropoutProb();
        const double log_q   = use_q ? std::log(sm.getDropoutProb()) : 0.0;
        const double log_1mq = use_q ? std::log(1.0 - sm.getDropoutProb()) : 0.0;
        const int nstates = barcodeAlignment().getNumTapeStates();
        for (int p = 0; p < nptn; ++p) {
            SciPhyTapeHead &h = out.head[p];
            const int obs = getObservedLeafState(node, p);
            if (obs == aln->STATE_UNKNOWN || obs < 0 || obs >= nstates) {
                setWildcard(h, log_q);   // dropped tape: present, identity unknown
                continue;
            }
            h.wildcard = false;
            h.rep = obs;
            h.depth = stateDepth(obs);
            h.log_scale = log_1mq;
            double *lh = &out.lh[tape_lh_offset[p]];
            for (int k = 0; k <= h.depth; ++k) lh[k] = (k == h.depth) ? 1.0 : 0.0;
        }
        dad_branch->partial_lh_computed |= 1;
        return;
    }

    std::vector<PhyloNeighbor*> kids;
    FOR_NEIGHBOR_IT(node, dad, it) kids.push_back((PhyloNeighbor*) *it);   // lower levels
    if (!kids.empty()) combineTapeChildren(kids, out, ws);
    dad_branch->partial_lh_computed |= 1;
}

void PhyloTreeSciPhy::combineTapeChildren(const std::vector<PhyloNeighbor*> &kids,
                                          SciPhyTapePartials &out, Workspace &ws) const {
    const int nptn = (int)aln->getNPattern();

    // support: prefixes of the observed children's common prefix; wildcard children
    // don't constrain it and add only a constant
    for (int p = 0; p < nptn; ++p) {
        SciPhyTapeHead &h = out.head[p];
        double scale = 0.0;
        for (PhyloNeighbor *k : kids) scale += k->sciphy_tape_partial->head[p].log_scale;
        int rep, depth;
        commonSupport(kids, [&](PhyloNeighbor *k) -> const SciPhyTapeHead & {
            return k->sciphy_tape_partial->head[p];
        }, rep, depth);
        if (rep < 0) { setWildcard(h, scale); continue; }
        h.wildcard = false;
        h.rep = rep;
        h.depth = depth;
        h.log_scale = scale;
        double *lh = &out.lh[tape_lh_offset[p]];
        for (int k = 0; k <= depth; ++k) lh[k] = 1.0;
    }

    // one Poisson table per child branch, shared by all tapes
    std::vector<char> &redo = ws.redo;
    redo.assign(nptn, 0);
    for (PhyloNeighbor *k : kids) {
        BranchTable bt;
        fillBranchTable(k->length, bt);
        const SciPhyTapePartials &cp = *k->sciphy_tape_partial;
        for (int p = 0; p < nptn; ++p) {
            SciPhyTapeHead &h = out.head[p];
            const SciPhyTapeHead &c = cp.head[p];
            if (h.wildcard || c.wildcard || redo[p]) continue;
            multiplyChild(bt, p, c, &cp.lh[tape_lh_offset[p]], h, &out.lh[tape_lh_offset[p]], redo);
        }
    }

    finishTapes(out, tape_lh_offset, redo, [&](int p, SciPhyTapeHead &h, double *lh) {
        h.log_scale = 0.0;
        for (PhyloNeighbor *k : kids) h.log_scale += k->sciphy_tape_partial->head[p].log_scale;
        h.log_scale += tapeLogs(p, h.rep, h.depth, nullptr, 0, 0.0, kids, nullptr, lh);
    });
}

double PhyloTreeSciPhy::tapeLogs(int p, int rep, int depth, const SciPhyTapePartials *up,
                                 int up_depth, double up_len,
                                 const std::vector<PhyloNeighbor*> &kids,
                                 const PhyloNeighbor *skip, double *lh) const {
    const int L = tape_len[p];
    const size_t off = tape_lh_offset[p];
    double acc[MAX_TAPE_SITES + 1], lv[MAX_TAPE_SITES + 1];
    for (int s = 0; s <= depth; ++s) acc[s] = 0.0;
    if (up) {
        BranchTable bt;
        fillBranchTable(up_len, bt);
        transUpLogs(bt, L, prefixProd(rep), prefixZeros(rep), &up->lh[off],
                    std::min(up_depth, depth), depth, lv);
        for (int s = 0; s <= depth; ++s) acc[s] += lv[s];
    }
    for (PhyloNeighbor *k : kids) {
        if (k == skip) continue;
        const SciPhyTapePartials &cp = *k->sciphy_tape_partial;
        const SciPhyTapeHead &c = cp.head[p];
        if (c.wildcard) continue;
        BranchTable bt;
        fillBranchTable(k->length, bt);
        transDownLogs(bt, L, prefixProd(c.rep), prefixZeros(c.rep), &cp.lh[off], c.depth, depth, lv);
        for (int s = 0; s <= depth; ++s) acc[s] += lv[s];
    }
    return fromLogs(acc, depth + 1, lh);
}

void PhyloTreeSciPhy::computeTapeFullUpward() {
    if (!rooted || !root || root->neighbors.empty()) return;
    PhyloNode *root_node = (PhyloNode*) root;
    PhyloNeighbor *origin_edge = (PhyloNeighbor*) root->neighbors[0];   // Down(root_int)
    PhyloNode *root_int = (PhyloNode*) origin_edge->node;
    if (!origin_edge->sciphy_tape_partial) return;                     // downward must exist
    const SciPhyTapePartials &downRoot = *origin_edge->sciphy_tape_partial;

    // Seed UpToC(founder) on (root_int -> root): founder = unedited (depth 0), mass 1.
    // seed.rep is copied for shape only; every consumer checks the down side's wildcard first.
    SciPhyTapePartials &seed = tapePartials((PhyloNeighbor*) root_int->findNeighbor(root_node));
    for (size_t p = 0; p < aln->getNPattern(); ++p) {
        SciPhyTapeHead &h = seed.head[p];
        h.wildcard = false;
        h.rep = downRoot.head[p].rep;
        h.depth = 0;
        h.log_scale = 0.0;
        seed.lh[tape_lh_offset[p]] = 1.0;
    }
    // by depth level from the founder; nodes in a level run in parallel
    const int nt = numWorkThreads();
    workspace(nt - 1);
    std::vector<std::pair<PhyloNode*, PhyloNode*>> level{{root_int, root_node}}, next;
    while (!level.empty()) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt) if (nt > 1 && level.size() > 16)
#endif
        for (int j = 0; j < (int)level.size(); ++j) {
            Workspace &ws = threadWorkspace();
            computeTapeUpwardNode(level[j].first, level[j].second, ws);
        }
        next.clear();
        for (auto &nd : level)
            FOR_NEIGHBOR_IT(nd.first, nd.second, it)
                if (!(*it)->node->isLeaf()) next.push_back({(PhyloNode*) (*it)->node, nd.first});
        level.swap(next);
    }
    sciphy_snapshot_valid = true;
}

// UpToC(node) on each (child -> node) half-edge of `node`.
void PhyloTreeSciPhy::computeTapeUpwardNode(PhyloNode *node, PhyloNode *dad, Workspace &ws) const {
    const int nptn = (int)aln->getNPattern();

    //   up_nei   = (node -> dad): holds UpToC(dad)   [function of dad's state]
    //   down_nei = (dad -> node): holds Down(node)   [function of node's state]
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) node->findNeighbor(dad);
    PhyloNeighbor *down_nei = (PhyloNeighbor*) dad->findNeighbor(node);
    if (!up_nei->sciphy_tape_partial || !down_nei->sciphy_tape_partial) return;
    const SciPhyTapePartials &upToDad  = *up_nei->sciphy_tape_partial;
    const SciPhyTapePartials &downNode = *down_nei->sciphy_tape_partial;

    // UpBar(node)[s] = sum_{s_dad} UpToC(dad)[s_dad] * T(s_dad -> s), s in Down(node)'s support
    SciPhyTapePartials &ub = ws.upbar;
    sizeTapePartials(ub);
    std::vector<char> &redo = ws.redo;
    {
        BranchTable bt;
        fillBranchTable(down_nei->length, bt);
        for (int p = 0; p < nptn; ++p) {
            const SciPhyTapeHead &ud = upToDad.head[p];
            const SciPhyTapeHead &dn = downNode.head[p];
            SciPhyTapeHead &h = ub.head[p];
            const size_t off = tape_lh_offset[p];
            // all-missing subtree or uninformative up side: collapses to the total up mass
            if (dn.wildcard || ud.wildcard) {
                setWildcard(h, totalTapeMass(ud, &upToDad.lh[off]));
                continue;
            }
            h.wildcard = false;
            h.rep = dn.rep;
            h.depth = dn.depth;
            h.log_scale = ud.log_scale;
            const double shift = transUp(bt, tape_len[p], prefixProd(dn.rep), prefixZeros(dn.rep),
                                         &upToDad.lh[off], std::min(ud.depth, dn.depth), dn.depth, &ub.lh[off]);
            const double lmax = normalizeTape(&ub.lh[off], h.depth);
            if (lmax < std::log(LH_UNDERFLOW))
                h.log_scale += tapeLogs(p, dn.rep, dn.depth, &upToDad, ud.depth, down_nei->length,
                                        std::vector<PhyloNeighbor*>(), nullptr, &ub.lh[off]);
            else
                h.log_scale += shift + lmax;
        }
    }

    std::vector<PhyloNeighbor*> childDown;
    std::vector<PhyloNode*> childNode;
    FOR_NEIGHBOR_IT(node, dad, it) {
        childDown.push_back((PhyloNeighbor*) *it);
        childNode.push_back((PhyloNode*) (*it)->node);
    }

    // UpToC(node) for child c = UpBar(node) * prod_{siblings d} sum_{s_d} T * Down(d),
    // stored on (c -> node).
    for (size_t ci = 0; ci < childDown.size(); ++ci) {
        PhyloNeighbor *c_up = (PhyloNeighbor*) childNode[ci]->findNeighbor(node);
        SciPhyTapePartials &out = tapePartials(c_up);
        std::copy(ub.head.begin(), ub.head.end(), out.head.begin());
        std::copy(ub.lh.begin(), ub.lh.end(), out.lh.begin());
        redo.assign(nptn, 0);
        for (size_t di = 0; di < childDown.size(); ++di) {
            if (di == ci) continue;
            const SciPhyTapePartials &dd = *childDown[di]->sciphy_tape_partial;
            BranchTable bt;
            fillBranchTable(childDown[di]->length, bt);
            for (int p = 0; p < nptn; ++p) {
                SciPhyTapeHead &h = out.head[p];
                const SciPhyTapeHead &d = dd.head[p];
                h.log_scale += d.log_scale;   // constant UpBar folds in all sibling constants
                if (h.wildcard || d.wildcard || redo[p]) continue;
                const size_t off = tape_lh_offset[p];
                multiplyChild(bt, p, d, &dd.lh[off], h, &out.lh[off], redo);
            }
        }
        finishTapes(out, tape_lh_offset, redo, [&](int p, SciPhyTapeHead &h, double *lh) {
            h.log_scale = upToDad.head[p].log_scale;
            for (size_t di = 0; di < childDown.size(); ++di)
                if (di != ci) h.log_scale += childDown[di]->sciphy_tape_partial->head[p].log_scale;
            h.log_scale += tapeLogs(p, h.rep, h.depth, &upToDad, upToDad.head[p].depth,
                                    down_nei->length, childDown, childDown[ci], lh);
        });
        c_up->partial_lh_computed |= 1;
    }
}

// Whole-tree logL from an up partial (parent state) and a down partial (child state)
// joined by a branch of length len.
double PhyloTreeSciPhy::combineTape(const SciPhyTapePartials &up, const SciPhyTapePartials &down,
                                    double len, double *pattern_lh_out) const {
    const size_t nptn = aln->getNPattern();
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    if (up.head.size() != nptn || down.head.size() != nptn) {
        return NEG_INF;
    }
    BranchTable bt;
    fillBranchTable(len, bt);
    double tree_lh = 0.0;
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyTapeHead &uc = up.head[p];     // s_parent
        const SciPhyTapeHead &dc = down.head[p];   // s_child
        const size_t off = tape_lh_offset[p];
        double lp;
        if (uc.wildcard && dc.wildcard) {
            lp = uc.log_scale + dc.log_scale;
        } else if (dc.wildcard) {
            // child side all missing: marginalize the up side
            lp = totalTapeMass(uc, &up.lh[off]) + dc.log_scale;
        } else if (uc.wildcard) {
            // up side uninformative (normally not reached): marginalize the down side
            lp = totalTapeMass(dc, &down.lh[off]) + uc.log_scale;
        } else {
            // parent states deeper than the child's support have T = 0
            const int dp = std::min(uc.depth, dc.depth);
            double v[MAX_TAPE_SITES + 1];
            const double shift = transDown(bt, tape_len[p], prefixProd(dc.rep), prefixZeros(dc.rep),
                                           &down.lh[off], dc.depth, dp, v);
            double s = 0.0;
            for (int i = 0; i <= dp; ++i) s += up.lh[off + i] * v[i];
            if (s >= LH_UNDERFLOW) {
                lp = std::log(s) + shift + uc.log_scale + dc.log_scale;
            } else {
                double lv[MAX_TAPE_SITES + 1];
                transDownLogs(bt, tape_len[p], prefixProd(dc.rep), prefixZeros(dc.rep),
                              &down.lh[off], dc.depth, dp, lv);
                double lse = NEG_INF;
                for (int i = 0; i <= dp; ++i)
                    if (up.lh[off + i] > 0.0) lse = logAdd(lse, std::log(up.lh[off + i]) + lv[i]);
                lp = (lse == NEG_INF) ? NEG_INF : lse + uc.log_scale + dc.log_scale;
            }
        }
        if (pattern_lh_out) pattern_lh_out[p] = lp;
        tree_lh += lp * ptn_freq[p];
    }
    return tree_lh;
}

// Founder (unedited, depth 0) -> root_int over the origin branch of length t.
double PhyloTreeSciPhy::originClosureTape(const SciPhyTapePartials &bio, double t,
                                          double *pattern_lh_out) {
    const int nptn = (int)aln->getNPattern();
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    BranchTable bt;
    fillBranchTable(t, bt);
    double tree_lh = 0.0;
    for (int p = 0; p < nptn; ++p) {   // sequential: fixed summation order
        const SciPhyTapeHead &b = bio.head[p];
        double lp;
        if (b.wildcard) {
            lp = b.log_scale;   // tape dropped in every cell: only the dropout constant
        } else {
            double v[MAX_TAPE_SITES + 1];
            const double shift = transDown(bt, tape_len[p], prefixProd(b.rep), prefixZeros(b.rep),
                                           &bio.lh[tape_lh_offset[p]], b.depth, 0, v);
            if (v[0] >= LH_UNDERFLOW) {
                lp = std::log(v[0]) + shift + b.log_scale;
            } else {
                double lv[MAX_TAPE_SITES + 1];
                transDownLogs(bt, tape_len[p], prefixProd(b.rep), prefixZeros(b.rep),
                              &bio.lh[tape_lh_offset[p]], b.depth, 0, lv);
                lp = (lv[0] == NEG_INF) ? NEG_INF : lv[0] + b.log_scale;
            }
        }
        if (pattern_lh_out) pattern_lh_out[p] = lp;
        tree_lh += lp * ptn_freq[p];
    }
    return tree_lh;
}

void PhyloTreeSciPhy::buildTapeDownwardLocal(PhyloNode *child, PhyloNode *parent,
                                             SciPhyTapePartials &out) {
    std::vector<PhyloNeighbor*> kids;
    FOR_NEIGHBOR_IT(child, parent, it) kids.push_back((PhyloNeighbor*) *it);
    buildTapeDownwardKids(kids, out, workspace(0));
}

void PhyloTreeSciPhy::buildTapeDownwardKids(const std::vector<PhyloNeighbor*> &kids,
                                            SciPhyTapePartials &out, Workspace &ws) const {
    out.head.clear();
    if (kids.empty()) return;
    for (PhyloNeighbor *k : kids)
        if (!k->sciphy_tape_partial) return;   // cache miss
    sizeTapePartials(out);
    combineTapeChildren(kids, out, ws);
}

void PhyloTreeSciPhy::buildTapeUpToCLocal(PhyloNode *parent, PhyloNode *child,
                                          const SciPhyTapePartials &child_down,
                                          SciPhyTapePartials &out) {
    out.head.clear();
    // parent's toward-root half-edge holds UpToC(grandparent or founder)
    PhyloNeighbor *pnei = nullptr;
    FOR_NEIGHBOR_IT(parent, nullptr, it)
        if (((PhyloNeighbor*) *it)->direction == TOWARD_ROOT) { pnei = (PhyloNeighbor*) *it; break; }
    if (!pnei) return;
    // parent's children: `child` plus its siblings
    std::vector<PhyloNeighbor*> kids;
    PhyloNeighbor *child_nei = nullptr;
    FOR_NEIGHBOR_IT(parent, nullptr, it) {
        PhyloNeighbor *k = (PhyloNeighbor*) *it;
        if (k == pnei) continue;
        kids.push_back(k);
        if (k->node == child) child_nei = k;
    }
    buildTapeUpToCKids(pnei, kids, child_nei, child_down, out, workspace(0));
}

void PhyloTreeSciPhy::buildTapeUpToCKids(PhyloNeighbor *pnei, const std::vector<PhyloNeighbor*> &kids,
                                         PhyloNeighbor *child_nei, const SciPhyTapePartials &child_down,
                                         SciPhyTapePartials &out, Workspace &ws) const {
    const int nptn = (int)aln->getNPattern();
    out.head.clear();
    if (!pnei || !pnei->sciphy_tape_partial) return;
    const SciPhyTapePartials &upP = *pnei->sciphy_tape_partial;
    if (kids.empty() || !child_nei || (int)child_down.head.size() != nptn) return;
    for (PhyloNeighbor *k : kids)
        if (!k->sciphy_tape_partial) return;
    auto kidHead = [&](PhyloNeighbor *k, int p) -> const SciPhyTapeHead & {
        return (k == child_nei) ? child_down.head[p] : k->sciphy_tape_partial->head[p];
    };
    sizeTapePartials(out);

    // support from the observed kids; all-missing (or uninformative up side) -> constant
    for (int p = 0; p < nptn; ++p) {
        SciPhyTapeHead &h = out.head[p];
        const SciPhyTapeHead &uP = upP.head[p];
        int rep, depth;
        commonSupport(kids, [&](PhyloNeighbor *k) -> const SciPhyTapeHead & { return kidHead(k, p); },
                      rep, depth);
        if (uP.wildcard || rep < 0) {
            double ls = totalTapeMass(uP, &upP.lh[tape_lh_offset[p]]);
            for (PhyloNeighbor *k : kids) {
                const SciPhyTapeHead &d = k->sciphy_tape_partial->head[p];
                if (k != child_nei && d.wildcard) ls += d.log_scale;
            }
            setWildcard(h, ls);
            continue;
        }
        h.wildcard = false;
        h.rep = rep;
        h.depth = depth;
        h.log_scale = uP.log_scale;
    }

    // UpBar(parent), then times each sibling's down contribution
    std::vector<char> &redo = ws.redo;
    redo.assign(nptn, 0);
    {
        BranchTable bt;
        fillBranchTable(pnei->length, bt);
        for (int p = 0; p < nptn; ++p) {
            SciPhyTapeHead &h = out.head[p];
            if (h.wildcard) continue;
            const SciPhyTapeHead &uP = upP.head[p];
            const size_t off = tape_lh_offset[p];
            h.log_scale += transUp(bt, tape_len[p], prefixProd(h.rep), prefixZeros(h.rep),
                                   &upP.lh[off], std::min((int)uP.depth, (int)h.depth), h.depth, &out.lh[off]);
            const double lmax = normalizeTape(&out.lh[off], h.depth);
            if (lmax < std::log(LH_UNDERFLOW)) redo[p] = 1;
            else h.log_scale += lmax;
        }
    }
    for (PhyloNeighbor *k : kids) {
        if (k == child_nei) continue;
        const SciPhyTapePartials &dd = *k->sciphy_tape_partial;
        BranchTable bt;
        fillBranchTable(k->length, bt);
        for (int p = 0; p < nptn; ++p) {
            SciPhyTapeHead &h = out.head[p];
            const SciPhyTapeHead &d = dd.head[p];
            if (h.wildcard) continue;
            h.log_scale += d.log_scale;
            if (d.wildcard || redo[p]) continue;
            const size_t off = tape_lh_offset[p];
            multiplyChild(bt, p, d, &dd.lh[off], h, &out.lh[off], redo);
        }
    }
    finishTapes(out, tape_lh_offset, redo, [&](int p, SciPhyTapeHead &h, double *lh) {
        h.log_scale = upP.head[p].log_scale;
        for (PhyloNeighbor *k : kids)
            if (k != child_nei) h.log_scale += k->sciphy_tape_partial->head[p].log_scale;
        h.log_scale += tapeLogs(p, h.rep, h.depth, &upP, upP.head[p].depth, pnei->length,
                                kids, child_nei, lh);
    });
}

// <----- Check mode (IQTREE_SCIPHY_KERNEL=check) ----->

void PhyloTreeSciPhy::checkTapes(const char *where, const double *ptn_new, const double *ptn_old) {
    CheckStats &st = check_stats[where];
    for (size_t p = 0; p < aln->getNPattern(); ++p) {
        const double a = ptn_new[p], b = ptn_old[p];
        double rel;
        if (a == b) rel = 0.0;
        else if (std::isinf(a) || std::isinf(b) || std::isnan(a) || std::isnan(b))
            rel = std::numeric_limits<double>::infinity();
        else rel = std::fabs(a - b) / std::max(std::fabs(b), 1.0);
        ++st.n;
        st.max_rel = std::max(st.max_rel, rel);
        if (rel > 1e-10) {
            if (++st.fail <= 20)
                std::cerr << std::setprecision(17) << "[sciphy kernel check] " << where
                          << " tape " << p << ": new=" << a << " old=" << b
                          << " rel=" << rel << std::endl;
        }
    }
}

// combine at every branch with both kernels (validates the upward pass)
void PhyloTreeSciPhy::checkAllBranches() {
    NodeVector nodes, nodes2;
    getBranches(nodes, nodes2);
    const size_t nptn = aln->getNPattern();
    std::vector<double> new_ptn(nptn), old_ptn(nptn);
    for (size_t j = 0; j < nodes.size(); ++j) {
        PhyloNode *n1 = (PhyloNode*) nodes[j], *n2 = (PhyloNode*) nodes2[j];
        if (n1 == root || n2 == root) continue;
        PhyloNeighbor *n12 = (PhyloNeighbor*) n1->findNeighbor(n2);
        PhyloNode *parent = (n12->direction == TOWARD_ROOT) ? n2 : n1;
        PhyloNode *child  = (n12->direction == TOWARD_ROOT) ? n1 : n2;
        PhyloNeighbor *down_nei = (PhyloNeighbor*) parent->findNeighbor(child);
        PhyloNeighbor *up_nei   = (PhyloNeighbor*) child->findNeighbor(parent);
        if (!down_nei->sciphy_tape_partial || !up_nei->sciphy_tape_partial ||
            !down_nei->sciphy_partial || !up_nei->sciphy_partial) continue;
        combineTape(*up_nei->sciphy_tape_partial, *down_nei->sciphy_tape_partial,
                    down_nei->length, new_ptn.data());
        combineVectors(*up_nei->sciphy_partial, *down_nei->sciphy_partial,
                       down_nei->length, old_ptn.data());
        checkTapes("upward", new_ptn.data(), old_ptn.data());
    }
}

// <----- Closed-form branch likelihood and Newton branch lengths ----->
// Per tape: S(mu) = sum_k c_k Pois(k; mu) + sum_r e_r P(K >= r; mu), k = number of new edits.

void PhyloTreeSciPhy::buildBranchCoeffs(const SciPhyTapePartials &U, const SciPhyTapePartials &D,
                                        BranchCoeffs &bc) const {
    const size_t nptn = aln->getNPattern();
    bc.c.assign(tape_lh_offset.back(), 0.0);
    bc.e.assign(tape_lh_offset.back(), 0.0);
    bc.scale.assign(nptn, 0.0);
    bc.active.assign(nptn, 0);
    bc.const_logl = 0.0;
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyTapeHead &uc = U.head[p];
        const SciPhyTapeHead &dc = D.head[p];
        const size_t off = tape_lh_offset[p];
        if (uc.wildcard || dc.wildcard) {   // constant in the branch length (as in combineTape)
            double lp;
            if (uc.wildcard && dc.wildcard) lp = uc.log_scale + dc.log_scale;
            else if (dc.wildcard)           lp = totalTapeMass(uc, &U.lh[off]) + dc.log_scale;
            else                            lp = totalTapeMass(dc, &D.lh[off]) + uc.log_scale;
            bc.const_logl += lp * ptn_freq[p];
            continue;
        }
        const int L = tape_len[p];
        const int dp = std::min(uc.depth, dc.depth);
        const double *F = prefixProd(dc.rep);
        const unsigned char *Z = prefixZeros(dc.rep);
        const int jp = (dc.depth == L) ? L - 1 : dc.depth;   // last non-full child state
        for (int i = 0; i <= dp; ++i) {
            const double Ui = U.lh[off + i];
            if (!(Ui > 0.0)) continue;
            int j = i;
            for (; j <= jp && Z[j] == Z[i]; ++j)
                bc.c[off + (j - i)] += Ui * D.lh[off + j] * (F[j] / F[i]);
            if (dc.depth == L && j == L && Z[L] == Z[i])
                bc.e[off + (L - i)] += Ui * D.lh[off + L] * (F[L] / F[i]);
        }
        bc.scale[p] = uc.log_scale + dc.log_scale;
        bc.active[p] = 1;
    }
}

void PhyloTreeSciPhy::buildOriginCoeffs(const SciPhyTapePartials &bio, BranchCoeffs &bc) const {
    const size_t nptn = aln->getNPattern();
    bc.c.assign(tape_lh_offset.back(), 0.0);
    bc.e.assign(tape_lh_offset.back(), 0.0);
    bc.scale.assign(nptn, 0.0);
    bc.active.assign(nptn, 0);
    bc.const_logl = 0.0;
    for (size_t p = 0; p < nptn; ++p) {
        const SciPhyTapeHead &b = bio.head[p];
        const size_t off = tape_lh_offset[p];
        if (b.wildcard) { bc.const_logl += b.log_scale * ptn_freq[p]; continue; }
        // founder = unedited (depth 0, weight 1), as in originClosureTape
        const int L = tape_len[p];
        const double *F = prefixProd(b.rep);
        const unsigned char *Z = prefixZeros(b.rep);
        const int jp = (b.depth == L) ? L - 1 : b.depth;
        int j = 0;
        for (; j <= jp && Z[j] == Z[0]; ++j) bc.c[off + j] += bio.lh[off + j] * (F[j] / F[0]);
        if (b.depth == L && j == L && Z[L] == Z[0]) bc.e[off + L] += bio.lh[off + L] * (F[L] / F[0]);
        bc.scale[p] = b.log_scale;
        bc.active[p] = 1;
    }
}

double PhyloTreeSciPhy::branchLogL(const BranchCoeffs &bc, double len, double *d1, double *d2,
                                   bool &ok) const {
    ok = false;
    BranchTable bt;
    fillBranchTable(len, bt);
    if (bt.log_space) return -std::numeric_limits<double>::infinity();
    const int Lm = max_tape_len;
    double dpois[MAX_TAPE_SITES + 1], d2pois[MAX_TAPE_SITES + 1];
    double dtail[MAX_TAPE_SITES + 1], d2tail[MAX_TAPE_SITES + 1];
    for (int k = 0; k <= Lm; ++k) {
        const double p1 = (k >= 1) ? bt.pois[k - 1] : 0.0;
        const double p2 = (k >= 2) ? bt.pois[k - 2] : 0.0;
        dpois[k]  = p1 - bt.pois[k];
        d2pois[k] = p2 - 2.0 * p1 + bt.pois[k];
        dtail[k]  = p1;
        d2tail[k] = p2 - p1;
    }
    double f = bc.const_logl, g = 0.0, h = 0.0;
    const size_t nptn = bc.active.size();
    for (size_t p = 0; p < nptn; ++p) {
        if (!bc.active[p]) continue;
        const int L = tape_len[p];
        const size_t off = tape_lh_offset[p];
        double S = 0.0, S1 = 0.0, S2 = 0.0;
        for (int k = 0; k < L; ++k) {
            const double c = bc.c[off + k];
            if (c == 0.0) continue;
            S += c * bt.pois[k]; S1 += c * dpois[k]; S2 += c * d2pois[k];
        }
        for (int r = 0; r <= L; ++r) {   // r = 0: parent already full, P(K >= 0) = 1
            const double e = bc.e[off + r];
            if (e == 0.0) continue;
            S += e * bt.tail[r]; S1 += e * dtail[r]; S2 += e * d2tail[r];
        }
        if (!(S >= LH_UNDERFLOW)) return -std::numeric_limits<double>::infinity();
        const double w = ptn_freq[p], r1 = S1 / S;
        f += w * (std::log(S) + bc.scale[p]);
        g += w * r1;
        h += w * (S2 / S - r1 * r1);
    }
    const double lambda = sciphyModel().getClockRate();   // dmu/dlen
    if (d1) *d1 = lambda * g;
    if (d2) *d2 = lambda * lambda * h;
    ok = true;
    return f;
}

// Newton in log(len), step capped at 8x, bracketed by the sign of f'; false = use Brent.
bool PhyloTreeSciPhy::newtonBranchLength(const BranchCoeffs &bc, double x, double f, double g,
                                         double h, double &best_len, double &best_logl) const {
    const double lo = params->min_branch_length, hi = params->max_branch_length;
    const double tol = std::max(params->min_branch_length, 1e-10);
    const double LOG_CAP = std::log(8.0);
    best_len = x; best_logl = f;
    double a = lo, b = hi;
    for (int it = 0; it < 100; ++it) {
        if (g > 0.0) a = x; else if (g < 0.0) b = x; else return true;
        if ((x <= lo && g <= 0.0) || (x >= hi && g >= 0.0)) return true;   // boundary optimum
        const double fu = x * g, fuu = x * x * h + x * g;
        double step = (fuu < 0.0) ? -fu / fuu : (g > 0.0 ? LOG_CAP : -LOG_CAP);
        step = std::min(LOG_CAP, std::max(-LOG_CAP, step));
        double xn = x * std::exp(step);
        if (!(xn > a && xn < b)) xn = std::sqrt(std::max(a, 1e-300) * b);   // geometric bisection
        xn = std::min(hi, std::max(lo, xn));
        const double dx = xn - x;
        x = xn;
        bool ok;
        f = branchLogL(bc, x, &g, &h, ok);
        if (!ok) return false;
        if (f > best_logl) { best_logl = f; best_len = x; }
        if (std::fabs(dx) < tol || b - a < tol) return true;
    }
    return false;
}

// Newton from the current length; keeps cur unless the new length is better.
double PhyloTreeSciPhy::optimizeBranchClosedForm(const BranchCoeffs &bc, double cur, double &new_len,
                                                 bool &used_newton) const {
    used_newton = false;
    new_len = cur;
    bool ok;
    const double x0 = std::min(std::max(cur, params->min_branch_length), params->max_branch_length);
    double g, h;
    const double f0 = branchLogL(bc, x0, &g, &h, ok);
    if (!ok) return f0;
    double x, fx;
    if (!newtonBranchLength(bc, x0, f0, g, h, x, fx)) return f0;
    used_newton = true;
    if (x0 != cur) {   // cur was outside the allowed range
        const double fc = branchLogL(bc, cur, nullptr, nullptr, ok);
        if (ok && fx < fc - 1e-8) return fc;
    } else if (fx < f0 - 1e-8) {
        return f0;
    }
    new_len = x;
    return fx;
}

void PhyloTreeSciPhy::recordCheck(const std::string &key, double err, double thresh) {
    CheckStats &st = check_stats[key];
    ++st.n;
    st.max_rel = std::max(st.max_rel, err);
    if (err > thresh && ++st.fail <= 20)
        std::cerr << std::setprecision(12) << "[sciphy brlen check] " << key << ": error " << err
                  << " > " << thresh << std::endl;
}

namespace {
// golden-section reference maximizer for the checks
template <class F>
double goldenMax(F f, double a, double b, double tol, double &fx) {
    const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
    double c = b - gr * (b - a), d = a + gr * (b - a), fc = f(c), fd = f(d);
    while (b - a > tol) {
        if (fc > fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
        else         { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
    }
    const double x = 0.5 * (a + b);
    fx = f(x);
    return x;
}
}  // anonymous namespace

template <class Exact>
void PhyloTreeSciPhy::checkBranchClosedForm(const char *where, const BranchCoeffs &bc, double cur,
                                            double newton_len, double newton_logl, Exact exact) {
    const std::string w(where);
    bool ok;
    // 1. closed form == exact likelihood (relative to |logL|)
    for (double l : {cur, newton_len, 0.5 * cur + 1e-4, 2.0 * cur + 1e-3}) {
        const double a = branchLogL(bc, l, nullptr, nullptr, ok), b = exact(l);
        if (ok && std::isfinite(b)) recordCheck(w + " identity", std::fabs(a - b) / std::max(1.0, std::fabs(b)), 1e-10);
    }
    // 2. analytic derivative == central difference of the exact likelihood
    double d1, d2;
    branchLogL(bc, cur, &d1, &d2, ok);
    const double hstep = std::max(1e-6, 1e-3 * cur);
    if (ok && cur - 2.0 * hstep > 0.0) {
        // 5-point stencil; the reference itself has rounding noise ~ eps |logL| / h
        const double f0 = exact(cur);
        const double num = (-exact(cur + 2 * hstep) + 8 * exact(cur + hstep)
                            - 8 * exact(cur - hstep) + exact(cur - 2 * hstep)) / (12.0 * hstep);
        const double scale = std::max(1.0, std::fabs(num));
        const double noise = 10.0 * 2.2e-16 * std::fabs(f0) / hstep / scale;
        recordCheck(w + " derivative", std::fabs(d1 - num) / scale, std::max(1e-6, noise));
    }
    // 3. Newton optimum vs an independent maximizer on [cur/16, 16 cur]
    const double lo = std::max(params->min_branch_length, cur / 16.0);
    const double hi = std::min(params->max_branch_length, std::max(16.0 * cur, lo + 1e-3));
    double fg;
    goldenMax(exact, lo, hi, std::max(params->min_branch_length, 1e-9), fg);
    recordCheck(w + " optimum (logL deficit vs reference)", std::max(0.0, fg - newton_logl), 1e-6);
}

// <----- Multithreading: parallel sweep and NNI scoring ----->

// resize only outside parallel regions: call workspace(nt - 1) before the loop
PhyloTreeSciPhy::Workspace &PhyloTreeSciPhy::workspace(int t) const {
    if ((int)workspaces.size() <= t) workspaces.resize(t + 1);
    return workspaces[t];
}

PhyloTreeSciPhy::Workspace &PhyloTreeSciPhy::threadWorkspace() const {
#ifdef _OPENMP
    return workspace(omp_get_thread_num());
#else
    return workspace(0);
#endif
}

bool PhyloTreeSciPhy::parallelPathEnabled() const {
    return hasBarcodeAlignment() && !useOldKernel() && !checking() && !brlen_brent && !brlen_check
           && !nni_cache_check;
}

int PhyloTreeSciPhy::sweepBranchClosedForm(PhyloNode *node1, PhyloNode *node2, double &new_len,
                                           Workspace &ws) const {
    if (rooted && (node1 == root || node2 == root)) return 0;   // origin edge: optimizeOriginBranch
    PhyloNeighbor *n12 = (PhyloNeighbor*) node1->findNeighbor(node2);
    PhyloNeighbor *n21 = (PhyloNeighbor*) node2->findNeighbor(node1);
    if (!n12 || !n21) return 0;
    PhyloNode *parent = node1, *child = node2;
    if (n12->direction == TOWARD_ROOT) { parent = node2; child = node1; }
    PhyloNeighbor *down_nei = (PhyloNeighbor*) parent->findNeighbor(child);
    PhyloNeighbor *up_nei   = (PhyloNeighbor*) child->findNeighbor(parent);
    if (!down_nei->sciphy_tape_partial || !up_nei->sciphy_tape_partial) return 0;
    buildBranchCoeffs(*up_nei->sciphy_tape_partial, *down_nei->sciphy_tape_partial, ws.bc);
    bool used;
    optimizeBranchClosedForm(ws.bc, n12->length, new_len, used);
    return used ? 1 : 2;
}

void PhyloTreeSciPhy::nniRawSwap(NNIMove &m) const {
    NeighborVec::iterator n1it = m.node1Nei_it, n2it = m.node2Nei_it;
    Neighbor *n1nei = *n1it, *n2nei = *n2it;
    m.node1->updateNeighbor(n1it, n2nei);
    n2nei->node->updateNeighbor(m.node2, m.node1);
    m.node2->updateNeighbor(n2it, n1nei);
    n1nei->node->updateNeighbor(m.node1, m.node2);
}

void PhyloTreeSciPhy::nniUndoSwap(NNIMove &m) const {
    NeighborVec::iterator n1it = m.node1Nei_it, n2it = m.node2Nei_it;
    Neighbor *cur_n1 = *n1it, *cur_n2 = *n2it;
    m.node1->updateNeighbor(n1it, cur_n2);
    cur_n2->node->updateNeighbor(m.node2, m.node1);
    m.node2->updateNeighbor(n2it, cur_n1);
    cur_n1->node->updateNeighbor(m.node1, m.node2);
}

// Virtual swap: scores both candidates like getBestNNIForBran without modifying the tree.
void PhyloTreeSciPhy::scoreBranchNNIsVirtual(PhyloNode *node1, PhyloNode *node2, NNIMove *moves,
                                             char *fallback, Workspace &ws) const {
    if (((PhyloNeighbor*) node1->findNeighbor(node2))->direction == TOWARD_ROOT) std::swap(node1, node2);
    const double NEG = -std::numeric_limits<double>::max();
    moves[0].ptnlh = moves[1].ptnlh = nullptr;
    int cnt = 0;
    FOR_NEIGHBOR_IT(node1, node2, node1_it)
        if (((PhyloNeighbor*) *node1_it)->direction != TOWARD_ROOT) {
            cnt = 0;
            FOR_NEIGHBOR_IT(node2, node1, node2_it) {
                moves[cnt].node1Nei_it = node1_it;
                moves[cnt].node2Nei_it = node2_it;
                cnt++;
            }
            break;
        }
    ASSERT(cnt == 2);
    moves[0].node1 = moves[1].node1 = node1;
    moves[0].node2 = moves[1].node2 = node2;
    moves[0].newloglh = moves[1].newloglh = NEG;
    fallback[0] = fallback[1] = 0;
    const double orig_len = ((PhyloNeighbor*) node1->findNeighbor(node2))->length;
    const size_t nptn = aln->getNPattern();

    for (int c = 0; c < 2; ++c) {
        if (!const_cast<ConstraintTree&>(constraintTree).isCompatible(moves[c])) continue;
        Neighbor *n1nei = *moves[c].node1Nei_it;   // node1 -> a
        Neighbor *n2nei = *moves[c].node2Nei_it;   // node2 -> b
        // node2's children after the swap (its neighbor order, b replaced by a)
        std::vector<PhyloNeighbor*> kids2;
        for (Neighbor *nb : node2->neighbors) {
            Neighbor *e = (nb == n2nei) ? n1nei : nb;
            if (e->node != node1) kids2.push_back((PhyloNeighbor*) e);
        }
        // node1's toward-root edge and children after the swap (a replaced by b)
        PhyloNeighbor *pnei = nullptr, *child_nei = nullptr;
        std::vector<PhyloNeighbor*> kids1;
        for (Neighbor *nb : node1->neighbors) {
            Neighbor *e = (nb == n1nei) ? n2nei : nb;
            if (!pnei && ((PhyloNeighbor*) e)->direction == TOWARD_ROOT) { pnei = (PhyloNeighbor*) e; continue; }
            kids1.push_back((PhyloNeighbor*) e);
            if (e->node == node2) child_nei = (PhyloNeighbor*) e;
        }
        buildTapeDownwardKids(kids2, ws.down, ws);
        buildTapeUpToCKids(pnei, kids1, child_nei, ws.down, ws.up, ws);
        bool ok = ws.down.head.size() == nptn && ws.up.head.size() == nptn;
        for (size_t p = 0; ok && p < nptn; ++p)
            if (ws.down.head[p].wildcard || ws.up.head[p].wildcard) ok = false;   // as in buildNNILocalPartials
        if (!ok) { fallback[c] = 1; continue; }
        buildBranchCoeffs(ws.up, ws.down, ws.bc);
        bool used;
        double nl;
        const double fl = optimizeBranchClosedForm(ws.bc, orig_len, nl, used);
        if (!used) { fallback[c] = 1; continue; }
        moves[c].newloglh = fl;
        moves[c].newLen[0].assign(1, nl);
    }
}

// Scores all candidates in parallel, then fallbacks sequentially; collects in branch order.
void PhyloTreeSciPhy::evaluateNNIs(Branches &nniBranches, vector<NNIMove> &positiveNNIs) {
    if (!parallelPathEnabled() || MPIHelper::getInstance().getNumProcesses() > 1) {
        IQTree::evaluateNNIs(nniBranches, positiveNNIs);
        return;
    }
    ensureSciPhySnapshot();
    const double backupScore = curScore;
    std::vector<Branch> br;
    br.reserve(nniBranches.size());
    for (auto &it : nniBranches) br.push_back(it.second);
    const int nb = (int)br.size(), nt = numWorkThreads();
    std::vector<NNIMove> moves(2 * nb);
    std::vector<char> fb(2 * nb, 0);
    workspace(nt - 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(nt) if (nt > 1)
#endif
    for (int i = 0; i < nb; ++i) {
        Workspace &ws = threadWorkspace();
        scoreBranchNNIsVirtual((PhyloNode*) br[i].first, (PhyloNode*) br[i].second, &moves[2 * i], &fb[2 * i], ws);
    }
    // fallbacks: swap, score and unswap on the tree
    for (int i = 0; i < nb; ++i) {
        if (!fb[2 * i] && !fb[2 * i + 1]) continue;
        ensureSciPhySnapshot();
        PhyloNeighbor *central      = (PhyloNeighbor*) moves[2 * i].node1->findNeighbor(moves[2 * i].node2);
        PhyloNeighbor *central_back = (PhyloNeighbor*) moves[2 * i].node2->findNeighbor(moves[2 * i].node1);
        const double orig_len = central->length;
        for (int c = 0; c < 2; ++c) {
            if (!fb[2 * i + c]) continue;
            NNIMove &m = moves[2 * i + c];
            nniRawSwap(m);
            double optlen;
            const double score = scoreNNICandidate(m.node1, m.node2, c, orig_len, optlen);
            central->length = central_back->length = orig_len;
            m.newloglh = score;
            m.newLen[0].assign(1, optlen);
            nniUndoSwap(m);
        }
    }
    ensureSciPhySnapshot();
    curScore = backupScore;
    for (int i = 0; i < nb; ++i) {
        const NNIMove &best = (moves[2 * i].newloglh > moves[2 * i + 1].newloglh) ? moves[2 * i] : moves[2 * i + 1];
        if (best.newloglh > curScore) positiveNNIs.push_back(best);
    }
}
