/*
 * modelsciphy.cpp
 *
 *  Created on: Jul 6, 2026
 *      Author: Maggie Lam
 */

#include "modelsciphy.h"
#include "tree/phylotree.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>


ModelSciPhy::ModelSciPhy(PhyloTree *tree,
                         const std::vector<double> &known_edit_probs,
                         double known_origin_time)
: ModelMarkov(tree, false, true), // reversible = false, adapt_tree = true
  clock_rate(1.0),
  clock_rate_fixed(false),
  edit_probs(known_edit_probs),
  origin_time(known_origin_time)
{
    // using ModelMarkov: reversible = false, adapt_tree = true, calls convertToRooted() via adapt_tree

    name      = "SciPhy";
    full_name = "SciPhy";

    // just for printing info about the model, not used for likelihood calculations
    freq_type = FREQ_EQUAL;

    // Populate the inherited rate_matrix and state_freq with Mk-style defaults. 
    // SciPhy itself doesn't use these but the base-class matrix machinery (used during the initial pairwise ML-distance
    // computation for BIONJ, and by writeInfo / reportModel) reads them
    // TODO: check how initial trees are being constructed, if it would be cleaner to just not use BIONJ
    if (num_states > 1) {
        // rate_matrix is num_states*num_states, stored row-major.
        const double off_diag = 1.0 / (double)(num_states - 1);
        for (int i = 0; i < num_states; ++i) {
            double *row = rate_matrix + (size_t)i * num_states;
            for (int j = 0; j < num_states; ++j) {
                row[j] = (i == j) ? -1.0 : off_diag;
            }
        }
        // uniform equilibrium frequencies
        for (int i = 0; i < num_states; ++i) {
            state_freq[i] = 1.0 / (double)num_states;
        }
    }
}

ModelSciPhy::~ModelSciPhy() {}


int ModelSciPhy::getNDim() {
    return clock_rate_fixed ? 0 : 1;
}

int ModelSciPhy::getNDimFreq() {
    return 0;
}

string ModelSciPhy::getName() {
    return "SciPhy";
}

string ModelSciPhy::getNameParams(bool show_fixed_params) {
    std::ostringstream oss;
    oss << "SciPhy{clock=" << clock_rate;
    if (show_fixed_params) {
        oss << ",origin=" << origin_time
            << ",edit_probs=[";
        for (size_t i = 0; i < edit_probs.size(); ++i) {
            if (i > 0) oss << ",";
            oss << edit_probs[i];
        }
        oss << "]";
        if (clock_rate_fixed) oss << ",clock_fixed";
    }
    oss << "}";
    return oss.str();
}

void ModelSciPhy::writeInfo(ostream &out) {
    out << "SciPhy model parameters:" << endl;
    out << "  clock rate:  " << clock_rate
        << (clock_rate_fixed ? "  (fixed)" : "  (estimated)") << endl;
    out << "  origin time: " << origin_time << "  (fixed by user)" << endl;
}

void ModelSciPhy::writeParameters(ostream &out) {
    out << clock_rate;
}

// readRates — unsupported, TODO: add support for this?

void ModelSciPhy::readRates(istream &in) noexcept(false) {
    throw std::runtime_error(
        "readRates() is not supported for the SciPhy model. Edit "
        "probabilities are supplied at construction, not read from a file.");
}

// MARK - Checkpointing

void ModelSciPhy::startCheckpoint() {
    checkpoint->startStruct("ModelSciPhy");
}

void ModelSciPhy::saveCheckpoint() {
    startCheckpoint();
    CKP_SAVE(clock_rate);
    CKP_SAVE(clock_rate_fixed);
    endCheckpoint();
    ModelMarkov::saveCheckpoint();
}

void ModelSciPhy::restoreCheckpoint() {
    ModelMarkov::restoreCheckpoint();
    startCheckpoint();
    CKP_RESTORE(clock_rate);
    CKP_RESTORE(clock_rate_fixed);
    endCheckpoint();
}

// MARK - Brent optimization

double ModelSciPhy::computeFunction(double value) {
    clock_rate = value;
    // need to recompute every time the clock rate changes, so clear partial likelihoods
    phylo_tree->clearAllPartialLH();
    return -phylo_tree->computeLikelihood();
}

double ModelSciPhy::optimizeParameters(double gradient_epsilon) {
    if (fixed_parameters || clock_rate_fixed) {
        return phylo_tree->computeLikelihood();
    }

    if (verbose_mode >= VB_MAX) {
        cout << "Optimizing " << name << " clock rate (Brent)..." << endl;
    }

    double negative_lh, ferror;
    double optx = minimizeOneDimen(MIN_RATE, clock_rate, MAX_RATE,
                                   std::max(gradient_epsilon, TOL_RATE),
                                   &negative_lh, &ferror);
    return -computeFunction(optx);
}