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

    // Free ModelMarkov's S x S rate_matrix / rates (S = #tape vectors; unused, GBs on
    // real data). aligned_free nulls the pointer, so no double free.
    aligned_free(rate_matrix);
    delete[] rates;
    rates = nullptr;

    // uniform equilibrium frequencies (O(S); only printed, never used in the likelihood)
    for (int i = 0; i < num_states; ++i) {
        state_freq[i] = 1.0 / (double)num_states;
    }
}

// <----- dense-matrix machinery: not available for SciPhy ----->

uint64_t ModelSciPhy::getMemoryRequired() {
    // state_freq (S doubles) + edit probabilities; no S x S storage.
    return ModelSubst::getMemoryRequired() + edit_probs.size() * sizeof(double);
}

namespace {
[[noreturn]] void sciphyNoDenseMatrix(const char *what) {
    outError(std::string("SciPhy model has no dense rate/transition matrix (") + what +
             "). Transition probabilities are computed in closed form by PhyloTreeSciPhy; "
             "this option is not supported for barcode data.");
    throw std::runtime_error(what);   // outError exits; keeps [[noreturn]] honest
}
}

void ModelSciPhy::getRateMatrix(double *) { sciphyNoDenseMatrix("getRateMatrix"); }
void ModelSciPhy::getQMatrix(double *, int) { sciphyNoDenseMatrix("getQMatrix"); }
void ModelSciPhy::computeTransMatrix(double, double *, int, int) { sciphyNoDenseMatrix("computeTransMatrix"); }
void ModelSciPhy::computeTransMatrixNonrev(double, double *, int) { sciphyNoDenseMatrix("computeTransMatrixNonrev"); }
void ModelSciPhy::computeTransDerv(double, double *, double *, double *, int) { sciphyNoDenseMatrix("computeTransDerv"); }
void ModelSciPhy::setRateMatrix(double *) { sciphyNoDenseMatrix("setRateMatrix"); }
void ModelSciPhy::setFullRateMatrix(double *, double *) { sciphyNoDenseMatrix("setFullRateMatrix"); }
void ModelSciPhy::readRates(string) noexcept(false) { sciphyNoDenseMatrix("readRates"); }

bool ModelSciPhy::isUnstableParameters() {
    // no rate entries; the clock is bounded in optimizeParameters
    return false;
}

void ModelSciPhy::adaptStateFrequency(double *freq) {
    ModelSubst::setStateFrequency(freq);       // no rates[] to rescale
}

void ModelSciPhy::setVariables(double *variables) {
    if (getNDim() > 0) variables[1] = clock_rate;
}

bool ModelSciPhy::getVariables(double *variables) {
    if (getNDim() <= 0) return false;
    const bool changed = (variables[1] != clock_rate);
    clock_rate = variables[1];
    return changed;
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
    out << "  dropout prob (q): ";
    if (hasDropoutProb()) out << dropout_prob << "  (tip weights q / 1-q)" << endl;
    else                  out << "not set  (dropped tapes weighted 1)" << endl;
}

void ModelSciPhy::writeParameters(ostream &out) {
    out << clock_rate;
}

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