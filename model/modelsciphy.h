/*
 * modelsciphy.h
 *
 *  Created on: Jul 6, 2026
 *      Author: Maggie Lam
 */

#ifndef MODELSCIPHY_H_
#define MODELSCIPHY_H_

#include "modelmarkov.h"

#include <vector>

class ModelSciPhy : public ModelMarkov {
public:
    ModelSciPhy(PhyloTree *tree,
                const std::vector<double> &known_edit_probs,
                double known_origin_time);

    // ModelMarkov overrides

    /**
     return the number of dimensions
     */
    virtual int getNDim();

    /**
     * @return the number of frequency dimensions
     */
    virtual int getNDimFreq();

    /**
        start structure for checkpointing
    */
    virtual void startCheckpoint();
    
    /**
     save object into the checkpoint
     */
    virtual void saveCheckpoint();
    
    /**
     restore object from the checkpoint
     */
    virtual void restoreCheckpoint();

    /**
     * @return model name
     */
    virtual string getName();

    /**
     * @return model name with parameters in form of e.g. GTR{a,b,c,d,e,f}
     */
    virtual string getNameParams(bool show_fixed_params = false);

    /**
     write information
     @param out output stream
     */
    virtual void writeInfo(ostream &out);

    /**
     write parameters, used with modeltest
     @param out output stream
     */
    virtual void writeParameters(ostream &out);

	/**
		read the rates from an input stream. it will throw error messages if failed
		@param in input stream
	*/
    virtual void readRates(istream &in) noexcept(false);

    virtual double optimizeParameters(double gradient_epsilon);

    virtual ~ModelSciPhy();

    virtual double computeFunction(double value);

    /**
        No-op override.
    */
    virtual void decomposeRateMatrix() {}

    // No dense rate matrix: transition probabilities are closed-form
    // (PhyloTreeSciPhy::logTransProbVec). These overrides error if one is requested.
    virtual uint64_t getMemoryRequired() override;
    virtual void getRateMatrix(double *rate_mat) override;
    virtual void getQMatrix(double *q_mat, int mixture = 0) override;
    virtual void computeTransMatrix(double time, double *trans_matrix,
                                    int mixture = 0, int selected_row = -1) override;
    virtual void computeTransMatrixNonrev(double time, double *trans_matrix,
                                          int mixture = 0) override;
    virtual void computeTransDerv(double time, double *trans_matrix,
                                  double *trans_derv1, double *trans_derv2,
                                  int mixture = 0) override;
    virtual void setRateMatrix(double *rate_mat) override;
    virtual void setFullRateMatrix(double *rate_mat, double *freq) override;
    virtual void readRates(string str) noexcept(false) override;
    using ModelMarkov::readRates;

    // base versions read the (freed) rates[] array
    virtual bool isUnstableParameters() override;
    virtual void adaptStateFrequency(double *state_freq) override;

protected:
    // optimizer vector: the only free parameter is the clock rate
    virtual void setVariables(double *variables) override;
    virtual bool getVariables(double *variables) override;

public:

    /** accessors for PhyloTreeSciPhy */
    double getClockRate() const                     { return clock_rate; }
    const std::vector<double>& getEditProbs() const { return edit_probs; }
    double getOriginTime() const                    { return origin_time; }

    /** fix a parameter at a user-supplied value (skips it in optimization) */
    void fixClockRate(double v) { clock_rate = v; clock_rate_fixed = true; }
    void setClockRate(double v) { clock_rate = v; }

    // Per-tip dropout probability q (SciPhy "missingProbability"): dropped -> q,
    // observed -> 1-q. Unset (<= 0) gives a flat weight of 1; q = 0 is treated as
    // unset because a dropped tape would otherwise make the likelihood -inf.
    double getDropoutProb() const { return dropout_prob; }
    bool   hasDropoutProb() const { return dropout_prob > 0.0; }
    void   setDropoutProb(double q) { dropout_prob = q; }

protected:
    /** the one free parameter */
    double clock_rate;
    bool   clock_rate_fixed;

    /** user-supplied constants — set in constructor, never modified */
    const std::vector<double> edit_probs;
    const double origin_time;

    /** per-tip dropout probability q; < 0 = not set (flat weight 1) */
    double dropout_prob = -1.0;
};

#endif // MODELSCIPHY_H_