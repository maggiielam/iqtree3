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

    /** accessors for PhyloTreeSciPhy */
    double getClockRate() const                     { return clock_rate; }
    const std::vector<double>& getEditProbs() const { return edit_probs; }
    double getOriginTime() const                    { return origin_time; }
    int    getNumScarTypes() const                  { return (int)edit_probs.size(); }

    /** fix a parameter at a user-supplied value (skips it in optimization) */
    void fixClockRate(double v) { clock_rate = v; clock_rate_fixed = true; }


protected:
    // // free parameters
    // double clock_rate;
    // std::vector<double> edit_probs;
    // double origin_time;

    // // which params are fixed/known
    // bool clock_rate_fixed;
    // bool edit_probs_fixed;
    // bool origin_time_fixed;

    /** the one free parameter */
    double clock_rate;
    bool   clock_rate_fixed;

    /** user-supplied constants — set in constructor, never modified */
    const std::vector<double> edit_probs;
    const double origin_time;

    // TODO: might have to use these if we want to infer edit probabilities or origin time
    // enum WhichParam { OPT_NONE, OPT_CLOCK, OPT_EDIT_ALPHA, OPT_ORIGIN };
    // WhichParam current_opt_param;
    // int current_alpha_idx;
    // std::vector<double> current_alphas;
    // double current_tree_height;

    // double optimizeClockRate(double eps);
    // double optimizeEditProbs(double eps);
    // double optimizeOriginTime(double eps);

    // void syncOriginBranchLength();
    // double currentTreeHeight() const;
};

#endif // MODELSCIPHY_H_