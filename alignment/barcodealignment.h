/***************************************************************************
 *   Copyright (C) 2025 by Barcode Alignment Module
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 ***************************************************************************/

#ifndef BARCODEALIGNMENT_H
#define BARCODEALIGNMENT_H

#include "alignment.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

/**
 * BarcodeAlignment class for reading barcode data from CSV format.
 * 
 * Data model:
 * - Each cell carries k independent tapes with editing events
 * - Each tape has N target sites that are edited sequentially
 * - Editing at site i requires that site i-1 is already edited
 * - Each site can receive M different types of insertions
 * - Insertions are encoded as states 0-M, where 0 = unedited
 * 
 * CSV Format:
 * "","site_1","site_2",...,"site_N"
 * "cell_1",e1,e2,...,eN
 * "cell_2",e1,e2,...,eN
 * 
 * Where ei is the insertion type at that site (0 = no edit, 1..M = insertion type)
 * 
 * @author Barcode Alignment Module
 */
class BarcodeAlignment : public Alignment {
public:
    /**
     * Constructor
     */
    BarcodeAlignment();

    /**
     * Constructor from CSV file
     * @param filename CSV file containing barcode data
     * @param sequence_type type of sequence (for character set, typically "BIN" for binary)
     * @param intype input type (will be set to IN_CSV)
     * @param model model name
     */
    BarcodeAlignment(char *filename, char *sequence_type, InputType &intype, string model);

    /**
     * Destructor
     */
    virtual ~BarcodeAlignment();

    /**
     * Read barcode alignment from CSV file
     * @param filename CSV file path
     * @return 1 on success, 0 on failure
     */
    int readBarcode(const char *filename);

    /**
     * Parse a single CSV line, handling quoted fields
     * @param line the CSV line to parse
     * @return vector of parsed fields (without quotes)
     */
    static std::vector<std::string> parseCSVLine(const std::string &line);

    /**
     * Parse comma-separated scars into integer vector.
     * @param scarsText comma-separated list, e.g. "0,3,2,0"
     * @param out parsed integer scars
     * @return true on success
     */
    static bool parseScarsList(const std::string &scarsText, std::vector<int> &out);

    /**
     * Build alignment from parsed barcode data
     * @param cellNames names of cells (sequences)
     * @param siteNames names of editing sites
     * @param data editing values matrix (cells x sites)
     * @return 1 on success, 0 on failure
     */
    int buildBarcodeAlignment(const std::vector<std::string> &cellNames,
                             const std::vector<std::string> &siteNames,
                             const std::vector<std::vector<int>> &data);

    /**
     * Read multi-tape barcode alignment from XML lineage format.
     * @param filename XML file path
     * @return 1 on success, 0 on failure
     */
    int readBarcodeXML(const char *filename);

    /**
     * Read single-tape barcode alignment from CSV format.
     * @param filename CSV file path
     * @return 1 on success, 0 on failure
     */
    int readBarcodeCSV(const char *filename);

    /**
     * Build tape-level alignment from multi-tape parsed barcode data.
     *
     * Each tape becomes one column (pattern) in the alignment.  The observed
     * tape vector for a cell is mapped to a compound state index so that the
     * full joint within-tape likelihood can be computed later.
     *
     * @param cellNames   names of cells (sequences / taxa)
     * @param tapeNames   names of tapes (one column produced per tape)
     * @param tapeSiteCounts  number of sites in each tape (may be non-uniform)
     * @param data        flat editing values matrix [cell][site], tapes concatenated
     * @return 1 on success, 0 on failure
     */
    int buildTapeAlignment(const std::vector<std::string> &cellNames,
                           const std::vector<std::string> &tapeNames,
                           const std::vector<int>         &tapeSiteCounts,
                           const std::vector<std::vector<int>> &data);

    /**
     * Get the maximum insertion type (M) found in data
     * @return maximum insertion value
     */
    int getMaxInsertionType() const { return max_insertion_type; }

    /**
     * Get number of sites in barcode
     * @return number of sites
     */
    int getNumSites() const { return num_sites; }

    /**
     * Get number of tapes (0 if tape encoding not used).
     */
    int getNumTapes() const { return num_tapes; }

    /**
     * Get site counts per tape.
     */
    const std::vector<int>& getTapeSiteCounts() const { return tape_site_counts; }

    /**
     * Get the tape vector for a compound state index.
     * @param state_idx  compound state index (as stored in patterns)
     * @return reference to the tape vector [s0, s1, ..., s_{L-1}]
     */
    const std::vector<int>& getTapeVec(int state_idx) const {
        return tape_states[state_idx];
    }

    /**
     * Total number of distinct tape vectors (equals num_states when tape
     * encoding is active).
     */
    int getNumTapeStates() const { return (int)tape_states.size(); }

protected:
    int num_sites;              ///< Total number of editing sites (all tapes)
    int max_insertion_type;     ///< Maximum insertion type value (M)
    std::vector<std::string> site_names;  ///< Names of editing sites

    // --- tape compound-state encoding (populated by buildTapeAlignment) ---
    int num_tapes;              ///< Number of independent tapes per cell (0 = not set)
    std::vector<int> tape_site_counts;  ///< Sites per tape
    /// tape_states[i] = the tape vector [s0,s1,...,s_{L-1}] for compound state i
    std::vector<std::vector<int>> tape_states;
    /// Reverse map: tape vector -> compound state index
    std::map<std::vector<int>, int> tape_to_state;
};

#endif // BARCODEALIGNMENT_H
