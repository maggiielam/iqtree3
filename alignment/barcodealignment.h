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
     * Get the maximum insertion type (M) found in data
     * @return maximum insertion value
     */
    int getMaxInsertionType() const { return max_insertion_type; }

    /**
     * Get number of sites in barcode
     * @return number of sites
     */
    int getNumSites() const { return num_sites; }

protected:
    int num_sites;              ///< Number of editing sites
    int max_insertion_type;     ///< Maximum insertion type value (M)
    std::vector<std::string> site_names;  ///< Names of editing sites
};

#endif // BARCODEALIGNMENT_H
