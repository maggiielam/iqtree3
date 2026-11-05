#include "barcodealignment.h"
#include "utils/tools.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <map>

using namespace std;

namespace {

string trimCopy(const string &s) {
    size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == string::npos) {
        return "";
    }
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

bool extractXmlAttribute(const string &line, const string &attr, string &value) {
    string token = attr + "=\"";
    size_t start = line.find(token);
    if (start == string::npos) {
        return false;
    }
    start += token.size();
    size_t end = line.find('"', start);
    if (end == string::npos) {
        return false;
    }
    value = line.substr(start, end - start);
    return true;
}

}

/**
 * BarcodeAlignment Implementation
 */

BarcodeAlignment::BarcodeAlignment() : Alignment() {
    num_sites = 0;
    max_insertion_type = 0;
    num_tapes = 0;
}

BarcodeAlignment::BarcodeAlignment(char *filename, char *sequence_type, InputType &intype, string model) 
    : Alignment() {
    num_sites = 0;
    max_insertion_type = 0;
    num_tapes = 0;
    name = "Noname";
    model_name = model;
    if (sequence_type)
        this->sequence_type = sequence_type;
    aln_file = filename;
    num_states = 0;
    frac_const_sites = 0.0;
    frac_invariant_sites = 0.0;
    codon_table = NULL;
    genetic_code = NULL;
    non_stop_codon = NULL;
    seq_type = SEQ_UNKNOWN;
    STATE_UNKNOWN = 126;
    pars_lower_bound = NULL;
    
    intype = IN_CSV;
    
    cout << "Reading alignment file " << filename << " ... ";
    cout << "CSV barcode format detected" << endl;
    
    if (readBarcode(filename) == 0) {
        outError((string)"Cannot read barcode alignment file: " + filename);
    }
    
    countConstSite();
    
    if (Params::getInstance().compute_seq_composition) {
        cout << "Alignment has " << getNSeq() << " sequences with " << getNSite()
             << " columns, " << getNPattern() << " distinct patterns" << endl
             << num_informative_sites << " parsimony-informative, "
             << num_variant_sites-num_informative_sites << " singleton sites, "
             << (int)(frac_const_sites*getNSite()) << " constant sites" << endl;
        cout << "Number of insertion types (states): " << num_states << endl;
    }
    
    checkSeqName();
}

BarcodeAlignment::~BarcodeAlignment() {
    // Nothing special to clean up
}

/**
 * Parse a CSV line handling quoted fields
 */
vector<string> BarcodeAlignment::parseCSVLine(const string &line) {
    vector<string> fields;
    string field;
    bool in_quotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            // End of field
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    
    // Add the last field
    fields.push_back(field);
    
    return fields;
}

/**
 * Read barcode alignment from CSV file
 */
int BarcodeAlignment::readBarcode(const char *filename) {
    max_insertion_type = 0;
    num_sites = 0;
    site_names.clear();

    ifstream infile(filename);

    if (!infile.is_open()) {
        cout << "Cannot open file " << filename << endl;
        return 0;
    }

    string firstNonEmpty;
    string line;
    while (getline(infile, line)) {
        string trimmed = trimCopy(line);
        if (!trimmed.empty()) {
            firstNonEmpty = trimmed;
            break;
        }
    }
    infile.close();

    if (firstNonEmpty.empty()) {
        cout << "Empty file: " << filename << endl;
        return 0;
    }

    if (firstNonEmpty[0] == '<') {
        return readBarcodeXML(filename);
    }
    return readBarcodeCSV(filename);
}

bool BarcodeAlignment::parseScarsList(const string &scarsText, vector<int> &out) {
    out.clear();
    string item;
    stringstream ss(scarsText);
    while (getline(ss, item, ',')) {
        item = trimCopy(item);
        if (item.empty()) {
            return false;
        }
        int value = 0;
        if (item == "?") {
            value = -1;                 // missing site (dropout); encoded as -1
        } else {
            try {
                value = stoi(item);
            } catch (...) {
                return false;
            }
            if (value < 0) {
                return false;
            }
        }
        out.push_back(value);
    }
    return !out.empty();
}

int BarcodeAlignment::readBarcodeXML(const char *filename) {
    ifstream infile(filename);
    
    if (!infile.is_open()) {
        cout << "Cannot open file " << filename << endl;
        return 0;
    }
    
    string line;
    vector<string> cellNames;
    vector<string> tapeOrder;
    vector<int> tapeSiteCounts;
    vector<vector<int>> data;
    int line_num = 0;
    string currentCell;
    map<string, vector<int>> currentCellTapes;
    vector<string> currentCellTapeOrder;
    bool inCell = false;
    
    try {
        while (getline(infile, line)) {
            line_num++;

            string trimmed = trimCopy(line);
            if (trimmed.empty()) {
                continue;
            }

            if (trimmed.find("<lineage_data") != string::npos || trimmed.find("</lineage_data") != string::npos) {
                continue;
            }

            if (trimmed.find("<cell") != string::npos) {
                if (inCell) {
                    cout << "Malformed XML near line " << line_num << ": nested <cell> not supported" << endl;
                    return 0;
                }
                if (!extractXmlAttribute(trimmed, "id", currentCell)) {
                    cout << "Missing cell id at line " << line_num << endl;
                    return 0;
                }
                currentCellTapes.clear();
                currentCellTapeOrder.clear();
                inCell = true;
                continue;
            }

            if (trimmed.find("</cell") != string::npos) {
                if (!inCell) {
                    cout << "Malformed XML near line " << line_num << ": closing </cell> without opening tag" << endl;
                    return 0;
                }

                if (currentCellTapes.empty()) {
                    cout << "Cell '" << currentCell << "' contains no <alignment> entries" << endl;
                    return 0;
                }

                if (tapeOrder.empty()) {
                    tapeOrder = currentCellTapeOrder;
                    for (size_t i = 0; i < tapeOrder.size(); ++i) {
                        tapeSiteCounts.push_back((int)currentCellTapes[tapeOrder[i]].size());
                    }
                } else if ((int)currentCellTapes.size() != (int)tapeOrder.size()) {
                    cout << "Cell '" << currentCell << "' has " << currentCellTapes.size()
                         << " alignments, expected " << tapeOrder.size() << endl;
                    return 0;
                }

                vector<int> row;
                for (size_t tape = 0; tape < tapeOrder.size(); ++tape) {
                    const string &tapeId = tapeOrder[tape];
                    auto it = currentCellTapes.find(tapeId);
                    if (it == currentCellTapes.end()) {
                        cout << "Cell '" << currentCell << "' is missing alignment id '" << tapeId << "'" << endl;
                        return 0;
                    }
                    if ((int)it->second.size() != tapeSiteCounts[tape]) {
                        cout << "Alignment id '" << tapeId << "' has inconsistent number of sites in cell '" << currentCell << "'" << endl;
                        return 0;
                    }
                    row.insert(row.end(), it->second.begin(), it->second.end());
                }

                cellNames.push_back(currentCell);
                data.push_back(row);
                inCell = false;
                currentCell.clear();
                currentCellTapes.clear();
                continue;
            }

            if (trimmed.find("<alignment") != string::npos) {
                if (!inCell) {
                    cout << "Malformed XML near line " << line_num << ": <alignment> outside <cell>" << endl;
                    return 0;
                }

                string alignmentId;
                string scarsText;
                if (!extractXmlAttribute(trimmed, "id", alignmentId)) {
                    cout << "Missing alignment id at line " << line_num << endl;
                    return 0;
                }
                if (!extractXmlAttribute(trimmed, "scars", scarsText)) {
                    cout << "Missing scars attribute for alignment id '" << alignmentId << "' at line " << line_num << endl;
                    return 0;
                }

                vector<int> scars;
                if (!parseScarsList(scarsText, scars)) {
                    cout << "Invalid scars list for alignment id '" << alignmentId << "' at line " << line_num << endl;
                    return 0;
                }

                for (size_t i = 0; i < scars.size(); ++i) {
                    max_insertion_type = max(max_insertion_type, scars[i]);
                }

                if (currentCellTapes.find(alignmentId) != currentCellTapes.end()) {
                    cout << "Duplicate alignment id '" << alignmentId << "' in cell '" << currentCell << "'" << endl;
                    return 0;
                }
                currentCellTapes[alignmentId] = scars;
                currentCellTapeOrder.push_back(alignmentId);
            }
        }

        infile.close();

        if (inCell) {
            cout << "Malformed XML: file ended before closing cell '" << currentCell << "'" << endl;
            return 0;
        }

        if (cellNames.empty()) {
            cout << "No cells found in barcode XML file" << endl;
            return 0;
        }

        vector<string> siteNames;
        for (size_t tape = 0; tape < tapeOrder.size(); ++tape) {
            for (int site = 0; site < tapeSiteCounts[tape]; ++site) {
                siteNames.push_back(tapeOrder[tape] + string("_site_") + convertIntToString(site + 1));
            }
        }

        num_sites = siteNames.size();
        this->site_names = siteNames;

        return buildTapeAlignment(cellNames, tapeOrder, tapeSiteCounts, data);
        
    } catch (exception &e) {
        cout << "Error reading barcode XML file: " << e.what() << endl;
        return 0;
    }
}

int BarcodeAlignment::readBarcodeCSV(const char *filename) {
    ifstream infile(filename);

    if (!infile.is_open()) {
        cout << "Cannot open file " << filename << endl;
        return 0;
    }

    string line;
    vector<string> siteNames;
    vector<string> cellNames;
    vector<vector<int>> data;
    int line_num = 0;

    try {
        // Read header row
        if (!getline(infile, line)) {
            cout << "Empty file: " << filename << endl;
            return 0;
        }
        line_num++;

        vector<string> header = parseCSVLine(line);

        if (header.empty()) {
            cout << "Invalid header in barcode file" << endl;
            return 0;
        }

        // Skip first column (empty or "cell_id")
        for (size_t i = 1; i < header.size(); ++i) {
            siteNames.push_back(header[i]);
        }

        if (siteNames.empty()) {
            cout << "No sites found in barcode file" << endl;
            return 0;
        }

        num_sites = siteNames.size();
        this->site_names = siteNames;

        // Read data rows
        while (getline(infile, line)) {
            line_num++;

            if (line.empty() || line.find_first_not_of(" \t\r\n") == string::npos) {
                continue; // Skip empty lines
            }

            vector<string> fields = parseCSVLine(line);

            if (fields.empty()) {
                continue;
            }

            // First field is cell name
            string cellName = fields[0];
            cellNames.push_back(cellName);

            // Parse numeric values
            vector<int> row;

            if ((int)fields.size() != (int)num_sites + 1) {
                cout << "Warning: Row " << line_num << " has " << fields.size()
                     << " fields, expected " << (num_sites + 1) << endl;
            }

            for (size_t i = 1; i <= (size_t)num_sites && i < fields.size(); ++i) {
                string &val_str = fields[i];

                // Remove whitespace
                val_str.erase(0, val_str.find_first_not_of(" \t"));
                val_str.erase(val_str.find_last_not_of(" \t") + 1);

                int val;
                try {
                    val = stoi(val_str);
                } catch (...) {
                    cout << "Invalid numeric value '" << val_str << "' at row " << line_num
                         << ", column " << i << endl;
                    return 0;
                }

                if (val < 0) {
                    cout << "Negative value " << val << " at row " << line_num
                         << ", column " << i << endl;
                    return 0;
                }

                row.push_back(val);
                max_insertion_type = max(max_insertion_type, val);
            }

            if ((int)row.size() != num_sites) {
                cout << "Row " << line_num << " has " << row.size()
                     << " values, expected " << num_sites << endl;
                return 0;
            }

            data.push_back(row);
        }

        infile.close();

        if (cellNames.empty()) {
            cout << "No cells found in barcode file" << endl;
            return 0;
        }

        // Build the alignment
        return buildBarcodeAlignment(cellNames, siteNames, data);

    } catch (exception &e) {
        cout << "Error reading barcode file: " << e.what() << endl;
        return 0;
    }
}

/**
 * Build alignment from parsed barcode data
 * 
 * Each site becomes a column in the alignment
 * Each cell becomes a sequence (row)
 * Values become state codes
 */
int BarcodeAlignment::buildBarcodeAlignment(const vector<string> &cellNames,
                                           const vector<string> &siteNames,
                                           const vector<vector<int>> &data) {
    if (cellNames.empty() || siteNames.empty() || data.empty()) {
        return 0;
    }
    
    // Clear any existing patterns
    this->clear();
    
    int num_seqs = cellNames.size();
    int num_sites = siteNames.size();
    
    // Set sequence names
    for (const auto &name : cellNames) {
        addSeqName(name);
    }
    
    // States: 0 (unedited), 1..M (insertion types)
    num_states = max_insertion_type + 1;
    
    if (num_states < 2) {
        cout << "Warning: only " << num_states << " state(s) found, setting minimum to 2" << endl;
        num_states = 2;
    }
    
    // Set sequence type based on number of states
    // TODO: TEMPORARY - treating barcode data as morphological data
    if (num_states == 2)
        seq_type = SEQ_BINARY;
    else
        seq_type = SEQ_MORPH;
        // seq_type = SEQ_TAPE;
    
    computeUnknownState();
    
    // Resize site_pattern before adding patterns
    site_pattern.resize(num_sites, -1);
    
    // Create patterns for each site
    for (int site = 0; site < num_sites; ++site) {
        Pattern pat(num_seqs, 1);
        
        for (int seq = 0; seq < num_seqs; ++seq) {
            int state = data[seq][site];
            pat[seq] = (StateType)state;
        }
        
        addPattern(pat, site, 1);
    }
    
    return 1;
}

/**
 * Build tape-level alignment: one IQ-TREE pattern column per tape.
 *
 * Each cell's tape vector (e.g. [0,3,2,0,0]) is mapped to a compound state
 * index.  The mapping is stored in tape_states / tape_to_state so the SciPhy
 * likelihood kernel can later recover the raw tape vector for any state.
 *
 * After this call:
 *   - getNPattern() == num_tapes  (one column per tape)
 *   - num_states    == number of distinct tape vectors seen across all cells
 *                      and tapes
 *   - tape_states[i] gives the tape vector for compound state i
 */
int BarcodeAlignment::buildTapeAlignment(const vector<string> &cellNames,
                                         const vector<string> &tapeNames,
                                         const vector<int>    &tapeSiteCounts,
                                         const vector<vector<int>> &data) {
    if (cellNames.empty() || tapeNames.empty() || data.empty()) {
        return 0;
    }
    if (tapeNames.size() != tapeSiteCounts.size()) {
        cout << "Error: tapeNames and tapeSiteCounts size mismatch" << endl;
        return 0;
    }

    // Clear existing state
    this->clear();
    tape_states.clear();
    tape_to_state.clear();

    int num_cells = (int)cellNames.size();
    int n_tapes   = (int)tapeNames.size();

    // Register sequence (taxon) names
    for (const auto &nm : cellNames) {
        addSeqName(nm);
    }

    this->num_tapes       = n_tapes;
    this->tape_site_counts = tapeSiteCounts;
    this->num_sites        = 0;
    for (int tc : tapeSiteCounts) this->num_sites += tc;

    // --- Pass 1: enumerate all distinct tape states across every tape and cell
    for (int t = 0; t < n_tapes; ++t) {
        int offset = 0;
        for (int tt = 0; tt < t; ++tt) offset += tapeSiteCounts[tt];
        int slen = tapeSiteCounts[t];

        for (int c = 0; c < num_cells; ++c) {
            vector<int> tape_vec(data[c].begin() + offset,
                                 data[c].begin() + offset + slen);

            // fully-missing tape (all '?') is a dropout, not a state
            bool missing = std::all_of(tape_vec.begin(), tape_vec.end(),
                                       [](int x){ return x < 0; });
            if (missing) continue;

            for (int v : tape_vec)
                max_insertion_type = max(max_insertion_type, v);

            if (tape_to_state.find(tape_vec) == tape_to_state.end()) {
                tape_to_state[tape_vec] = (int)tape_states.size();
                tape_states.push_back(tape_vec);
            }
        }
    }

    // Now num_states is known — set it before any addPattern call
    num_states = (int)tape_states.size();
    if (num_states < 2) num_states = 2;

    seq_type = SEQ_MORPH;
    computeUnknownState();

    // Allocate site_pattern for n_tapes columns (one per tape)
    site_pattern.resize(n_tapes, -1);

    // --- Pass 2: build one IQ-TREE pattern column per tape
    for (int t = 0; t < n_tapes; ++t) {
        int offset = 0;
        for (int tt = 0; tt < t; ++tt) offset += tapeSiteCounts[tt];
        int slen = tapeSiteCounts[t];

        Pattern pat(num_cells, 0);

        for (int c = 0; c < num_cells; ++c) {
            vector<int> tape_vec(data[c].begin() + offset,
                                 data[c].begin() + offset + slen);
            bool missing = std::all_of(tape_vec.begin(), tape_vec.end(),
                                       [](int x){ return x < 0; });
            // dropout -> STATE_UNKNOWN (wildcard in the SciPhy likelihood)
            pat[c] = missing ? (StateType) STATE_UNKNOWN
                             : (StateType) tape_to_state.at(tape_vec);
        }

        addPattern(pat, t, 1);
    }

    cout << "Tape encoding: " << n_tapes << " tape column(s), "
         << num_states << " distinct tape states" << endl;

    return 1;
}

std::vector<double> BarcodeAlignment::estimateEditProbs(double alpha) const {
    const int M = std::max(1, max_insertion_type);
    std::vector<double> counts((size_t)M, 0.0);
    const int n_states = (int)tape_states.size();
    if (n_states == 0) return std::vector<double>((size_t)M, 1.0 / M);
    for (size_t ptn = 0; ptn < size(); ++ptn) {
        const Pattern &pat = at(ptn);
        for (size_t seq = 0; seq < getNSeq(); ++seq) {
            const int st = (int)pat[seq];
            if (st < 0 || st >= n_states) continue;      // missing / unknown
            for (int scar : tape_states[st])
                if (scar >= 1 && scar <= M) counts[scar - 1] += pat.frequency;
        }
    }
    double tot = 0.0;
    for (double &c : counts) { c += alpha; tot += c; }
    for (double &c : counts) c /= tot;
    return counts;
}
