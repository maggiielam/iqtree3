#include "barcodealignment.h"
#include "utils/tools.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace std;

/**
 * BarcodeAlignment Implementation
 */

BarcodeAlignment::BarcodeAlignment() : Alignment() {
    num_sites = 0;
    max_insertion_type = 0;
}

BarcodeAlignment::BarcodeAlignment(char *filename, char *sequence_type, InputType &intype, string model) 
    : Alignment() {
    num_sites = 0;
    max_insertion_type = 0;
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
