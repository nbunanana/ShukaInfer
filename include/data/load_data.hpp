/*
 * @file    load_data.hpp
 * @author  nbunanana
 * @date    10/23/26
 * @brief   
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_DATA_LOADDATA_HPP
#define SHUKAINFER_DATA_LOADDATA_HPP

#include <glog/logging.h>
#include <armadillo>
#include <string>

namespace shukainfer {
    class CSVDataLoader {
    public:
    /**
      * @brief Loads CSV file into matrix
      *
      * Loads data from a CSV file into an Armadillo matrix.
      *
      * @param file_path Path to CSV file
      * @param split_char Delimiter character
      * @return Matrix containing loaded data
      */
    template <typename T>
    static arma::Mat<T> LoadData(const std::string& file_path, char split_char = ',');

    private:
    /**
     * @brief Gets matrix dimensions from CSV
     *
     * Gets number of rows and cols for matrix based on CSV file.

     * @param file CSV file stream
     * @param split_char Delimiter character
     * @return Pair of rows and cols
    */
    static std::pair<size_t, size_t> GetMatrixSize(std::ifstream& file, char split_char);
    };
} // shukainfer
#endif //SHUKAINFER_DATA_LOADDATA_HPP
