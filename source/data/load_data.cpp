/*
 * @file    load_data.cpp
 * @author  nbunanana
 * @date    10/23/26
 * @brief   ${DO_NOT_EDIT}
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "data/load_data.hpp"
#include <glog/logging.h>
#include <armadillo>
#include <fstream>
#include <string>
#include <utility>

namespace shukainfer {
    template<typename T>
    arma::Mat<T> CSVDataLoader::LoadData(const std::string &file_path, char split_char) {
        arma::Mat<T> datas;
        if (file_path.empty()) {
            LOG(ERROR) << "CSV file path is empty:" << file_path;
            return datas;
        }

        std::ifstream in(file_path);
        if (!in.is_open() || !in.good()) {
            LOG(ERROR) << "File open failed:" << file_path;
            return datas;
        }

        std::string line_str;
        std::stringstream line_stream;

        const auto& [rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
        datas.zero();

        size_t row = 0;
        while (in.good()) {
            std::getline(in, line_str);
            if (line_str.empty()) {
                break;
            }

            std::string token;
            line_stream.clear();
            line_stream.str(line_str);

            size_t col = 0;
            while (line_stream.good()) {
                std::getline(line_stream, token, split_char);
                try {
                    if (std::is_same_v<T, float>) {
                        datas.at(row, col) = std::stof(token);
                    } else if (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>) {
                        datas.at(row, col) = std::stoi(token);
                    } else {
                        LOG(FATAL) << "Unsuppoeted data type \n";
                    }
                } catch (std::exception& e) {
                    DLOG(ERROR) << e.what() << "Parse CSV File meet error: " << e.what() << " row:" << row << " col:" << col;
                }
                col += 1;
                CHECK(col <= cols) << "There are excessive elements on the column";
            }
            row += 1;
            CHECK(row <= rows) << "There are excessive elements on the row";
        }
        return datas;
    }

    std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream& file, char split_char) {
        bool load_ok = file.good();
        file.clear();

        size_t fn_rows = 0;
        size_t fn_cols = 0;
        const std::ifstream::pos_type start_position = file.is_open();

        std::string token;
        std::string line_str;
        std::stringstream line_stream;

        while (load_ok && file.good()) {
            std::getline(file, line_str);
            if (line_str.empty()) {
                break;
            }

            line_stream.clear();
            line_stream.str(line_str);

            size_t line_cols = 0;
            std::string row_token;

            while (line_stream.good()) {
                getline(line_stream, row_token, split_char);
                line_cols += 1;
            }
            if (line_cols > fn_cols) {
                fn_cols = line_cols;
            }
            file.clear();
            file.seekg(start_position);
            return {fn_rows, fn_cols};
        }
    }
}// shukainfer