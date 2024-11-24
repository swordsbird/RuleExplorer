#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <tuple>

using namespace std;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


class Table {
public:
    std::vector<std::vector<double>> cols;
    void add(const std::vector<double>& col) {
        cols.push_back(col);
    }
    std::vector<int> query(const std::vector<std::tuple<int, std::vector<double>, bool>>& conds) {
        std::vector<int> ret;
        bool is_first = true;
        for (auto cond: conds) {
            int key = std::get<0>(cond);
            double left = std::get<1>(cond)[0];
            double right = std::get<1>(cond)[1];
            bool missing = std::get<2>(cond);
            if (is_first) {
                for (int i = 0; i < cols[key].size(); ++i) {
                    if (cols[key][i] >= left && cols[key][i] < right || missing && isnan(cols[key][i])) {
                        ret.push_back(i);
                    }
                }
            } else {
                int k = 0;
                for (int i = 0; i < ret.size(); ++i) {
                    if (cols[key][ret[i]] >= left && cols[key][ret[i]] < right || missing && isnan(cols[key][ret[i]])) {
                        ret[k++] = ret[i];
                    }
                }
                ret.resize(k);
            }
            is_first = false;
        }
        return ret;
    }
};



namespace py = pybind11;

PYBIND11_MODULE(rule_query, m) {
    py::class_<Table>(m, "Table")
        .def(py::init<>())
        .def("add", &Table::add)
        .def("query", &Table::query);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}