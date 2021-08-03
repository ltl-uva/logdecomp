#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>

#include "logval.h"

namespace py = pybind11;


typedef LogVal<double> LogValD;
namespace Eigen {
    typedef Eigen::Matrix<LogValD, Dynamic, Dynamic> MatrixXlogd;
    typedef Eigen::Matrix<float, Dynamic, Dynamic> MatrixXf;
}


class log_domain_lu { public:
    log_domain_lu(const Eigen::MatrixXf& X)
    : lu(X.unaryExpr([] (const float& f) { return LogValD((double) f, false); }))
    { }

    float logdet() {
        return (float) lu.determinant().logabs();
    }

    auto inv() {
        Eigen::MatrixXlogd Xinv = lu.inverse();
        Eigen::MatrixXf Xinvf = Xinv.cast<double>().cast<float>();
        return Xinvf;
    }

private:
    Eigen::FullPivLU<Eigen::MatrixXlogd> lu;
};


class batch_log_domain_lu {
public:
    batch_log_domain_lu(
        const py::array_t<float>& X,
        const std::vector<int>& lengths)
    : lengths{ lengths }
    {
        auto X_acc = X.unchecked<3>();

        batch_size = X_acc.shape(0);
        dim1 = X_acc.shape(1);
        dim2 = X_acc.shape(2);

        for (py::ssize_t k = 0; k < batch_size; ++k) {
            auto dk = lengths[k];
            Eigen::MatrixXlogd Xk(dk, dk);
            for (py::ssize_t i = 0; i < dk; ++i) {
                for (py::ssize_t j = 0; j < dk; ++j) {
                    Xk(i, j) = LogValD((double) X_acc(k, i, j), false);
                }
            }
            lus.emplace_back(Xk);
        }
    }

    py::array_t<float> logdet() {
        auto res = py::array_t<float>({ batch_size });
        auto res_acc = res.mutable_unchecked<1>();

        for (int k = 0; k < batch_size; ++k) {
            res_acc(k) = (float) lus[k].determinant().logabs();
        }

        return res;
    }

    py::array_t<float> inv(bool zero_pad) {
        auto res = py::array_t<float>({ batch_size, dim1, dim2 });

        if (zero_pad) {
            std::fill(res.mutable_data(), res.mutable_data() + res.size(), float{});
        }

        auto res_acc = res.mutable_unchecked<3>();
        for (int k = 0; k < batch_size; ++k) {
            auto dk = lengths[k];
            Eigen::MatrixXlogd Xinv = lus[k].inverse();
            for (int i = 0; i < dk; ++i) {
                for (int j = 0; j < dk; ++j) {
                    res_acc(k, i, j) = (float) Xinv(i, j).as_float();
                }
            }
        }

        return res;
    }

private:
    std::vector<Eigen::FullPivLU<Eigen::MatrixXlogd>> lus;
    std::vector<int> lengths;
    py::ssize_t batch_size;
    py::ssize_t dim1;
    py::ssize_t dim2;
};


PYBIND11_MODULE(lu, m) {

    py::class_<log_domain_lu>(m, "LogDomainLU")
        .def(py::init<const Eigen::MatrixXf&>())
        .def("logdet",
             &log_domain_lu::logdet)
        .def("inv",
             &log_domain_lu::inv,
             py::return_value_policy::move);

    py::class_<batch_log_domain_lu>(m, "BatchLogDomainLU")
        .def(py::init<py::array_t<float>, std::vector<int>>())
        .def("logdet",
             &batch_log_domain_lu::logdet,
             py::return_value_policy::move)
        .def("inv",
             &batch_log_domain_lu::inv,
             py::arg("zero_pad") = true,
             py::return_value_policy::move);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
