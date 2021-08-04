#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <functional>

#include <Eigen/Dense>

#include "logval.h"

namespace py = pybind11;


typedef LogVal<double> LogValD;
namespace Eigen {
    typedef Eigen::Matrix<LogValD, Dynamic, Dynamic> MatrixXlogd;
    typedef Eigen::Matrix<float, Dynamic, Dynamic> MatrixXf;

}

Eigen::MatrixXlogd to_log(const Eigen::MatrixXf& X) {
    // the one-liner unaryExpr fails on windows. Maybe for good reason?
    //return X.unaryExpr([](float f) { return LogValD((double) f, false); });
    Eigen::MatrixXlogd res(X.rows(), X.cols());
    for (py::ssize_t i = 0; i < X.rows(); ++i) {
        for (py::ssize_t j = 0; j < X.cols(); ++j) {
            res(i, j) = LogValD((double) X(i, j), false);
        }
    }
    return res;
}

class log_domain_lu {
public:
    log_domain_lu(const Eigen::MatrixXf& X): lu(to_log(X)) { }

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
        const std::vector<int>& lengths,
        const py::array_t<bool>& sign)
    : lengths{ lengths }
    {
        auto X_acc = X.unchecked<3>();

        // there are three possibilities for the sign:
        // ndim=0 means sign is the same everywhere.
        // ndim=2 means sign at k,i,j is sign(i,j)
        // ndim=3 means sign at k,i,j is sign(k,i,j)
        // Dynamically design a sign extraction function for ecah case.

        std::function<bool (py::ssize_t, py::ssize_t, py::ssize_t)> _sign;
        auto sign_buf = sign.request();

        if (sign_buf.ndim == 0) {
            auto sign_val = static_cast<bool*>(sign_buf.ptr)[0];
            _sign = [sign_val](py::ssize_t k, py::ssize_t i, py::ssize_t j) { return sign_val; };
        } else if (sign_buf.ndim == 2) {
            auto sign_2d = sign.unchecked<2>();
            _sign = [sign_2d](py::ssize_t k, py::ssize_t i, py::ssize_t j) { return sign_2d(i, j); };
        } else if (sign_buf.ndim == 3) {
            auto sign_3d = sign.unchecked<3>();
            _sign = [sign_3d](py::ssize_t k, py::ssize_t i, py::ssize_t j) { return sign_3d(k, i, j); };
        } else {
            std::runtime_error("wrong dimension");
        }

        batch_size = X_acc.shape(0);
        dim1 = X_acc.shape(1);
        dim2 = X_acc.shape(2);


        for (py::ssize_t k = 0; k < batch_size; ++k) {

            auto dk = lengths[k];

            // pass 1. extract min
            float xkmax = X_acc(k, 0, 0);
            for (py::ssize_t i = 0; i < dk; ++i) {
                for (py::ssize_t j = 0; j < dk; ++j) {
                    xkmax = static_cast<double>(std::max(xkmax, X_acc(k, i, j)));
                }
            }

            xmax.push_back(xkmax);

            Eigen::MatrixXlogd Xk(dk, dk);
            for (py::ssize_t i = 0; i < dk; ++i) {
                for (py::ssize_t j = 0; j < dk; ++j) {
                    // upcast to avoid underflow
                    auto val = static_cast<double>(X_acc(k, i, j)) - xkmax;
                    Xk(i, j) = LogValD(val, _sign(k, i, j));
                }
            }
            lus.emplace_back(Xk);
        }
    }

    py::array_t<float> logdet() {
        auto res = py::array_t<float>({ batch_size });
        auto res_acc = res.mutable_unchecked<1>();

        for (py::ssize_t k = 0; k < batch_size; ++k) {
            double val = lus[k].determinant().logabs() + lengths[k] * xmax[k];
            res_acc(k) = static_cast<float>(val);
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
            LogValD exp_xkmax(xmax[k], false);
            Eigen::MatrixXlogd Xinv = lus[k].inverse();
            for (int i = 0; i < dk; ++i) {
                for (int j = 0; j < dk; ++j) {
                    LogValD exp_uij = Xinv(i, j);
                    exp_uij /= exp_xkmax;
                    res_acc(k, i, j) = static_cast<float>(exp_uij.as_float());
                }
            }
        }

        return res;
    }

private:
    std::vector<Eigen::FullPivLU<Eigen::MatrixXlogd>> lus;
    std::vector<int> lengths;
    std::vector<float> xmax;
    py::ssize_t batch_size;
    py::ssize_t dim1;
    py::ssize_t dim2;
};


PYBIND11_MODULE(lu, m) {

    py::class_<log_domain_lu>(m, "LogDomainLU")
        .def(py::init<const Eigen::MatrixXf&>(),
             py::arg().noconvert().none(false))
        .def("logdet",
             &log_domain_lu::logdet)
        .def("inv",
             &log_domain_lu::inv,
             py::return_value_policy::move);

    py::class_<batch_log_domain_lu>(m, "BatchLogDomainLU")
        .def(py::init<py::array_t<float>,
                      std::vector<int>,
                      py::array_t<bool>>(),
             py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert())
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
