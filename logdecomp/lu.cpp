#include <torch/extension.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "logval.h"

using namespace torch::indexing;
namespace py = pybind11;

typedef LogVal<double> LogValD;
namespace Eigen {
    typedef Eigen::Matrix<LogValD, Dynamic, Dynamic> MatrixXlogd;
    typedef Eigen::Matrix<LogValD, Dynamic, 1> VectorXlogd;
    typedef Eigen::Matrix<double, Dynamic, 1> VectorXd;
    typedef Eigen::Matrix<double, Dynamic, Dynamic> MatrixXd;
}


auto torch_to_eigen_log(torch::Tensor X)
{
    TORCH_CHECK(X.size(0) == X.size(1), "X must be square");

    auto d = X.size(0);
    torch::TensorAccessor<float, 2> X_acc{X.accessor<float, 2>()};
    Eigen::MatrixXlogd eigX(d, d);

    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            eigX(i, j) = LogValD((double) X_acc[i][j], false);

    return eigX;
}


class log_domain_lu {
public:
    log_domain_lu(torch::Tensor X): lu(torch_to_eigen_log(X)) { }

    float logdet() {
        return lu.determinant().logabs();
    }

    auto inv() {
        Eigen::MatrixXlogd Xinv = lu.inverse();
        Eigen::MatrixXd Xinvd = Xinv.unaryExpr([](const LogValD& lv) {
            return lv.as_float();
        });
        return Xinvd;
    }

private:
    Eigen::FullPivLU<Eigen::MatrixXlogd> lu;
};


class batch_log_domain_lu {
public:
    batch_log_domain_lu(torch::Tensor X, std::vector<int> lengths)
    : lengths{lengths}
    , shape{X.sizes()} {

        for (int k = 0; k < lengths.size(); ++k) {
            auto slice = Slice(0, lengths[k]);
            auto Xk = X.index({k, slice, slice});
            lus.emplace_back(torch_to_eigen_log(Xk));
        }

    }

    at::Tensor logdet() {
        auto res = at::empty({lengths.size()}); // CPU float;
        auto res_acc = res.accessor<float, 1>();
        for (int k = 0; k < lengths.size(); ++k) {
            res_acc[k] = lus[k].determinant().logabs();
        }
        return res;
    }

    auto inv() {
        auto res = at::zeros(shape);
        auto res_acc = res.accessor<float, 3>();

        for (int k = 0; k < lengths.size(); ++k) {

            Eigen::MatrixXlogd Xinv = lus[k].inverse();

            // TODO: possible / worth it to avoid for loop?
            for (int i = 0; i < lengths[k]; ++i) {
                for (int j = 0; j < lengths[k]; ++j) {
                    res_acc[k][i][j] = Xinv(i, j).as_float();
                }
            }
        }

        return res;
    }

private:
    std::vector<Eigen::FullPivLU<Eigen::MatrixXlogd>> lus;
    std::vector<int> lengths;
    at::IntArrayRef shape;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::class_<batch_log_domain_lu>(m, "BatchLogDomainLU")
        .def(py::init<torch::Tensor, std::vector<int>>())
        .def("logdet", &batch_log_domain_lu::logdet)
        .def("inv", &batch_log_domain_lu::inv);

    py::class_<log_domain_lu>(m, "LogDomainLU")
        .def(py::init<torch::Tensor>())
        .def("logdet", &log_domain_lu::logdet)
        .def("inv", &log_domain_lu::inv);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
