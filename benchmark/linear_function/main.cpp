#include <SaddlePoint/LMSolver.h>
#include <SaddlePoint/EigenSolverWrapper.h>
#include <SaddlePoint/check_traits.h>
#include <SaddlePoint/DiagonalDamping.h>
#include <iostream>
#include <Eigen/core>


typedef SaddlePoint::EigenSolverWrapper<Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > > LinearSolver;

#define m 1000
#define n 250

class LineaFunctionTraits{
public:
    Eigen::VectorXi JRows, JCols;
    int xSize;
    int ESize;

    Eigen::MatrixXd A;
    
    void init()
    {
        using namespace Eigen;
        xSize=n;
        A.resize(m,n);
        A<<Matrix<double, n, n>::Identity()-MatrixXd::Constant(n,n,2.0/(double)m),
            -MatrixXd::Constant(m-n, n,2.0/(double)m);

        JRows.resize(m*n);
        JCols.resize(m*n);
        for (int i=0;i<m;i++){
            for (int j=0;j<n;j++){
                JRows(n*i+j)=i;
                JCols(n*i+j)=j;
            }
        }
    }
    
    void initial_solution(Eigen::VectorXd& x0){
        x0=Eigen::VectorXd::Constant(n, 1.0);
    }
    void pre_iteration(const Eigen::VectorXd& prevx){}
    bool post_iteration(const Eigen::VectorXd& x){return false;}
    void objective(const Eigen::VectorXd& x, Eigen::VectorXd& EVec){
      if (EVec.size()!=m)
        EVec.resize(m);
        EVec<<A*x-Eigen::VectorXd::Constant(m, 1.0);
    }
    void jacobian(const Eigen::VectorXd& x, Eigen::VectorXd& JVals){
        if (JVals.size()!=m*n)
          JVals.resize(m*n);
        for (int i=0;i<m;i++){
            for (int j=0;j<n;j++){
                JVals(n*i+j)=A(i,j);
            }
        }
    }
    bool post_optimization(const Eigen::VectorXd& x){
        //std::cout<<"x:"<<x<<std::endl;
        return true;
    }
};



LineaFunctionTraits slTraits;
LinearSolver lSolver;
SaddlePoint::DiagonalDamping<LineaFunctionTraits> dTraits;
SaddlePoint::LMSolver<LinearSolver,LineaFunctionTraits, SaddlePoint::DiagonalDamping<LineaFunctionTraits> > lmSolver;



int main(int argc, char *argv[])
{
    
    // Load a mesh in OFF format
    using namespace std;
    using namespace Eigen;
    
    slTraits.init();
    lmSolver.init(&lSolver, &slTraits, &dTraits, 1000);
    SaddlePoint::check_traits(slTraits);
    lmSolver.solve(true);
    
    return 0;
}
