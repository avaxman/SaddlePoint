#include <SaddlePoint/LMSolver.h>
#include <SaddlePoint/EigenSolverWrapper.h>
#include <SaddlePoint/check_traits.h>
#include <SaddlePoint/DiagonalDamping.h>
#include <iostream>
#include <Eigen/core>


typedef SaddlePoint::EigenSolverWrapper<Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > > LinearSolver;

#define VALLEY_COEFF 25.0


class RosenbrockTraits{
public:
  Eigen::VectorXi JRows, JCols;
  int xSize;
  int ESize;
  
  void init()
  {
    xSize=2;
    ESize=2;
    
    JRows.resize(4);
    JCols.resize(4);
    
    //f1(x)
    JRows(0)=0;
    JCols(0)=0;
    JRows(1)=0;
    JCols(1)=1;
    
    //f2(x)
    JRows(2)=1;
    JCols(2)=0;
    JRows(3)=1;
    JCols(3)=1;
    
  }
  
  void initial_solution(Eigen::VectorXd& x0){
    x0<<-100,100;
  }
  void pre_iteration(const Eigen::VectorXd& prevx){}
  bool post_iteration(const Eigen::VectorXd& x){return false;}
  void objective(const Eigen::VectorXd& x,  Eigen::VectorXd& EVec){
    if (EVec.size()!=ESize)
      EVec.resize(ESize);
    
    EVec<<VALLEY_COEFF*(x(1)-x(0)*x(0)), 1.0-x(0);
  }
  void jacobian(const Eigen::VectorXd& x, Eigen::VectorXd& JVals){
    
    if (JVals.size()!=JRows.size())
      JVals.resize(JRows.size());
    JVals(0)=-2.0*VALLEY_COEFF*x(0);
    JVals(1)=VALLEY_COEFF;
    JVals(2)=-1.0;
    JVals(3)=0.0;
    
  }
  bool post_optimization(const Eigen::VectorXd& x){
    std::cout<<"x:"<<x<<std::endl;
    return true;
  }
};



RosenbrockTraits slTraits;
LinearSolver lSolver;
SaddlePoint::DiagonalDamping<RosenbrockTraits> dTraits(0.01);
SaddlePoint::LMSolver<LinearSolver,RosenbrockTraits, SaddlePoint::DiagonalDamping<RosenbrockTraits> > lmSolver;


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
