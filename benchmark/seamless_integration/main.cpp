#include <SaddlePoint/LMSolver.h>
#include <SaddlePoint/EigenSolverWrapper.h>
#include <SaddlePoint/check_traits.h>
#include <SaddlePoint/DiagonalDamping.h>
#include <SIInitialSolutionTraits.h>
#include <IterativeRoundingTraits.h>
#include <iostream>
#include <Eigen/core>
#include <igl/serialize.h>


typedef SaddlePoint::EigenSolverWrapper<Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > > LinearSolver;




SIInitialSolutionTraits<LinearSolver> slTraits;
LinearSolver lSolver1,lSolver2;
SaddlePoint::DiagonalDamping<SIInitialSolutionTraits<LinearSolver>> dISTraits(0.01);
SaddlePoint::LMSolver<LinearSolver,SIInitialSolutionTraits<LinearSolver>, SaddlePoint::DiagonalDamping<SIInitialSolutionTraits<LinearSolver> > > initialSolutionLMSolver;

IterativeRoundingTraits<LinearSolver> irTraits;
SaddlePoint::DiagonalDamping<IterativeRoundingTraits<LinearSolver>> dIRTraits(0.01);
SaddlePoint::LMSolver<LinearSolver,IterativeRoundingTraits<LinearSolver>, SaddlePoint::DiagonalDamping<IterativeRoundingTraits<LinearSolver> > > iterativeRoundingLMSolver;


int main(int argc, char *argv[])
{
  
  // Load a mesh in OFF format
  using namespace std;
  using namespace Eigen;
  
  std::string filename("/Users/amirvaxman/PatternsParam/build/Release/poisson.srl");
  igl::deserialize(slTraits.A,"A",filename);
  igl::deserialize(slTraits.x0,"x0",filename);
  igl::deserialize(slTraits.rawField,"rawField",filename);
  igl::deserialize(slTraits.fixedIndices,"fixedIndices",filename);
  igl::deserialize(slTraits.fixedValues,"fixedValues",filename);
  igl::deserialize(slTraits.singularIndices,"singularIndices",filename);
  igl::deserialize(slTraits.b,"b",filename);
  igl::deserialize(slTraits.C,"C",filename);
  igl::deserialize(slTraits.G,"G",filename);
  igl::deserialize(slTraits.FN,"FN",filename);
  igl::deserialize(slTraits.N,"N",filename);
  igl::deserialize(slTraits.lengthRatio, "lengthRatio",filename);
  //mw.save(length,"gradLength");
  igl::deserialize(slTraits.n,"n",filename);
  //mw.save(pd.intSpanMat,"intSpanMat");
  igl::deserialize(slTraits.V,"V",filename);
  igl::deserialize(slTraits.F,"F",filename);
  igl::deserialize(slTraits.x2CornerMat,"x2CornerMat",filename);
  igl::deserialize(slTraits.integerIndices,"integerIndices",filename);
  
  //initial solution
  slTraits.init();
  //Eigen::VectorXd JVals;
  //slTraits.jacobian(slTraits.initXandFieldSmall, JVals);
  initialSolutionLMSolver.init(&lSolver1, &slTraits, &dISTraits, 1000);
  //SaddlePoint::check_traits(slTraits, slTraits.initXandFieldSmall);
  initialSolutionLMSolver.solve(true);
  cout<<"(x-x0).lpNorm<Infinity>(): "<<(initialSolutionLMSolver.x-initialSolutionLMSolver.x0).lpNorm<Infinity>()<<endl;
  
  //Iterative rounding
  irTraits.init(slTraits, initialSolutionLMSolver.x);
  iterativeRoundingLMSolver.init(&lSolver2, &irTraits, &dIRTraits, 1000);
  //SaddlePoint::check_traits(irTraits, irTraits.x0Small);
  iterativeRoundingLMSolver.solve(true);
  
  return 0;
}
