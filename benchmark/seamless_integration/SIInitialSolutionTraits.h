//
//  SIInitialSolutionTraits.h
//  seamless_integration_bin
//
//  Created by Amir Vaxman on 06/02/2021.
//

#ifndef SIInitialSolutionTraits_h
#define SIInitialSolutionTraits_h

#include <igl/local_basis.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <igl/speye.h>
#include <igl/slice.h>


template <class LinearSolver>
class SIInitialSolutionTraits{
public:
  
  
  Eigen::VectorXi JRows, JCols;
  int xSize;
  int ESize;
  
  Eigen::SparseMatrix<double> A,C,G,G2, UFull, x2CornerMat;
  Eigen::MatrixXd rawField, rawField2, FN, V,B1,B2, origFieldVolumes;
  Eigen::MatrixXi F;
  Eigen::VectorXd b,xPoisson, fixedValues, x0;
  Eigen::VectorXi fixedIndices, integerIndices, singularIndices;
  Eigen::MatrixXi IImagField, JImagField;
  int N,n;
  double lengthRatio, paramLength;
  double wintegration,wconst, wbarrier, wclose, s;
  
  void initial_solution(Eigen::VectorXd& _x0){
    _x0 = x0;
  }
  void pre_iteration(const Eigen::VectorXd& prevx){}
  bool post_iteration(const Eigen::VectorXd& x){return false;}
  void objective(const Eigen::VectorXd& x,  Eigen::VectorXd& EVec){
 
  }
  void jacobian(const Eigen::VectorXd& x, Eigen::VectorXd& JVals){
    
   
    
  }
  bool post_optimization(const Eigen::VectorXd& x){
    std::cout<<"x:"<<x<<std::endl;
    return true;
  }
  
  
  void init(){
    using namespace std;
    using namespace Eigen;
    
    wintegration=10e3;
    wconst=10e3;
    wbarrier=0.0001;
    wclose=1;
    s=0.1;
    
    paramLength = (V.colwise().maxCoeff()-V.colwise().minCoeff()).norm()*lengthRatio;
    
    igl::local_basis(V,F,B1,B2, FN);
    
    //creating G2
    vector<Triplet<double>> reducMatTris;
    SparseMatrix<double> reducMat;
    for (int i=0;i<F.rows();i++){
      for (int j=0;j<3;j++){
        for (int k=0;k<N;k++){
          reducMatTris.push_back(Triplet<double>(2*N*i+2*k,3*N*i+3*k+j,B1(i,j)));
          reducMatTris.push_back(Triplet<double>(2*N*i+2*k+1,3*N*i+3*k+j,B2(i,j)));
        }
      }
    }
    
    reducMat.resize(2*N*F.rows(), 3*N*F.rows());
    reducMat.setFromTriplets(reducMatTris.begin(), reducMatTris.end());
    G2=reducMat*G;
    
    //Reducing constraint matrix
    VectorXi I(C.nonZeros()),J(C.nonZeros()),S(C.nonZeros());
    set<int> uniqueJ;
    int counter=0;
    for (int k=0; k<C.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(C,k); it; ++it)
      {
        I(counter)=it.row();
        J(counter)=it.col();
        uniqueJ.insert(it.col());
        S(counter++)=it.value();
      }
    
    
    //creating small dense matrix with all non-zero columns
    VectorXi uniqueJVec(uniqueJ.size());
    VectorXi JMask=VectorXi::Constant(C.cols(),-1);
    counter=0;
    for (set<int>::iterator ji=uniqueJ.begin();ji!=uniqueJ.end();ji++){
      uniqueJVec(counter++)=*ji;
      JMask(*ji)=counter++;
    }
    
    MatrixXd CSmall(C.rows(), JMask.maxCoeff()+1);
    for (int i=0;i<I.rows();i++)
      CSmall(I(i),JMask(J(i)))=S(i);
    
    FullPivLU<MatrixXd> lu_decomp(CSmall);
    MatrixXd USmall=lu_decomp.kernel();
    
    
    //converting into the big matrix
    VectorXi nonPartIndices, stub;
    VectorXi allIndices(C.cols());
    for (int i=0;i<allIndices.size();i++) allIndices(i)=i;
    igl::setdiff(allIndices, uniqueJVec, nonPartIndices, stub);
    
    SparseMatrix<double> URaw(nonPartIndices.size()+USmall.rows(),nonPartIndices.size()+USmall.cols());
    vector<Triplet<double>> URawTriplets;
    for (int i=0;i<nonPartIndices.size();i++)
      URawTriplets.push_back(Triplet<double>(i,i,1.0));
    
    for (int i=0;i<USmall.rows();i++)
      for (int j=0;j<USmall.cols();j++)
        URawTriplets.push_back(Triplet<double>(nonPartIndices.size()+i,j,USmall(i,j)));
    
    URaw.setFromTriplets(URawTriplets.begin(), URawTriplets.end());
    
    SparseMatrix<double> permMat(URaw.rows()+nonPartIndices.size(),URaw.rows()+nonPartIndices.size());
    vector<Triplet<double>> permMatTriplets;
    for (int i=0;i<nonPartIndices.size();i++)
      permMatTriplets.push_back(Triplet<double>(nonPartIndices(i),i,1.0));
    
    for (int i=0;i<uniqueJVec.size();i++)
      permMatTriplets.push_back(Triplet<double>(nonPartIndices.size()+uniqueJVec(i),nonPartIndices.size()+i,1.0));
    
    permMat.setFromTriplets(permMatTriplets.begin(), permMatTriplets.end());
    
    UFull=permMat*URaw;
    
    //computing original volumes and (row,col) of that functional
    
    rawField2.resize(F.rows(),2*N);
    for (int i=0;i<N-1;i++)
      rawField2.middleCols(2*i,2)<<rawField.middleCols(3*i,3).cwiseProduct(B1).rowwise().sum(),rawField.middleCols(3*i,3).cwiseProduct(B2).rowwise().sum();
    
    VectorXd rawFieldVec(3*N*F.rows());
    for (int i=0;i<rawField.rows();i++)
      rawFieldVec.segment(3*i*F.rows(),3*N)=rawField.row(i).transpose();
    
    IImagField.resize(N*F.rows(),4);
    JImagField.resize(N*F.rows(),4);
    origFieldVolumes.resize(F.rows(),N);
    for (int i=0;i<F.rows();i++){
      IImagField.row(i)=VectorXi::Constant(4,i);
      for (int j=0;j<N;j++){
        RowVector2d currVec=rawField2.block(i,2*j,1,2);
        RowVector2d nextVec=rawField2.block(i,2*((j+1)%N),1,2);
        JImagField(i*N+j,2*j)=2*N*i+2*j;
        JImagField(i*N+j,2*j+1)=2*N*i+2*j+1;
        JImagField(i*N+j,2*j)=2*N*i+2*((j+1)%N);
        JImagField(i*N+j,2*j+1)=2*N*i+2*((j+1)%N)+1;
        origFieldVolumes(i*N+j)=currVec(0)*nextVec(1)-currVec(1)*nextVec(0);
      }
    }
    
    //Generating naive poisson solution
    SparseMatrix<double> Mx;
    igl::speye(3*N*F.rows(),Mx);   //TODO: change to correct masses
    SparseMatrix<double> L = G.transpose()*Mx*G;
    SparseMatrix<double> E = UFull.transpose()*G.transpose()*Mx*G*UFull;
    VectorXd f = UFull.transpose()*G.transpose()*Mx*(rawFieldVec/paramLength);
    SparseMatrix<double> constMat(fixedIndices.size(),UFull.cols());
    
    igl::slice(UFull, fixedIndices, 1,constMat);
    
    vector<Triplet<double>> bigMatTriplets;
    
    for (int k=0; k<E.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(E,k); it; ++it)
        bigMatTriplets.push_back(Triplet<double>(it.row(),it.col(), it.value()));
    
    for (int k=0; k<constMat.outerSize(); ++k){
      for (SparseMatrix<double>::InnerIterator it(constMat,k); it; ++it){
        bigMatTriplets.push_back(Triplet<double>(it.row()+E.rows(),it.col(), it.value()));
        bigMatTriplets.push_back(Triplet<double>(it.col(), it.row()+E.rows(), it.value()));
      }
    }
    
    SparseMatrix<double> bigMat(E.rows()+constMat.rows(),E.rows()+constMat.rows());
    bigMat.setFromTriplets(bigMatTriplets.begin(), bigMatTriplets.end());
    
    VectorXd bigRhs(f.size()+fixedValues.size());
    bigRhs<<f,fixedValues;
    
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int> >   solver;
    solver.analyzePattern(bigMat);
    solver.factorize(bigMat);
    VectorXd initXSmallFull = solver.solve(bigRhs);
    VectorXd initXSmall=initXSmallFull.head(UFull.cols());
    
    double initialIntegrationError = (rawField2 - paramLength*G2*UFull*initXSmall).template lpNorm<Infinity>();
    std::cout<<"initialIntegrationError: "<<initialIntegrationError<<std::endl;
    
    x0=UFull*initXSmall;
  
  }
};



#endif /* SIInitialSolutionTraits_h */
