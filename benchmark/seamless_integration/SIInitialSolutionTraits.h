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
#include <igl/diag.h>
#include "sparse_block.h"


template <class LinearSolver>
class SIInitialSolutionTraits{
public:
  
  
  Eigen::VectorXi JRows, JCols;
  int xSize;
  int ESize;
  
  Eigen::SparseMatrix<double> A,C,G,G2, UFull, x2CornerMat, UExt;
  Eigen::MatrixXd rawField, rawField2, FN, V,B1,B2, origFieldVolumes,SImagField;
  Eigen::MatrixXi F;
  Eigen::VectorXd b,xPoisson, fixedValues, x0, initXandFieldSmall,rawField2Vec,rawFieldVec;
  Eigen::VectorXi fixedIndices, integerIndices, singularIndices;
  Eigen::MatrixXi IImagField, JImagField;
  int N,n;
  double lengthRatio, paramLength;
  double wIntegration,wConst, wBarrier, wClose, s;
  
  bool pre_optimization(Eigen::VectorXd& prev){return true;}
  void initial_solution(Eigen::VectorXd& _x0){
    _x0 = initXandFieldSmall;
  }
  void pre_iteration(const Eigen::VectorXd& prevx){}
  bool post_iteration(const Eigen::VectorXd& x){return false;}
  
  void objective(const Eigen::VectorXd& xAndCurrFieldSmall,  Eigen::VectorXd& EVec){
    using namespace std;
    using namespace Eigen;
    
    VectorXd xAndCurrField = UExt*xAndCurrFieldSmall;
    VectorXd xcurr=xAndCurrField.head(x0.size());
    VectorXd currField=xAndCurrField.tail(rawField2Vec.size());
    
    //cout<<"currField.tail(100): "<<currField.tail(100)<<endl;
    
    VectorXd fIntegration = (currField - paramLength*G2*xcurr);
    VectorXd fClose = currField-rawField2Vec;
    
    VectorXd fConst(fixedIndices.size());
    for (int i=0;i<fixedIndices.size();i++)
      fConst(i) = xcurr(fixedIndices(i))-fixedValues(i);
    
    VectorXd fBarrier = VectorXd::Zero(N*FN.rows());
    
    for (int i=0;i<FN.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2d currVec=currField.segment(2*N*i+2*j,2);
        RowVector2d nextVec=currField.segment(2*N*i+2*((j+1)%N),2);
        double imagProduct = (currVec(0)*nextVec(1) - currVec(1)*nextVec(0))/origFieldVolumes(i,j);
        double barResult = (imagProduct/s)*(imagProduct/s)*(imagProduct/s) - 3.0*(imagProduct/s)*(imagProduct/s) + 3.0*(imagProduct/s);
        double barResult2 = 1.0/barResult - 1.0;
        if (imagProduct<=0) barResult2 = std::numeric_limits<double>::infinity();
        if (imagProduct>=s) barResult2 = 0.0;
        fBarrier(N*i+j)=barResult2;
      }
    }
    
    EVec.conservativeResize(fIntegration.size()+fClose.size()+fConst.size()+fBarrier.size());
    EVec<<fIntegration*wIntegration,fClose*wClose,fConst*wConst,fBarrier*wBarrier;
    //EVec.conservativeResize(fBarrier.size());
    //EVec<<fBarrier;
    
    //cout<<"fIntegration.head(10): "<<fIntegration.head(10)<<endl;
    //cout<<"wIntegration: "<<wIntegration<<endl;
    
  }
  
  
  void jacobian(const Eigen::VectorXd& xAndCurrFieldSmall, Eigen::VectorXd& JVals){
    using namespace Eigen;
    using namespace std;
    
    VectorXd xAndCurrField = UExt*xAndCurrFieldSmall;
    VectorXd xcurr=xAndCurrField.head(x0.size());
    VectorXd currField=xAndCurrField.tail(rawField2Vec.size());
    
    //integration
    SparseMatrix<double> gIntegration;
    vector<Triplet<double>> gIntegrationTriplets;
    for (int k=0; k<G2.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(G2,k); it; ++it)
        gIntegrationTriplets.push_back(Triplet<double>(it.row(), it.col(), -paramLength*it.value()));
    
    for (int i=0;i<currField.size();i++)
      gIntegrationTriplets.push_back(Triplet<double>(i,G2.cols()+i,1.0));
    
    gIntegration.resize(G2.rows(), G2.cols()+currField.size());
    gIntegration.setFromTriplets(gIntegrationTriplets.begin(), gIntegrationTriplets.end());
    gIntegration=gIntegration*UExt;
    
    //closeness
    SparseMatrix<double> gClose(currField.size(), xAndCurrField.size());
    vector<Triplet<double>> gCloseTriplets;
    for (int i=0;i<currField.size();i++)
      gCloseTriplets.push_back(Triplet<double>(i,x0.size()+i,1.0));
    
    gClose.setFromTriplets(gCloseTriplets.begin(), gCloseTriplets.end());
    gClose=gClose*UExt;
    
    //fixedIndices constness
    SparseMatrix<double> gConst(fixedIndices.size(), xAndCurrField.size());
    vector<Triplet<double>> gConstTriplets;
    for (int i=0;i<fixedIndices.size();i++)
      gConstTriplets.push_back(Triplet<double>(i,fixedIndices(i),1.0));
    
    gConst.setFromTriplets(gConstTriplets.begin(), gConstTriplets.end());
    gConst=gConst*UExt;
    
    //barrier
    VectorXd splineDerivative= VectorXd::Zero(N*FN.rows(),1);
    VectorXd fBarrier = VectorXd::Zero(N*FN.rows());
    VectorXd barSpline = VectorXd::Zero(N*FN.rows());
    SImagField.conservativeResize(IImagField.rows(), IImagField.cols());
    for (int i=0;i<FN.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2d currVec=currField.segment(2*N*i+2*j,2);
        RowVector2d nextVec=currField.segment(2*N*i+2*((j+1)%N),2);
        double imagProduct = (currVec(0)*nextVec(1) - currVec(1)*nextVec(0))/origFieldVolumes(i,j);
        double barResult = (imagProduct/s)*(imagProduct/s)*(imagProduct/s) - 3.0*(imagProduct/s)*(imagProduct/s) + 3.0*(imagProduct/s);
        double barResult2 = 1.0/barResult - 1.0;
        if (imagProduct<=0) barResult2 = std::numeric_limits<double>::infinity();
        if (imagProduct>=s) barResult2 = 0.0;
        fBarrier(N*i+j)=barResult2;
        barSpline(N*i+j)=barResult;
        
        double splineDerivativeLocal=3.0*(imagProduct*imagProduct/(s*s*s)) -6.0*(imagProduct/(s*s)) + 3.0/s;
        if (imagProduct<=0) splineDerivativeLocal=std::numeric_limits<double>::infinity();
        if (imagProduct>=s) splineDerivativeLocal=0.0;
        splineDerivative(N*i+j)=splineDerivativeLocal;
        
        SImagField.row(N*i+j)<<nextVec(1)/origFieldVolumes(i,j), -nextVec(0)/origFieldVolumes(i,j), -currVec(1)/origFieldVolumes(i,j),currVec(0)/origFieldVolumes(i,j);
        
      }
    }
    
    /*cout<<"xAndCurrFieldSmall: "<<xAndCurrFieldSmall<<endl;
    
    cout<<"IImagField: "<<IImagField<<endl;
    cout<<"JImagField: "<<JImagField<<endl;
    cout<<"SImagField: "<<SImagField<<endl;*/
    SparseMatrix<double> gImagField(N*FN.rows(), currField.size());
    vector<Triplet<double>> gImagTriplets;
    //cout<<"IImagField.maxCoeff(): "<<IImagField.maxCoeff()<<endl;
    //cout<<"JImagField.maxCoeff(): "<<JImagField.maxCoeff()<<endl;
    for (int i=0;i<IImagField.rows();i++)
      for (int j=0;j<IImagField.cols();j++)
        gImagTriplets.push_back(Triplet<double>(IImagField(i,j), JImagField(i,j), SImagField(i,j)));
    
    gImagField.setFromTriplets(gImagTriplets.begin(), gImagTriplets.end());
    
    //cout<<"fBarrier(11): "<<fBarrier(11)<<endl;
    //cout<<"fBarrierDerivative(11): "<<fBarrierDerivative(11)<<endl;
    SparseMatrix<double> gFieldReduction = gClose;
    VectorXd barDerVec=-splineDerivative.array()/((barSpline.array()*barSpline.array()).array());
    //cout<<"barDerVec(11): "<<barDerVec(11)<<endl;
    /*barDerVec(fBarrier==Inf)=Inf;
    barDerVec(isinf(barDerVec))=0;
    barDerVec(isnan(barDerVec))=0;*/
    for (int i=0;i<fBarrier.size();i++)
      if (std::abs(fBarrier(i))<10e-9)
        barDerVec(i)=0.0;
    else if (fBarrier(i)==std::numeric_limits<double>::infinity())
      barDerVec(i)=std::numeric_limits<double>::infinity();
    
    SparseMatrix<double> gBarrierFunc(barDerVec.size(), barDerVec.size());
    vector<Triplet<double>> gBarrierFuncTris;
    for (int i=0;i<barDerVec.size();i++)
      gBarrierFuncTris.push_back(Triplet<double>(i,i,barDerVec(i)));
    gBarrierFunc.setFromTriplets(gBarrierFuncTris.begin(), gBarrierFuncTris.end());
    
    SparseMatrix<double> gBarrier = gBarrierFunc*gImagField*gFieldReduction;
    
    MatrixXi blockIndices(4,1);
    blockIndices<<0,1,2,3;
    vector<SparseMatrix<double>> JMats;
    JMats.push_back(gIntegration*wIntegration);
    JMats.push_back(gClose*wClose);
    JMats.push_back(gConst*wConst);
    JMats.push_back(gBarrier*wBarrier);
    SparseMatrix<double> J;
    SaddlePoint::sparse_block(blockIndices, JMats,J);
    //J=gBarrier;
    //cout<<"gBarrier: "<<gBarrier<<endl;
    /*cout<<"gIntegration.rows(): "<<gIntegration.rows()<<endl;
    cout<<"gClose.rows(): "<<gClose.rows()<<endl;
    cout<<"gConst.rows(): "<<gConst.rows()<<endl;
    cout<<"gBarrier.rows(): "<<gBarrier.rows()<<endl;*/
    
    JRows.conservativeResize(J.nonZeros());
    JCols.conservativeResize(J.nonZeros());
    JVals.conservativeResize(J.nonZeros());
    int counter=0;
    SparseMatrix<double> JT=J.transpose();
    for (int k=0; k<JT.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(JT,k); it; ++it){
        JRows(counter)=it.col();
        JCols(counter)=it.row();
        JVals(counter++)=it.value();
      }
    
    //cout<<"JRows.maxCoeff(): "<<JRows.maxCoeff()<<endl;
    //cout<<"JCols.maxCoeff(): "<<JCols.maxCoeff()<<endl;
    //cout<<"done!" <<endl;
    
    
  }
  bool post_optimization(const Eigen::VectorXd& x){
    //std::cout<<"x:"<<x<<std::endl;
  
    return true;
  }
  
  
  void init(){
    using namespace std;
    using namespace Eigen;
    
    wIntegration=10e3;
    wConst=10e3;
    wBarrier=0.0001;
    wClose=1;
    s=0.1;
    
    
    paramLength = (V.colwise().maxCoeff()-V.colwise().minCoeff()).norm()*lengthRatio;
    
    //normalizing field and putting extra in paramLength
    double avgGradNorm=0;
    for (int i=0;i<F.rows();i++)
      for (int j=0;j<N;j++)
        avgGradNorm+=rawField.block(i,3*j,1,3).norm();
    
    avgGradNorm/=(double)(N*F.rows());
    
    rawField.array()/=avgGradNorm;
    paramLength/=avgGradNorm;
    
    igl::local_basis(V,F,B1,B2, FN);
    //cout<<"B1.block(0,0,10,3): "<<B1.block(0,0,10,3)<<endl;
    //cout<<"B2.block(0,0,10,3): "<<B2.block(0,0,10,3)<<endl;
    
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
    
    //for (int k=0; k<reducMat.outerSize(); ++k)
    //     for (SparseMatrix<double>::InnerIterator it(reducMat,k); it; ++it)
     //      cout<<it.row()<<","<<it.col()<<","<<it.value()<<";"<<endl;
    
    //Reducing constraint matrix
    VectorXi I(C.nonZeros()),J(C.nonZeros());
    VectorXd S(C.nonZeros());
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
    
    //cout<<"I: "<<I<<endl;
    //cout<<"J: "<<J<<endl;
    // cout<<"S: "<<S<<endl;
    /**
    //creating small dense matrix with all non-zero columns
    VectorXi uniqueJVec(uniqueJ.size());
    VectorXi JMask=VectorXi::Constant(C.cols(),-1);
    counter=0;
    for (set<int>::iterator ji=uniqueJ.begin();ji!=uniqueJ.end();ji++){
      uniqueJVec(counter)=*ji;
      JMask(*ji)=counter++;
    }
    
    //cout<<"uniqueJVec: "<<uniqueJVec<<endl;
    
    MatrixXd CSmall=MatrixXd::Zero(C.rows(), JMask.maxCoeff()+1);
    for (int i=0;i<I.rows();i++)
      CSmall(I(i),JMask(J(i)))=S(i);
    
    //cout<<"CSmall: "<<CSmall<<endl;
    
    
    FullPivLU<MatrixXd> lu_decomp(CSmall);
    MatrixXd USmall=lu_decomp.kernel();
    
    //cout<<"USmall: "<<USmall<<endl;
    
    
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
        URawTriplets.push_back(Triplet<double>(nonPartIndices.size()+i,nonPartIndices.size()+j,USmall(i,j)));
    
    URaw.setFromTriplets(URawTriplets.begin(), URawTriplets.end());
    
    SparseMatrix<double> permMat(URaw.rows(),URaw.rows());
    vector<Triplet<double>> permMatTriplets;
    for (int i=0;i<nonPartIndices.size();i++)
      permMatTriplets.push_back(Triplet<double>(nonPartIndices(i),i,1.0));
    
    for (int i=0;i<uniqueJVec.size();i++)
      permMatTriplets.push_back(Triplet<double>(uniqueJVec(i),nonPartIndices.size()+i,1.0));
    
    permMat.setFromTriplets(permMatTriplets.begin(), permMatTriplets.end());
    
    UFull=permMat*URaw;
     **/
    
    //cout<<"(C*UFull).lpNorm<Infinity>(): "<<(C*UFull).norm()<<endl;
    
    //computing original volumes and (row,col) of that functional
    
    rawField2.resize(F.rows(),2*N);
    for (int i=0;i<N;i++)
      rawField2.middleCols(2*i,2)<<rawField.middleCols(3*i,3).cwiseProduct(B1).rowwise().sum(),rawField.middleCols(3*i,3).cwiseProduct(B2).rowwise().sum();
    
    rawFieldVec.resize(3*N*F.rows());
    rawField2Vec.resize(2*N*F.rows());
    for (int i=0;i<rawField.rows();i++){
      rawFieldVec.segment(3*N*i,3*N)=rawField.row(i).transpose();
      rawField2Vec.segment(2*N*i,2*N)=rawField2.row(i).transpose();
    }
    
    //cout<<"rawField2Vec: "<<rawField2Vec<<endl;
    
    
    IImagField.resize(N*F.rows(),4);
    JImagField.resize(N*F.rows(),4);
    origFieldVolumes.resize(F.rows(),N);
    for (int i=0;i<F.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2d currVec=rawField2.block(i,2*j,1,2);
        RowVector2d nextVec=rawField2.block(i,2*((j+1)%N),1,2);
        IImagField.row(i*N+j)=VectorXi::Constant(4,i*N+j);
        JImagField.row(i*N+j)<<2*N*i+2*j, 2*N*i+2*j+1, 2*N*i+2*((j+1)%N), 2*N*i+2*((j+1)%N)+1;
        origFieldVolumes(i,j)=currVec.norm()*nextVec.norm();//currVec(0)*nextVec(1)-currVec(1)*nextVec(0);
      }
    }
    
    cout<<"min origFieldVolumes: "<<origFieldVolumes.colwise().minCoeff()<<endl;
    cout<<"origFieldVolumes.row(0): "<<origFieldVolumes.row(0)<<endl;
    
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
    //cout<<"bigRhs: "<<bigRhs<<endl;
    
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int> >   solver;
    solver.analyzePattern(bigMat);
    solver.factorize(bigMat);
    if (solver.info()!=Eigen::Success){
      cout<<"Factorization of bigMat failed!!"<<endl;
      return;
    }
    VectorXd initXSmallFull = solver.solve(bigRhs);
    VectorXd initXSmall=initXSmallFull.head(UFull.cols());
    
    double initialIntegrationError = (rawField2Vec - paramLength*G2*UFull*initXSmall).template lpNorm<Infinity>();
    std::cout<<"initialIntegrationError: "<<initialIntegrationError<<std::endl;
    
    x0=UFull*initXSmall;
    
    initXandFieldSmall.resize(initXSmall.size()+rawField2.size());
    initXandFieldSmall<<initXSmall,rawField2Vec;
    
    vector<Triplet<double>> UExtTriplets;
    for (int k=0; k<UFull.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(UFull,k); it; ++it)
        UExtTriplets.push_back(Triplet<double>(it.row(), it.col(), it.value()));
    
    for (int k=0; k<rawField2.size(); k++)
      UExtTriplets.push_back(Triplet<double>(UFull.rows()+k,UFull.cols()+k,1.0));
    
    UExt.resize(UFull.rows()+rawField2Vec.size(), UFull.cols()+rawField2Vec.size());
    UExt.setFromTriplets(UExtTriplets.begin(), UExtTriplets.end());
    
    //restarting rows cols and vals
    VectorXd JVals;
    jacobian(Eigen::VectorXd::Random(UExt.cols()), JVals);
    
    xSize = UExt.cols();
    
  }
};



#endif /* SIInitialSolutionTraits_h */
