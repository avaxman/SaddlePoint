//
//  IterativeRoundingTraits.h
//  seamless_integration_bin
//
//  Created by Amir Vaxman on 17/02/2021.
//

#ifndef ITERATIVE_ROUNDING_TRAITS_H
#define ITERATIVE_ROUNDING_TRAITS_H

#include <igl/local_basis.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <igl/speye.h>
#include <igl/slice.h>
#include <igl/diag.h>
#include <SIInitialSolutionTraits.h>
#include "sparse_block.h"


template <class LinearSolver>
class IterativeRoundingTraits{
public:
  
  
  Eigen::VectorXi JRows, JCols;
  int xSize;
  int ESize;
  
  Eigen::SparseMatrix<double> A,C,G,G2, UFull, x2CornerMat, UExt;
  Eigen::MatrixXd rawField, rawField2, FN, V,B1,B2, origFieldVolumes,SImagField;
  Eigen::MatrixXi F;
  Eigen::VectorXd b,xPoisson, fixedValues, x0, x0Small, xCurrSmall, xPrevSmall,rawField2Vec,rawFieldVec, xCurr;
  Eigen::VectorXi fixedIndices, integerIndices, singularIndices;
  Eigen::MatrixXi IImagField, JImagField;
  int N,n;
  double lengthRatio, paramLength, fraction;
  double wConst, wBarrier, wClose, s, wPoisson;
  Eigen::VectorXi leftIndices;
  
  bool success;
  
  void initial_solution(Eigen::VectorXd& _x0){
    _x0 = x0Small;
  }
  
  bool pre_optimization(const Eigen::VectorXd& prevx){
    using namespace Eigen;
    using namespace std;
    xPrevSmall=prevx;
    xCurr=UFull*xCurrSmall;
    
    VectorXd roundDiffs(leftIndices.size());
    double minRoundDiff=3276700.0;
    int minRoundIndex=-1;
    for (int i=0;i<leftIndices.size();i++){
      cout<<"fraction*xCurr(leftIndices(i)): "<<fraction*xCurr(leftIndices(i))<<endl;
      cout<<"std::round(fraction*xCurr(leftIndices(i))): "<<std::round(fraction*xCurr(leftIndices(i)))<<endl;
      roundDiffs(i) = std::fabs(fraction*xCurr(leftIndices(i))-std::round(fraction*xCurr(leftIndices(i))));
      if (roundDiffs(i)<minRoundDiff){
        minRoundIndex=i;
        minRoundDiff=roundDiffs(i);
        cout<<"minRoundDiff: "<<minRoundDiff<<endl;
      }
    }
    
    double origValue = xCurr(leftIndices(minRoundIndex));
    double roundValue = std::round(fraction*xCurr(leftIndices(minRoundIndex)))/fraction;
    cout<<"origValue,roundValue: "<<origValue<<","<<roundValue<<endl;
    fixedIndices.conservativeResize(fixedIndices.size()+1);
    fixedIndices(fixedIndices.size()-1)=leftIndices[minRoundIndex];  //is this under-performing?
    fixedValues.conservativeResize(fixedValues.size()+1);
    fixedValues(fixedValues.size()-1)=roundValue;
   
    VectorXi newLeftIndices(leftIndices.size()-1);
    newLeftIndices.head(minRoundIndex)=leftIndices.head(minRoundIndex);
    newLeftIndices.tail(newLeftIndices.size()-minRoundIndex)=leftIndices.tail(newLeftIndices.size()-minRoundIndex);
    leftIndices=newLeftIndices;
    return (minRoundDiff>10e-7); //only proceeding if there is a need to round
  }
  void pre_iteration(const Eigen::VectorXd& prevx){}
  bool post_iteration(const Eigen::VectorXd& x){return false;}
  
  void objective(const Eigen::VectorXd& xCurrSmall,  Eigen::VectorXd& EVec){
    using namespace std;
    using namespace Eigen;
    
    /*xcurr = U*xcurrSmall;
    fObj = G2*U*xcurrSmall*paramLength - rawField2;
    gObj = G2*U*paramLength;
    %fObj = A*xcurr-b;
    %gObj = A*U;
    fClose = (xcurrSmall-xprevSmall);
    gClose=speye(length(xcurrSmall));
    %fLinConst=C*xcurr;

    fConst = (xcurr(fixedIndices)-fixedValues);

    nf = length(FN);
    currField = G2*xcurr*paramLength;
    fBarrier = zeros(N*nf,1);
    splineDerivative= zeros(N*nf,1);
    barSpline=zeros(N*nf,1);*/
    
    VectorXd xCurr = UFull*xCurrSmall;
    VectorXd fObj = G2*UFull*xCurrSmall*paramLength - rawField2Vec;
    VectorXd fClose = (xCurrSmall-xPrevSmall);
       
    VectorXd fConst(fixedIndices.size());
    for (int i=0;i<fixedIndices.size();i++)
      fConst(i) = xCurr(fixedIndices(i))-fixedValues(i);
    
    
    VectorXd currField = G2*xCurr*paramLength;
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
    
    EVec.conservativeResize(fObj.size()+fClose.size()+fConst.size()+fBarrier.size());
    EVec<<fObj*wPoisson,fClose*wClose,fConst*wConst,fBarrier*wBarrier;
    //EVec.conservativeResize(fBarrier.size());
    //EVec<<fBarrier;
  }
  
  
  void jacobian(const Eigen::VectorXd& xAndCurrFieldSmall, Eigen::VectorXd& JVals){
    using namespace Eigen;
    using namespace std;
    
    VectorXd xCurr = UFull*xCurrSmall;
    VectorXd currField = G2*xCurr*paramLength;
       
    //Poisson error
    SparseMatrix<double> gObj = G2*UFull*paramLength;
    
    //Closeness
    SparseMatrix<double> gClose;
    igl::speye(xCurrSmall.size(), gClose);
  
    
    //fixedIndices constness
    SparseMatrix<double> gConst(fixedIndices.size(), xCurr.size());
    vector<Triplet<double>> gConstTriplets;
    for (int i=0;i<fixedIndices.size();i++)
      gConstTriplets.push_back(Triplet<double>(i,fixedIndices(i),1.0));
    
    gConst.setFromTriplets(gConstTriplets.begin(), gConstTriplets.end());
    gConst=gConst*UFull;
    
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
    
    SparseMatrix<double> gBarrier = gBarrierFunc*gImagField*G2*UFull*paramLength;
  
    MatrixXi blockIndices(4,1);
    blockIndices<<0,1,2,3;
    vector<SparseMatrix<double>> JMats;
    JMats.push_back(gObj*wPoisson);
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
    
    xCurr = UFull*x;
    Eigen::VectorXd roundDiffs(fixedIndices.size());
    int minRoundDiff=3276700.0;
    int minRoundIndex=-1;
    for (int i=0;i<fixedIndices.size();i++){
      roundDiffs(i) = std::abs(fraction*xCurr(fixedIndices[i])-std::round(fraction*xCurr(fixedIndices[i])));
      if (roundDiffs(i)<minRoundDiff){
        minRoundIndex=i;
        minRoundDiff=roundDiffs(i);
      }
    }
    
    if (roundDiffs.maxCoeff()>10e-7){
      success=false;
      return true;  //terminate
    } else {
      return (leftIndices.size()==0);
    }
  }
  
  
  void init(const SIInitialSolutionTraits<LinearSolver>& sist, const Eigen::VectorXd& initCurrXandFieldSmall){
    using namespace std;
    using namespace Eigen;
    
    
    A=sist.A; C=sist.C; G=sist.G; G2=sist.G2; UFull=sist.UFull; x2CornerMat=sist.x2CornerMat; UExt=sist.UExt;
    rawField=sist.rawField; rawField2=sist.rawField2; FN=sist.FN, V=sist.V; B1=sist.B1; B2=sist.B2; origFieldVolumes=sist.origFieldVolumes; SImagField=sist.SImagField;
    F=sist.F;
    b=sist.b; xPoisson=sist.xPoisson; fixedValues=sist.fixedValues;  rawField2Vec=sist.rawField2Vec; rawFieldVec=sist.rawFieldVec;
    fixedIndices=sist.fixedIndices; integerIndices=sist.integerIndices; singularIndices=sist.singularIndices;
    IImagField=sist.IImagField; JImagField=sist.JImagField;
    N=sist.N,n=sist.n;
    lengthRatio=sist.lengthRatio; paramLength=sist.paramLength;
    wConst=sist.wConst, wBarrier=sist.wBarrier, wClose=sist.wClose, s=sist.s;
    
    wPoisson=1;
    wClose = 0.01;
    wConst=10e5;
    
    //Updating initial quantities
    
    
    VectorXd currXandField=UExt*initCurrXandFieldSmall;

    xSize=UFull.cols();
    x0=currXandField.head(UFull.rows());
    x0Small=initCurrXandFieldSmall.head(UFull.cols());
    rawField2Vec=currXandField.tail(2*N*FN.rows());
    
    rawField2.conservativeResize(FN.rows(),2*N);
    double avgGradNorm=0.0;
    for (int i=0;i<FN.rows();i++)
      rawField2.row(i)=rawField2Vec.segment(2*N*i,2*N);
    
    for (int i=0;i<FN.rows();i++)
      for (int j=0;j<N;j++)
        avgGradNorm+=rawField2.block(i,2*j,1,2).norm();
    
    avgGradNorm/=(double)(N*FN.rows());
    cout<<"avgGradNorm: "<<avgGradNorm<<endl;
    
    rawField.array()/=avgGradNorm;
    rawField2.array()/=avgGradNorm;
    x0.array()/=avgGradNorm;
    
    origFieldVolumes.resize(FN.rows(),N);
    for (int i=0;i<FN.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2d currVec=rawField2.block(i,2*j,1,2);
        RowVector2d nextVec=rawField2.block(i,2*((j+1)%N),1,2);
        
        origFieldVolumes(i,j)=currVec.norm()*nextVec.norm();//currVec(0)*nextVec(1)-currVec(1)*nextVec(0);
      }
    }
    
    
    cout<<"min origFieldVolumes: "<<origFieldVolumes.colwise().minCoeff()<<endl;

    fixedIndices=VectorXi::Zero(0);
    fixedValues=VectorXd::Zero(0.0);
    
    leftIndices=singularIndices;
    /*for (int i=0;i<singularIndices.size();i++)
      leftIndices.push_back(singularIndices.size());*/
      
    xCurrSmall=x0Small;
    xPrevSmall=xCurrSmall;
    fraction=1;
    
    VectorXd JVals;
    jacobian(Eigen::VectorXd::Random(UFull.cols()), JVals);
  }
};



#endif /* Iterative ROunding Traits */
