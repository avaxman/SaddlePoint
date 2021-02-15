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
  
  void initial_solution(Eigen::VectorXd& _x0){
    _x0 = initXandFieldSmall;
  }
  void pre_iteration(const Eigen::VectorXd& prevx){}
  bool post_iteration(const Eigen::VectorXd& x){return false;}
  
  
  /*
   function [f,g]=initialPoissonObjective(xAndCurrFieldSmall,initField,sizeX, UExt,G2, N, FN, fixedIndices, fixedValues, paramLength, s, wintegration,wclose, wconst, wbarrier, IImagField, JImagField, origFieldVolumes)

   xAndCurrField = UExt*xAndCurrFieldSmall;
   xcurr=xAndCurrField(1:sizeX);
   currField=xAndCurrField(sizeX+1:end);

   fIntegration = (currField - paramLength*G2*xcurr);
   gIntegration = [-paramLength*G2, speye(length(currField))]*UExt;

   fClose = currField-initField;
   gClose = sparse(1:length(currField), sizeX+1:length(xAndCurrField), ones(1,length(currField)), length(currField), length(xAndCurrField))*UExt;

   %fLinConst=CExt*xAndCurrField;
   %gLinConst=CExt;

   fConst = xcurr(fixedIndices)-fixedValues;
   gConst = sparse((1:length(fixedIndices))', fixedIndices, ones(length(fixedIndices),1), length(fixedIndices),length(xAndCurrField))*UExt;

   nf = length(FN);

   fBarrier = zeros(N*nf,1);
   fBarrierDerivative= zeros(N*nf,1);

   SImagField=IImagField;

   for i=0:nf-1
       varOffset=i*2*N+1;
       barOffset=i*N+1;
       faceField = reshape(currField(varOffset:varOffset+2*N-1),2,N)';
       faceFieldNext = faceField([2:N,1],:);
       
       %tripleProducts = dot(repmat(FN(i+1,:),N,1), cross(faceField, faceField([2:end,1],:)),2);
       imagProduct = (faceField(:,1).*faceFieldNext(:, 2) - faceField(:,2).*faceFieldNext(:, 1))./origFieldVolumes(i+1,:)';
       barResult = (imagProduct/s).^3 - 3*(imagProduct/s).^2 + 3*(imagProduct/s);
       barResult2 = 1./barResult -1;
       barResult2(imagProduct<=0)=Inf;
       barResult2(imagProduct>=s)=0;
       fBarrier(barOffset:barOffset+N-1)=barResult2;
       
       barDerivative=(3*(imagProduct.^2/s^3) -6*(imagProduct/s^2) + 3/s)./origFieldVolumes(i+1,:)';
       barDerivative(imagProduct<=0)=Inf;
       barDerivative(imagProduct>=s)=0;
       fBarrierDerivative(barOffset:barOffset+N-1) =  barDerivative;
       
       %ImagField(barOffset:barOffset+N-1,1:2)=reshape(varOffset:varOffset+2*N-1, 2, N)';
       %JImagField(barOffset:barOffset+N-1,3:4)=JImagField([barOffset+1:barOffset+N-1,barOffset],1:2);
       
       SImagField(barOffset:barOffset+N-1,:)=[faceFieldNext(:, 2), -faceFieldNext(:, 1), -faceField(:,2),faceField(:,1)];
   end

   if (nargout<2) %don't compute jacobian
       f=[wintegration*fIntegration;wclose*fClose;wconst*fConst;wbarrier*fBarrier];
       return
   end


   gImagField=sparse(IImagField, JImagField, SImagField, N*nf, length(currField));
   gFieldReduction = gClose;
   barDerVec=-fBarrierDerivative./(fBarrier.^2);
   barDerVec(isinf(barDerVec))=0;
   barDerVec(isnan(barDerVec))=0;
   gBarrierFunc = spdiags(barDerVec, 0, length(fBarrier), length(fBarrier));   %./fBarrier.^2
   gBarrier=gBarrierFunc*gImagField*gFieldReduction;
   %f=sum(wobj*fAb.^2)+sum(wconst*fConst.^2)+sum(wbarrier*fBarrier.^2);
   %g=2*wobj*gAb'*fAb+2*wconst*gConst'*fConst+2*wbarrier*gBarrier'*fBarrier;
   f=[wintegration*fIntegration;wclose*fClose;wconst*fConst;wbarrier*fBarrier];
   g=[wintegration*gIntegration;wclose*gClose;wconst*gConst;wbarrier*gBarrier];
   end*/
   
  void objective(const Eigen::VectorXd& xAndCurrFieldSmall,  Eigen::VectorXd& EVec){
    using namespace std;
    using namespace Eigen;
    
    VectorXd xAndCurrField = UExt*xAndCurrFieldSmall;
    VectorXd xcurr=xAndCurrField.head(x0.size());
    VectorXd currField=xAndCurrField.tail(rawField2Vec.size());

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
        double imagProduct = (currVec(0)*nextVec(1) - currVec(1)*currVec(0))/origFieldVolumes(i,j);
        double barResult = (imagProduct/s)*(imagProduct/s)*(imagProduct/s) - 3.0*(imagProduct/s)*(imagProduct/s) + 3.0*(imagProduct/s);
        double barResult2 = 1.0/barResult - 1.0;
        if (imagProduct<=0) barResult2 = std::numeric_limits<double>::infinity();
        if (imagProduct>=s) barResult2 = 0.0;
        fBarrier(N*i+j)=barResult2;
      }
    }
    
    EVec.conservativeResize(fIntegration.size()+fClose.size()+fConst.size()+fBarrier.size());
    EVec<<fIntegration*wIntegration,fClose*wClose,fConst*wConst,fBarrier*wBarrier;
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
      gCloseTriplets.push_back(Triplet<double>(i,fixedIndices(i),1.0));
    
    gConst.setFromTriplets(gConstTriplets.begin(), gConstTriplets.end());
    gConst=gConst*
    UExt;
    
    //barrier
    VectorXd fBarrierDerivative= VectorXd::Zero(N*FN.rows(),1);
    VectorXd fBarrier = VectorXd::Zero(N*FN.rows());
    SImagField.conservativeResize(IImagField.rows(), IImagField.cols());
    for (int i=0;i<FN.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2d currVec=currField.segment(2*N*i+2*j,2);
        RowVector2d nextVec=currField.segment(2*N*i+2*((j+1)%N),2);
        double imagProduct = (currVec(0)*nextVec(1) - currVec(1)*currVec(0))/origFieldVolumes(i,j);
        double barResult = (imagProduct/s)*(imagProduct/s)*(imagProduct/s) - 3.0*(imagProduct/s)*(imagProduct/s) + 3.0*(imagProduct/s);
        double barResult2 = 1.0/barResult - 1.0;
        if (imagProduct<=0) barResult2 = std::numeric_limits<double>::infinity();
        if (imagProduct>=s) barResult2 = 0.0;
        fBarrier(N*i+j)=barResult2;
        
        double barDerivative=(3.0*(imagProduct*imagProduct/(s*s*s)) -6.0*(imagProduct/(s*s)) + 3.0/s)/origFieldVolumes(i,j);
        if (imagProduct<=0) barDerivative=std::numeric_limits<double>::infinity();
        if (imagProduct>=s) barDerivative=0.0;
        fBarrierDerivative(N*i+j)=barDerivative;
        
        SImagField.row(N*i+j)<<nextVec(1), -nextVec(0), -currVec(1),currVec(0);
      }
    }
    
    SparseMatrix<double> gImagField(N*FN.rows(), currField.size());
    vector<Triplet<double>> gImagTriplets;
    for (int i=0;i<IImagField.size();i++)
      gImagTriplets.push_back(Triplet<double>(IImagField(i), JImagField(i), SImagField(s)));
    
    gImagField.setFromTriplets(gImagTriplets.begin(), gImagTriplets.end());
    
    SparseMatrix<double> gFieldReduction = gClose;
    VectorXd barDerVec=-fBarrierDerivative.array()/(fBarrier.array()*fBarrier.array());
    for (int i=0;i<fBarrier.size();i++)
      if (fBarrierDerivative(i)>std::numeric_limits<double>::infinity()*0.9)
        barDerVec(i)=0.0;
    
    SparseMatrix<double> gBarrierFunc;
    igl::diag(barDerVec,gBarrierFunc);
    
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
    
    JRows.conservativeResize(J.nonZeros());
    JCols.conservativeResize(J.nonZeros());
    JVals.conservativeResize(J.nonZeros());
    int counter=0;
    for (int k=0; k<G2.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(G2,k); it; ++it){
        JRows(counter)=it.row();
        JCols(counter)=it.col();
        JVals(counter++)=it.value();
      }
    
    
  }
  bool post_optimization(const Eigen::VectorXd& x){
    std::cout<<"x:"<<x<<std::endl;
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
    
    cout<<"I: "<<I<<endl;
    cout<<"J: "<<J<<endl;
    cout<<"S: "<<S<<endl;
    //creating small dense matrix with all non-zero columns
    VectorXi uniqueJVec(uniqueJ.size());
    VectorXi JMask=VectorXi::Constant(C.cols(),-1);
    counter=0;
    for (set<int>::iterator ji=uniqueJ.begin();ji!=uniqueJ.end();ji++){
      uniqueJVec(counter)=*ji;
      JMask(*ji)=counter++;
    }
    
    cout<<"uniqueJVec: "<<uniqueJVec<<endl;
    
    MatrixXd CSmall=MatrixXd::Zero(C.rows(), JMask.maxCoeff()+1);
    for (int i=0;i<I.rows();i++)
      CSmall(I(i),JMask(J(i)))=S(i);
    
    cout<<"CSmall: "<<CSmall<<endl;
    
    
    FullPivLU<MatrixXd> lu_decomp(CSmall);
    MatrixXd USmall=lu_decomp.kernel();
    
    cout<<"USmall: "<<USmall<<endl;
    
    
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
    
    cout<<"(C*UFull).lpNorm<Infinity>(): "<<(C*UFull).norm()<<endl;
    
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
    
    cout<<"rawField2Vec: "<<rawField2Vec<<endl;
    
    
    IImagField.resize(N*F.rows(),4);
    JImagField.resize(N*F.rows(),4);
    origFieldVolumes.resize(F.rows(),N);
    for (int i=0;i<F.rows();i++){
      IImagField.row(i)=VectorXi::Constant(4,i);
      for (int j=0;j<N;j++){
        RowVector2d currVec=rawField2.block(i,2*j,1,2);
        RowVector2d nextVec=rawField2.block(i,2*((j+1)%N),1,2);
        JImagField.row(i*N+j)<<2*N*i+2*j, 2*N*i+2*j+1, 2*N*i+2*((j+1)%N), 2*N*i+2*((j+1)%N)+1;
        origFieldVolumes(i,j)=currVec(0)*nextVec(1)-currVec(1)*nextVec(0);
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
    cout<<"bigRhs: "<<bigRhs<<endl;
    
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
    
                    
  }
};



#endif /* SIInitialSolutionTraits_h */
