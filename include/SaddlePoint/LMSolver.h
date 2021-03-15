// This file is part of SaddlePoint, a simple library for Eigen-based sparse nonlinear optimization
//
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SADDLEPOINT_LEVENBERG_MARQUADT_SOLVER_H
#define SADDLEPOINT_LEVENBERG_MARQUADT_SOLVER_H
#include <Eigen/Core>
#include <string>
#include <vector>
#include <list>
#include <cstdio>
#include <iostream>

namespace SaddlePoint
  {
  
  template<class LinearSolver, class SolverTraits, class DampingTraits>
  class LMSolver{
  public:
    Eigen::VectorXd x;      //current solution; always updated
    Eigen::VectorXd prevx;  //the solution of the previous iteration
    Eigen::VectorXd x0;     //the initial solution to the system
    Eigen::VectorXd d;             //the direction taken.
    Eigen::VectorXd currObjective;    //the current value of the energy
    Eigen::VectorXd prevObjective;    //the previous value of the energy
    
    Eigen::VectorXi HRows, HCols;  //(row,col) pairs for H=J^T*J matrix
    Eigen::VectorXd HVals;      //values for H matrix
    Eigen::MatrixXi S2D;        //single J to J^T*J indices
    
    LinearSolver* LS;
    SolverTraits* ST;
    DampingTraits* DT;
    
    
    int maxIterations;
    double xTolerance;
    double fooTolerance;
    
    /*void TestMatrixOperations(){
     
     using namespace Eigen;
     using namespace std;
     int RowSize=1000;
     int ColSize=1000;
     int TriSize=4000;
     
     VectorXi Rows;
     VectorXi Cols;
     VectorXd Vals=VectorXd::Random(TriSize);
     
     VectorXd dRows=VectorXd::Random(TriSize);
     VectorXd dCols=VectorXd::Random(TriSize);
     
     cout<<"dRows Range: "<<dRows.minCoeff()<<","<<dRows.maxCoeff()<<endl;
     
     Rows=((dRows+MatrixXd::Constant(dRows.size(),1,1.0))*500.0).cast<int>();
     Cols=((dCols+MatrixXd::Constant(dRows.size(),1,1.0))*500.0).cast<int>();
     VectorXi SortRows, stub;
     VectorXi MRows, MCols;
     VectorXd MVals;
     
     igl::sortrows(Rows, true, SortRows, stub);
     
     MatrixXi S2D;
     
     double miu=15.0;
     
     MatrixPattern(SortRows, Cols,  MRows, MCols, S2D);
     MVals.resize(MRows.size());
     MatrixValues(MRows, MCols, Vals, S2D, miu, MVals);
     
     //testing multiplyadjoint
     SparseMatrix<double> M(SortRows.maxCoeff()+1,Cols.maxCoeff()+1);
     
     //cout<<"Our I"<<I<<endl;
     //cout<<"Our J"<<J<<endl;
     //cout<<"Our S"<<S<<endl;
     
     
     vector<Triplet<double> > Tris;
     
     for (int i=0;i<SortRows.size();i++)
     Tris.push_back(Triplet<double>(SortRows(i), Cols(i), Vals(i)));
     
     M.setFromTriplets(Tris.begin(), Tris.end());
     Tris.clear();
     
     
     SparseMatrix<double> I;
     igl::speye(M.cols(),M.cols(),I);
     
     SparseMatrix<double> MtM1=M.transpose()*M+miu*I;
     SparseMatrix<double> MtM2(MtM1.rows(), MtM1.cols());
     for (int i=0;i<MRows.size();i++)
     Tris.push_back(Triplet<double>(MRows(i), MCols(i), MVals(i)));
     
     MtM2.setFromTriplets(Tris.begin(), Tris.end());
     
     bool Discrepancy=false;
     SparseMatrix<double> DiffMat=MtM1-MtM2;
     for (int k=0; k<DiffMat.outerSize(); ++k){
     for (SparseMatrix<double>::InnerIterator it(DiffMat,k); it; ++it){
     if ((abs(it.value())>10e-6)&&(it.row()<=it.col())){
     cout<<"Discrepancy at ("<<it.row()<<","<<it.col()<<"): "<<it.value()<<endl;
     cout<<"MtM Our Values:"<<MtM2.coeffRef(it.row(),it.col())<<endl;
     cout<<"MtM Evaluated:"<<MtM1.coeffRef(it.row(),it.col())<<endl;
     Discrepancy=true;
     }
     }
     }
     if (!Discrepancy) cout<<"Matrices are equal!"<<endl;
     }*/
    
    
    //Input: pattern of matrix M by (iI,iJ) representation
    //Output: pattern of matrix M^T*M by (oI, oJ) representation
    //        map between values in the input to values in the output (Single2Double). The map is aggregating values from future iS to oS
    //prerequisite: iI are sorted by rows (not necessary columns)
    void MatrixPattern(const Eigen::VectorXi& iI,
                       const Eigen::VectorXi& iJ,
                       Eigen::VectorXi& oI,
                       Eigen::VectorXi& oJ,
                       Eigen::MatrixXi& S2D)
    {
      int CurrTri=0;
      using namespace Eigen;
      //std::list<int> oIlist;
      //std::list<int> oJlist;
      //std::list<std::pair<int, int> > S2Dlist;
      int ISize=0, JSize=0, S2DSize=0;
      do{
        int CurrRow=iI(CurrTri);
        int NumCurrTris=0;
        while ((CurrTri+NumCurrTris<iI.size())&&(iI(CurrTri+NumCurrTris)==CurrRow))
          NumCurrTris++;
        
        for (int i=CurrTri;i<CurrTri+NumCurrTris;i++){
          for (int j=CurrTri;j<CurrTri+NumCurrTris;j++){
            if (iJ(j)>=iJ(i)){
              /*oIlist.push_back(iJ(i));
               oJlist.push_back(iJ(j));
               S2Dlist.push_back(std::pair<int,int>(i,j));*/
              ISize++; JSize++; S2DSize++;
            }
          }
        }
        CurrTri+=NumCurrTris;
      }while (CurrTri!=iI.size());
      
      ISize+=iJ.maxCoeff()+1;
      JSize+=iJ.maxCoeff()+1;
      
      oI.resize(ISize);
      oJ.resize(JSize);
      S2D.resize(S2DSize,2);
      
      CurrTri=0;
      int ICounter=0, JCounter=0, S2DCounter=0;
      do{
        int CurrRow=iI(CurrTri);
        int NumCurrTris=0;
        while ((CurrTri+NumCurrTris<iI.size())&&(iI(CurrTri+NumCurrTris)==CurrRow))
          NumCurrTris++;
        
        for (int i=CurrTri;i<CurrTri+NumCurrTris;i++){
          for (int j=CurrTri;j<CurrTri+NumCurrTris;j++){
            if (iJ(j)>=iJ(i)){
              oI(ICounter++)=iJ(i);
              oJ(JCounter++)=iJ(j);
              S2D.row(S2DCounter++)<<i,j;
              /*oIlist.push_back(iJ(i));
               oJlist.push_back(iJ(j));
               S2Dlist.push_back(std::pair<int,int>(i,j));*/
            }
          }
        }
        CurrTri+=NumCurrTris;
      }while (CurrTri!=iI.size());
      
      /*int oldIlistSize=oIlist.size();
       int oldJlistSize=oJlist.size();
       oIlist.resize(oldIlistSize+iJ.maxCoeff()+1);
       oJlist.resize(oldJlistSize+iJ.maxCoeff()+1);*/
      //triplets for miu
      for (int i=0;i<iJ.maxCoeff()+1;i++){
        oI(ICounter+i)=i;
        oJ(JCounter+i)=i;
      }
      
      /*oI.resize(oIlist.size());
       oJ.resize(oJlist.size());
       S2D.resize(S2Dlist.size(),2);
       
       int counter=0;
       for (std::list<int>::const_iterator iter=oIlist.begin();iter!=oIlist.end();iter++)
       oI(counter++)=*iter;
       
       counter=0;
       for (std::list<int>::const_iterator iter=oJlist.begin();iter!=oJlist.end();iter++)
       oJ(counter++)=*iter;
       
       counter=0;
       for (std::list<std::pair<int, int> >::const_iterator iter=S2Dlist.begin();iter!=S2Dlist.end();iter++)
       S2D.row(counter++)<<iter->first, iter->second;*/
    }
    
    //returns the values of M^T*M+miu*I by multiplication and aggregating from Single2double list.
    //prerequisite - oS is allocated
    
    void set_lhs_matrix(const Eigen::VectorXi& oI,
                      const Eigen::VectorXi& oJ,
                      const Eigen::VectorXd& iS,
                      const Eigen::MatrixXi& S2D,
                      const Eigen::VectorXd& lambda,
                      Eigen::VectorXd& oS)
    {
      for (int i=0;i<S2D.rows();i++)
        oS(i)=iS(S2D(i,0))*iS(S2D(i,1));
      
      //adding miu*I
      for (int i=S2D.rows();i<oI.rows();i++)
        oS(i)=lambda(i-S2D.rows());

    }
    
    //returns M^t*ivec by (I,J,S) representation
    void multiply_adjoint_vector(const Eigen::VectorXi& iI,
                               const Eigen::VectorXi& iJ,
                               const Eigen::VectorXd& iS,
                               const Eigen::VectorXd& iVec,
                               Eigen::VectorXd& oVec)
    {
      oVec.setZero();
      for (int i=0;i<iI.size();i++)
        oVec(iJ(i))+=iS(i)*iVec(iI(i));
    }
    
    
  public:
    
    LMSolver(){};
    
    void init(LinearSolver* _LS,
              SolverTraits* _ST,
              DampingTraits* _DT,
              int _maxIterations=100,
              double _xTolerance=10e-9,
              double _fooTolerance=10e-9){
      
      LS=_LS;
      ST=_ST;
      DT=_DT;
      maxIterations=_maxIterations;
      xTolerance=_xTolerance;
      fooTolerance=_fooTolerance;
      //analysing pattern
      MatrixPattern(ST->JRows, ST->JCols,HRows,HCols,S2D);
      HVals.resize(HRows.size());
      
      LS->analyze(HRows,HCols, true);
      
      d.resize(ST->xSize);
      x.resize(ST->xSize);
      x0.resize(ST->xSize);
      prevx.resize(ST->xSize);
      currObjective.resize(ST->ESize);
      currObjective.resize(ST->ESize);
      
      //TestMatrixOperations();
    }
    
    
    bool solve(const bool verbose) {
      
      using namespace Eigen;
      using namespace std;
      ST->initial_solution(x0);
      prevx<<x0;
      int currIter=0;
      bool stop=false;
      //double currError, prevError;
      VectorXd rhs(ST->xSize);
      VectorXd direction;
      if (verbose)
        cout<<"******Beginning Optimization******"<<endl;
      
      //estimating initial miu
      VectorXd modifyVector;
      VectorXd JVals, EVec;
      
     /* ST->objective(prevx, EVec);
      ST->jacobian(prevx, JVals);
      set_lhs_matrix(HRows, HCols, JVals, S2D, Eigen::VectorXd::Zero(ST->xSize), HVals);
      //cout<<"HVals: "<<HVals<<endl;
      DT->init(HRows, HCols, HVals, x0, modifyVector);
      //cout<<"modifyVector: "<<modifyVector<<endl;
      set_lhs_matrix(HRows, HCols, JVals, S2D, modifyVector, HVals);*/
      //cout<<"HRows: "<<HRows<<endl;
      //cout<<"HCols: "<<HCols<<endl;
      //cout<<"HVals: "<<HVals<<endl;
      do{
        currIter=0;
        stop=false;
        if (!ST->pre_optimization(prevx)){
          x=prevx;
          continue;
        }else {
          ST->objective(prevx, EVec);
          ST->jacobian(prevx, JVals);
          set_lhs_matrix(HRows, HCols, JVals, S2D, Eigen::VectorXd::Zero(ST->xSize), HVals);
          //cout<<"HVals: "<<HVals<<endl;
          DT->init(HRows, HCols, HVals, prevx, modifyVector);
          //cout<<"modifyVector: "<<modifyVector<<endl;
          set_lhs_matrix(HRows, HCols, JVals, S2D, modifyVector, HVals);
        }
        do{
          ST->pre_iteration(prevx);
        
          if (verbose)
            cout<<"Initial objective for Iteration "<<currIter<<": "<<EVec.squaredNorm()<<endl;
          
          multiply_adjoint_vector(ST->JRows, ST->JCols, JVals, -EVec, rhs);
          
          double firstOrderOptimality=rhs.template lpNorm<Infinity>();
          if (verbose)
            cout<<"firstOrderOptimality: "<<firstOrderOptimality<<endl;
          
          if (firstOrderOptimality<fooTolerance){
            x=prevx;
            if (verbose){
              cout<<"First-order optimality has been reached"<<endl;
              break;
            }
          }
          
          //solving to get the GN direction
          if(!LS->factorize(HVals, true)) {
            // decomposition failed
            cout<<"Solver Failed to factorize! "<<endl;
            return false;
          }
          
          MatrixXd mRhs=rhs;
          MatrixXd mDirection;
          LS->solve(mRhs,mDirection);
          //cout<<"mRhs.maxCoeff(): "<<mRhs.maxCoeff()<<endl;
          direction=mDirection.col(0);
          //cout<<"direction.tail(100): "<<direction.tail(100)<<endl;
          if (verbose)
            cout<<"direction magnitude: "<<direction.norm()<<endl;
          if (direction.norm() < xTolerance * prevx.norm()){
            x=prevx;
            if (verbose)
              cout<<"Stopping since direction magnitude small."<<endl;
            break;
          }
          
          ST->objective(prevx,EVec);
          double prevEnergy2=EVec.squaredNorm();
          ST->objective(prevx+direction,EVec);
          double newEnergy2=EVec.squaredNorm();
          
          if (prevEnergy2>newEnergy2){
              x=prevx+direction;
  
          }else
              x=prevx;
          
          
          ST->objective(x, EVec);
          cout<<"New energy: "<<EVec.squaredNorm()<<endl;
          int where;
          cout<<"EVec.maxCoeff(): "<<EVec.maxCoeff(&where)<<endl;
          cout<<"where: "<<where<<endl;
           cout<<"EVec.segment(where-20,40): "<<EVec.segment(where-20,40)<<endl;
          ST->jacobian(x, JVals);
          set_lhs_matrix(HRows, HCols, JVals, S2D, Eigen::VectorXd::Zero(ST->xSize), HVals);
          DT->update(*ST, HRows, HCols, HVals,  prevx, direction, modifyVector);
          set_lhs_matrix(HRows, HCols, JVals, S2D,  modifyVector, HVals);
        
          //else
           // x=prevx;
          
           //cout<<"modifyVector: "<<modifyVector<<endl;
              
          //The SolverTraits can order the optimization to stop by giving "true" of to continue by giving "false"
          if (ST->post_iteration(x)){
            if (verbose)
              cout<<"ST->Post_iteration() gave a stop"<<endl;
            break;
          }
          currIter++;
          prevx=x;
        }while (currIter<=maxIterations);
      }while (!ST->post_optimization(x));
      return true;
    }
  };
  
  }


#endif
