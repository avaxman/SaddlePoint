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
#include <Eigen/Sparse>
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
    
    //Eigen::VectorXi HRows, HCols;  //(row,col) pairs for H=J^T*J matrix
    //Eigen::VectorXd HVals;      //values for H matrix
    //Eigen::MatrixXi S2D;        //single J to J^T*J indices
    
    LinearSolver* LS;
    SolverTraits* ST;
    DampingTraits* DT;
    
    
    int maxIterations;
    double funcTolerance;
    double fooTolerance;
    
  public:
    
    LMSolver(){};
    
    void init(LinearSolver* _LS,
              SolverTraits* _ST,
              DampingTraits* _DT,
              int _maxIterations=100,
              double _funcTolerance=10e-10,
              double _fooTolerance=10e-10){
      
      LS=_LS;
      ST=_ST;
      DT=_DT;
      maxIterations=_maxIterations;
      funcTolerance=_funcTolerance;
      fooTolerance=_fooTolerance;
      
      //analysing pattern
      //Eigen::SparseMatrix<int> JPattern;
      //ST->jacobian_pattern(JPattern);
      
      //LS->analyze(JPattern);
      
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
      SparseMatrix<double> dampMatrix;
      VectorXd EVec;
      SparseMatrix<double> J, Jt, JtJ;
      
      currIter=0;
      stop=false;
      ST->objective_jacobian(prevx, EVec, J, true);
      Jt=J.transpose();
      JtJ = Jt*J;
      
      DT->init(JtJ, prevx, dampMatrix);
      JtJ=JtJ+dampMatrix;
      
      do{
        ST->pre_iteration(prevx);
        
        if (verbose)
          cout<<"Initial objective for Iteration "<<currIter<<": "<<EVec.squaredNorm()<<endl;
        
        //multiply_adjoint_vector(ST->JRows, ST->JCols, JVals, -EVec, rhs);
        rhs = -(Jt*EVec);
        
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
        if(!LS->factorize(JtJ)) {
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
        if (direction.norm() < funcTolerance){
          x=prevx;
          if (verbose)
            cout<<"Stopping since direction magnitude small."<<endl;
          return true;
        }
        
        ST->objective_jacobian(prevx,EVec, J, false);
        double prevEnergy2=EVec.squaredNorm();
        ST->objective_jacobian(prevx+direction,EVec, J, false);
        double newEnergy2=EVec.squaredNorm();
        
        if (prevEnergy2>newEnergy2){
          x=prevx+direction;{
            if (std::abs(prevEnergy2-newEnergy2)<funcTolerance){
              if (verbose)
                cout<<"Stopping sincefunction didn't change above tolerance."<<endl;
              break;
            }
          }
          
        }else
          x=prevx;
        
        cout<<"New energy: "<<EVec.squaredNorm()<<endl;
        ST->objective_jacobian(x,EVec, J, true);
        Jt=J.transpose();
        JtJ = Jt*J;
        
        DT->update(*ST, JtJ,  prevx, direction, dampMatrix);
        JtJ=JtJ+dampMatrix;
        
        //The SolverTraits can order the optimization to stop by giving "true" of to continue by giving "false"
        if (ST->post_iteration(x)){
          if (verbose)
            cout<<"ST->Post_iteration() gave a stop"<<endl;
          return true;
        }
        currIter++;
        prevx=x;
      }while (currIter<=maxIterations);
      
      return false;
    }
  };
  
  }


#endif
