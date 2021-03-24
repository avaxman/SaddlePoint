// This file is part of SaddlePoint, a simple library for Eigen-based sparse nonlinear optimization
//
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SADDLEPOINT_DIAGONAL_DAMPING_H
#define SADDLEPOINT_DIAGONAL_DAMPING_H
#include <igl/diag.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <cstdio>
#include <set>


namespace SaddlePoint{
template<class SolverTraits>
class DiagonalDamping{
public:
  double currLambda;
  void init(const Eigen::SparseMatrix<double>& H,
            const Eigen::VectorXd& initSolution,
            Eigen::SparseMatrix<double>& D){
    
    
    //collecting the diagonal values
    Eigen::VectorXd dampVector=Eigen::VectorXd::Zero(initSolution.size());
    for (int k=0; k<H.outerSize(); ++k){
      for (Eigen::SparseMatrix<double>::InnerIterator it(H,k); it; ++it){
        if (it.row()==it.col()) //a diagonal value
          dampVector(it.row())+=currLambda*it.value();
      }
    }
    igl::diag(dampVector, D);
  }
  
  bool update(SolverTraits& ST,
              const Eigen::SparseMatrix<double>& H,
              const Eigen::VectorXd& currSolution,
              const Eigen::VectorXd& direction,
              Eigen::SparseMatrix<double>& D){
    
    Eigen::VectorXd EVec;
    Eigen::SparseMatrix<double> J;
    ST.objective_jacobian(currSolution,EVec, J, false);
    double prevEnergy2=EVec.squaredNorm();
    ST.objective_jacobian(currSolution+direction,EVec, J, false);
    double newEnergy2=EVec.squaredNorm();
    //std::cout<<"prevEnergy2: "<<prevEnergy2<<std::endl;
    //std::cout<<"newEnergy2: "<<newEnergy2<<std::endl;
    //currLambda=0.0;
    if ((prevEnergy2>newEnergy2)&&(newEnergy2!=std::numeric_limits<double>::infinity()))  //progress; making it more gradient descent
      currLambda/=10.0;
    else
      currLambda*=10.0;
    
    //std::cout<<"currLambda: "<<currLambda<<std::endl;
    
    Eigen::VectorXd dampVector=Eigen::VectorXd::Zero(currSolution.size());
    for (int k=0; k<H.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(H,k); it; ++it){
        if (it.row()==it.col()) //a diagonal value
          dampVector(it.row())+=currLambda*it.value();
      }
    
    igl::diag(dampVector, D);
    
    //dampVector=Eigen::VectorXd::Constant(currSolution.size(),currLambda);
    
    return (prevEnergy2>newEnergy2);  //this preconditioner always approves new direction
    
  }
  
  DiagonalDamping(double _currLambda=0.01):currLambda(_currLambda){}
  ~DiagonalDamping(){}
};
}

#endif /* DiagonalDamping_h */
