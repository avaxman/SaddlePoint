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
  void init(const Eigen::SparseMatrix<double>& J,
            const Eigen::VectorXd& initSolution,
            Eigen::SparseMatrix<double>& dampJ){
    
    
    //collecting the diagonal values
    Eigen::VectorXd dampVector=Eigen::VectorXd::Zero(initSolution.size());
    std::vector<Eigen::Triplet<double>> dampJTris;
    for (int k=0; k<J.outerSize(); ++k){
      for (Eigen::SparseMatrix<double>::InnerIterator it(J,k); it; ++it){
        dampVector(it.col())+=currLambda*it.value()*it.value();
        dampJTris.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
      }
    }
    for (int i=0;i<dampVector.size();i++)
      dampJTris.push_back(Eigen::Triplet<double>(J.rows()+i,i,sqrt(dampVector(i))));
    
    dampJ.conservativeResize(J.rows()+dampVector.size(),J.cols());
    dampJ.setFromTriplets(dampJTris.begin(), dampJTris.end());
  }
  
  bool update(SolverTraits& ST,
              const Eigen::SparseMatrix<double>& J,
              const Eigen::VectorXd& currSolution,
              const Eigen::VectorXd& direction,
              Eigen::SparseMatrix<double>& dampJ){
    
    Eigen::VectorXd EVec;
    Eigen::SparseMatrix<double> stubJ;
    ST.objective_jacobian(currSolution,EVec, stubJ, false);
    double prevEnergy2=EVec.squaredNorm();
    ST.objective_jacobian(currSolution+direction,EVec, stubJ, false);
    double newEnergy2=EVec.squaredNorm();
    //std::cout<<"prevEnergy2: "<<prevEnergy2<<std::endl;
    //std::cout<<"newEnergy2: "<<newEnergy2<<std::endl;
    //currLambda=0.0;
    if ((prevEnergy2>newEnergy2)&&(newEnergy2!=std::numeric_limits<double>::infinity()))  //progress; making it more gradient descent
      currLambda/=10.0;
    else
      currLambda*=10.0;
    //collecting the diagonal values
    Eigen::VectorXd dampVector=Eigen::VectorXd::Zero(currSolution.size());
    std::vector<Eigen::Triplet<double>> dampJTris;
    for (int k=0; k<J.outerSize(); ++k){
      for (Eigen::SparseMatrix<double>::InnerIterator it(J,k); it; ++it){
        dampVector(it.col())+=currLambda*it.value()*it.value();
        dampJTris.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
      }
    }
    for (int i=0;i<dampVector.size();i++)
      dampJTris.push_back(Eigen::Triplet<double>(J.rows()+i,i,sqrt(dampVector(i))));
    
    dampJ.conservativeResize(J.rows()+dampVector.size(),J.cols());
    dampJ.setFromTriplets(dampJTris.begin(), dampJTris.end());
    
    //dampVector=Eigen::VectorXd::Constant(currSolution.size(),currLambda);
    
    return (prevEnergy2>newEnergy2);  //this preconditioner always approves new direction
    
  }
  
  DiagonalDamping(double _currLambda=0.01):currLambda(_currLambda){}
  ~DiagonalDamping(){}
};
}

#endif /* DiagonalDamping_h */
