// This file is part of SaddlePoint, a simple library for Eigen-based sparse nonlinear optimization
//
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SADDLEPOINT_DIAGONAL_DAMPING_H
#define SADDLEPOINT_DIAGONAL_DAMPING_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <cstdio>
#include <set>


namespace SaddlePoint{
template<class SolverTraits>
class DiagonalDamping{
private:
  
  double currLambda;
public:
  void init(const Eigen::VectorXi& HRows,
            const Eigen::VectorXi& HCols,
            const Eigen::VectorXd& HVals,
            const Eigen::VectorXd& initSolution,
            Eigen::VectorXd& dampVector){
    
    //collecting the diagonal values
    dampVector=Eigen::VectorXd::Zero(initSolution.size());
    for (int i=0;i<HRows.size();i++)
      if (HRows(i)==HCols(i))  //a diagonal value
        dampVector(HRows(i))+=currLambda*HVals(i);
  }
  
  bool update(SolverTraits& ST,
              const Eigen::VectorXi& HRows,
              const Eigen::VectorXi& HCols,
              const Eigen::VectorXd& HVals,
              const Eigen::VectorXd& currSolution,
              const Eigen::VectorXd& direction,
              Eigen::VectorXd& dampVector){
    
    Eigen::VectorXd EVec;
    ST.objective(currSolution,EVec);
    double prevEnergy2=EVec.squaredNorm();
    ST.objective(currSolution+direction,EVec);
    double newEnergy2=EVec.squaredNorm();
    std::cout<<"prevEnergy2: "<<prevEnergy2<<std::endl;
    std::cout<<"newEnergy2: "<<newEnergy2<<std::endl;
    //currLambda=0.0;
    if (prevEnergy2>newEnergy2)  //progress; making it more gradient descent
      currLambda/=10.0;
    else
      currLambda*=10.0;
    
    std::cout<<"currLambda: "<<currLambda<<std::endl;
    
    dampVector=Eigen::VectorXd::Zero(currSolution.size());
    for (int i=0;i<HRows.size();i++)
      if (HRows(i)==HCols(i))  //a diagonal value
        dampVector(HRows(i))+=currLambda*HVals(i);
    
    //dampVector=Eigen::VectorXd::Constant(currSolution.size(),currLambda);
    
    return (prevEnergy2>newEnergy2);  //this preconditioner always approves new direction
    
  }
  
  DiagonalDamping(double _currLambda=0.01):currLambda(_currLambda){}
  ~DiagonalDamping(){}
};
}

#endif /* DiagonalDamping_h */
