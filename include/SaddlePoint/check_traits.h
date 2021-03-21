// This file is part of SaddlePoint, a simple library for Eigen-based sparse nonlinear optimization
//
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SADDLEPOINT_CHECK_TRAITS_H
#define SADDLEPOINT_CHECK_TRAITS_H
#include <Eigen/Core>
#include <string>
#include <vector>
#include <cstdio>

namespace SaddlePoint
  {
  //This function checks the Jacobian of a traits class that is put for optimization by approximate finite differences, and reports the difference. It is important to use after coding, but it is not for the actual optimization process, since it is really brute-force and slow.
  template<class SolverTraits>
  void check_traits(SolverTraits& Traits, const Eigen::VectorXd& CurrSolution){
    using namespace Eigen;
    using namespace std;
    cout<<"WARNING: FE gradient checking, reducing performance!!!"<<endl;
    cout<<"******************************************************"<<endl;
    //VectorXd CurrSolution(Traits.xSize);
    //CurrSolution<<VectorXd::Random(Traits.xSize, 1);
    
    cout<<"Solution Size: "<<Traits.xSize<<endl;
    
    VectorXd EVec;
    //VectorXd JVals;
    SparseMatrix<double> TraitGradient;
    Traits.objective_jacobian(CurrSolution, EVec, TraitGradient, true);
    //Traits.jacobian(CurrSolution, JVals);
    
    /*int MaxRow=Traits.JRows.maxCoeff()+1;
    vector<Triplet<double> > GradTris;
    
    for (int i=0;i<Traits.JRows.size();i++)
      GradTris.push_back(Triplet<double>(Traits.JRows(i), Traits.JCols(i), JVals(i)));
    
  
    SparseMatrix<double> TraitGradient(MaxRow, CurrSolution.size());
    TraitGradient.setFromTriplets(GradTris.begin(),GradTris.end());*/

    SparseMatrix<double> stubMatrix;
    SparseMatrix<double> FEGradient(TraitGradient.rows(), CurrSolution.size());
    vector<Triplet<double> > FEGradientTris;
    for (int i=0;i<CurrSolution.size();i++){
      VectorXd vh(CurrSolution.size()); vh.setZero(); vh(i)=10e-5;
      Traits.objective_jacobian(CurrSolution+vh, EVec, stubMatrix, false);
      VectorXd EnergyPlus=EVec;
      Traits.objective_jacobian(CurrSolution-vh, EVec, stubMatrix, false);
      VectorXd EnergyMinus=EVec;
      VectorXd CurrGradient=(EnergyPlus-EnergyMinus)/(2*10e-5);
      //cout<<CurrGradient<<endl;
      for (int j=0;j<CurrGradient.size();j++)
        if (abs(CurrGradient(j))>10e-7)
          FEGradientTris.push_back(Triplet<double>(j,i,CurrGradient(j)));
    }
    
    FEGradient.setFromTriplets(FEGradientTris.begin(), FEGradientTris.end());
    SparseMatrix<double> DiffMat=FEGradient-TraitGradient;
    double maxcoeff=-32767.0;
    int Maxi,Maxj;
    for (int k=0; k<DiffMat.outerSize(); ++k)
      for (SparseMatrix<double>::InnerIterator it(DiffMat,k); it; ++it){
        if (maxcoeff<abs(it.value())){
          maxcoeff=abs(it.value());
          Maxi=it.row();
          Maxj=it.col();
          
        }
        if (abs(it.value())>10e-7){
          cout<<"Gradient Discrepancy at: ("<<it.row()<<","<<it.col()<<") of "<<it.value()<<endl;
          cout<<"Our Value: "<<TraitGradient.coeffRef(it.row(), it.col())<<endl;
          cout<<"Computed Value: "<<FEGradient.coeffRef(it.row(), it.col())<<endl;
        }
      }
    cout<<"Maximum gradient difference: "<<maxcoeff<<endl;
    cout<<"At Location: ("<<Maxi<<","<<Maxj<<")"<<endl;
    
    
  }
  }


#endif
