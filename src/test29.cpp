//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;
//using namespace arma;

SEXP Lik(arma::vec X, arma::vec de, arma::mat Z, arma::vec beta){
  int delen = de.n_elem;              
  arma::mat one1(delen,1);
  for (int i=0; i<delen; i++){
    one1(i) = 1;
  }
  arma::vec one11 = vectorise(one1);  
  arma::umat test1 = (de == one11);   
  int dimen = beta.n_elem;            
  arma::mat test11(delen,dimen);      
  for (int i=0; i<delen; i++){
    for (int j=0; j<dimen; j++){
      test11(i,j) = test1(i,0);       
    }
  }
  arma::mat newx = nonzeros(X % test1); 
  int newxlen = newx.n_elem;            
  arma::mat atrisk; atrisk.set_size(newxlen,delen); 
  for (int i=0; i<delen; i++){
    for (int j=0; j<newxlen; j++){
      if (newx(j) <= X(i)){
        atrisk (j,i) = 1;
      }
      if (newx(j) > X(i)){
        atrisk (j,i) = 0;
      }
    }
  }                                       
  arma::mat Zbeta = Z * beta;
  int Zbetacol = Zbeta.n_cols;
  for (int i=0; i<Zbetacol; i++){
    if (Zbeta(i,0) > 600){
      Zbeta(i,0) = 600;
    }
  }                                 
  arma::mat expZbeta = exp(Zbeta);
  arma::mat S0 = atrisk * expZbeta;
  arma::mat newz = Z % test11;
  arma::mat newlik = sum(newz * beta, 0) - sum(log(S0) , 0);
  double result = as_scalar(newlik);    
  List result1 = List :: create (Named("lik")=result);
  return wrap(result1);
}

SEXP LikScorV(arma::vec X, arma::vec de, arma::mat Z, arma::vec beta){
  int delen = de.n_elem;
  arma::mat one1(delen,1);
  for (int i=0; i<delen; i++){
    one1(i) = 1;
  }
  arma::vec one11 = vectorise(one1);
  arma::umat test1 = (de == one11);
  int dimen = beta.n_elem;
  arma::mat test11(delen,dimen);
  for (int i=0; i<delen; i++){
    for (int j=0; j<dimen; j++){
      test11(i,j) = test1(i,0);
    }
  }
  arma::mat newx = nonzeros(X % test1);
  int newxlen = newx.n_elem;
  arma::mat atrisk; atrisk.set_size(newxlen,delen);
  for (int i=0; i<delen; i++){
    for (int j=0; j<newxlen; j++){
      if (newx(j) <= X(i)){
        atrisk (j,i) = 1;
      }
      if (newx(j) > X(i)){
        atrisk (j,i) = 0;
      }
    }
  }
  arma::mat Zbeta = Z * beta;
  int Zbetacol = Zbeta.n_cols;
  for (int i=0; i<Zbetacol; i++){
    if (Zbeta(i,0) > 600){
      Zbeta(i,0) = 600;
    }
  }
  arma::mat expZbeta = exp(Zbeta);
  arma::mat S0 = atrisk * expZbeta;
  arma::mat newz = Z % test11;
  arma::mat newlik = sum(newz * beta, 0) - sum(log(S0) , 0);    
  double result = as_scalar(newlik);
  arma::mat expZbeta1(delen,dimen);  
  for (int i=0; i<delen; i++){
    for (int j=0; j<dimen; j++){
      expZbeta1(i,j) = expZbeta(i,0);
    }
  }                                 
  arma::mat S1 = atrisk * (expZbeta1 % Z);
  int atlen = atrisk.n_rows;        
  arma::mat S01(atlen,dimen);
  for (int i=0; i<atlen; i++){
    for (int j=0; j<dimen; j++){
      S01(i,j) = S0(i,0);    
    }
    
  }
  arma::mat score = sum(newz,0) - sum(S1/S01,0);
  arma::mat score1 = score.t();
  arma::vec score2 = vectorise(score1);
  List result1 = List :: create (Named("lik")=result, Named("score")=score2);
  return wrap(result1);
}

//' Non_monotone proximal gradient algorithm
//'
//' SRMPLE is used to do joint feature screening for ultra-high Cox's model. It uses Lasso intializer and non-monotone proximal gradient algorithm which is more efficient than previous method.
//' @param X a vector of observed time for the population
//' @param de a vector of censoring indicator for the population
//' @param Z a matrix of high-dimensional covariates
//' @param betaini a vector of initial beta after Lasso regularization
//' @param k a positive integar of sparsity constraints
//' @param tol a positive number for termination condition
//' @param maxiter a positive number for the maximum number of iterations before termination
//'
//' @return a list of remaining covariates index, estimated beta value after algorithm and number of iterations
//' @export
//'
//' @usage  SRMPLE(X,de,Z,betaini,k,tol=1e-3,maxiter=50)
//'
//' @import glmnet
//' @details Before SRMPLE, beta should go through Lasso algorithm to get a new intializer. The recommended value for k is floor(n/(3*log(n))). The value of k and tol is recommended to use our default value, but other pre-specified value is acceptable as well.
//' @examples
//' n=120; p=10000;
//' Z <- matrix(rnorm(n*p),n,p);
//' betaini <- rep(0,p);
//' s=6;
//' betaini[1:s] <- c(-1.6328,1.3988,-1.6497,1.6353,-1.4209,1.7022);
//' T <- rexp(n,rate=exp(Z%*%betaini));
//' c0=5;
//' C <- runif(n,min=0,max=c0);
//' de <- as.numeric(T<=C);
//' X <- pmin(T,C);
//' k <- floor(n/(3*log(n)));
//' result <- SRMPLE(X,de,Z,betaini,k,tol=1e-3,maxiter=50);
// [[Rcpp::export]]
SEXP SRMPLE (arma::vec X, arma::vec de, arma::mat Z, arma::vec betaini, int k=-1, double tol=1e-3, int maxiter=50){
  int p = Z.n_cols;
  int n = Z.n_rows;
  if(k == -1){
    k = floor(n/(3*log(n)));
    return SRMPLE(X, de, Z, betaini, k, tol, maxiter);
  }
  arma::vec ind;
  arma::vec oldbeta = betaini;
  arma::vec newbeta;
  double diff = 1;
  int iter = 1;
  double Lmin = 1;
  double Lmax = 1E8;
  double tau = 2;
  double c = 1E-4;
  int M = 4;
  double L0 = 1;
  double Terrul = 1;
  arma::mat Fl(M,1);      
  for (int i=0; i<M; i++){
    Fl(i) = arma::datum::inf;
  }                       
  while((iter <= maxiter)&&(Terrul>=tol)){
    List old = LikScorV(X, de, Z, oldbeta);   
    if(iter == 1){
      Fl(0,0) = old["lik"];
    }                                         
    double oldlik = old["lik"];               
    double Lk = L0;                           
    arma::mat newbeta1(p,1);
    for (int i=0; i<p; i++){
      newbeta1(i,0) = 0;
    }                                         
    newbeta = vectorise(newbeta1);
    arma::vec oldscore = old["score"];        
    arma::vec gamma = oldbeta + (1/Lk)*oldscore;
    arma::uvec indices = sort_index(-abs(gamma),"ascend");  
    ind = arma::conv_to<arma::vec>::from(indices);    
    for (int i =0; i<k; i++){
      newbeta(ind(i)) = gamma(ind(i));
    }                                               
    List newnew = Lik(X,de,Z,newbeta);              
    arma::vec Fl1 = vectorise(Fl);
    int count = 1;
    double newnewlik = newnew["lik"];              
    while((newnewlik - min(Fl1)<=0.5*c*(as_scalar((oldbeta - newbeta).t() * (oldbeta - newbeta)))&&(Lk < Lmax))){
      arma::vec talm;
      talm<<(tau*Lk)<<Lmax;
      Lk = min(talm);                            
      for (int i=0; i<p; i++){
        newbeta1(i,0) = 0;
      }
      newbeta = vectorise(newbeta1);            
      gamma = oldbeta + (1/Lk)*oldscore;
      indices = sort_index(-abs(gamma),"ascend");
      for (int i =0; i<k; i++){
        ind(i) = indices(i);
      }
      for (int i =0; i<k; i++){
        newbeta(ind(i)) = gamma(ind(i));
      }
      newnew = Lik(X,de,Z,newbeta);
      newnewlik = newnew["lik"];                
      count = count + 1;
    }
    newnew = LikScorV(X,de,Z,newbeta);          
    arma::vec newnewscore1 = newnew["score"];
    arma::mat diff1 = sum((oldbeta - newbeta)%(oldbeta - newbeta),0);
    arma::mat diff2 = sum(diff1,1);
    diff = diff2(0,0);
    arma::mat diff3 = (newbeta-oldbeta).t()*(newnewscore1 - oldscore);    
    double dif = diff3(0,0);
    arma::vec dif1;
    dif1<<(-1*dif/(diff))<<Lmin;
    arma::vec dif2;
    dif2<<max(dif1)<<Lmax;
    L0 = min(dif2);
    double newnewlik1 = newnew["lik"];    
    Fl(iter%M,0) = newnewlik1;            
    double newold = newnewlik1-oldlik;
    double Terrul1 = sqrt(arma::sum(newold*newold));
    arma::mat matr1 = sqrt(sum(square(newbeta),0));
    double num1 = matr1(0,0);
    arma::vec vec1;
    vec1<<1<<num1;
    arma::mat matr2 = sqrt(square(sum((newbeta - oldbeta),0)));
    double num2 = matr2(0,0);
    Terrul = (Terrul1 + num2)/(max(vec1));
    oldbeta = newbeta;
    iter = iter + 1;
  }
  for (int i =0; i<k; i++){
    ind(i) = ind(i) + 1;
  }                   
  List result1 = List :: create (Named("index")=ind,Named("beta")=newbeta, Named("iter")=iter);
  return wrap(result1);
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R

*/



