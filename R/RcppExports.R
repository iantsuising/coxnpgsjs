# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Non_monotone proximal gradient algorithm
#'
#' SRMPLE is used to do joint feature screening for ultra-high Cox's model. It uses Lasso intializer and non-monotone proximal gradient algorithm which is more efficient than previous method.
#' @param X a vector of observed time for the population
#' @param de a vector of censoring indicator for the population
#' @param Z a matrix of high-dimensional covariates
#' @param betaini a vector of initial beta after Lasso regularization
#' @param k a positive integar of sparsity constraints
#' @param tol a positive number for termination condition
#' @param maxiter a positive number for the maximum number of iterations before termination
#'
#' @return a list of remaining covariates index, estimated beta value after algorithm and number of iterations
#' @export
#'
#' @usage  SRMPLE(X,de,Z,betaini,k,tol=1e-3,maxiter=50)
#'
#' @import glmnet
#' @details Before SRMPLE, beta should go through Lasso algorithm to get a new intializer. The recommended value for k is floor(n/(3*log(n))). The value of k and tol is recommended to use our default value, but other pre-specified value is acceptable as well.
#' @examples
#' n=120; p=10000;
#' Z <- matrix(rnorm(n*p),n,p);
#' betaini <- rep(0,p);
#' s=6;
#' betaini[1:s] <- c(-1.6328,1.3988,-1.6497,1.6353,-1.4209,1.7022);
#' T <- rexp(n,rate=exp(Z%*%betaini));
#' c0=5;
#' C <- runif(n,min=0,max=c0);
#' de <- as.numeric(T<=C);
#' X <- pmin(T,C);
#' k <- floor(n/(3*log(n)));
#' result <- SRMPLE(X,de,Z,betaini,k,tol=1e-3,maxiter=50);
SRMPLE <- function(X, de, Z, betaini, k = -1L, tol = 1e-3, maxiter = 50L) {
    .Call(`_coxnpgsjs_SRMPLE`, X, de, Z, betaini, k, tol, maxiter)
}

