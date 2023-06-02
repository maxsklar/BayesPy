// Find an optimal Beta/Dirichlet distribution from count data
// Library copied from open source github.com/maxsklar/bayespy

export class CompressedRowData {
  K: number;
  V: number[];
  U: number[][];

  constructor(K: number) {
    this.K = K;
    this.V = [];
    this.U = [];

    for (let k = 0; k < K; k++) {
      this.U.push([]);
    }
  }

  appendRow(row: number[]) {
    if (row.length !== this.K) {
      console.error("row must have K=" + this.K + " counts");
      return;
    }

    for (let k = 0; k < this.K; k++) {
      for (let j = 0; j < row[k]; j++) {
        if (this.U[k].length === j) this.U[k].push(0);
        this.U[k][j] += 1;
      }
    }

    for (let j = 0; j < row.reduce((a, b) => a + b, 0); j++) {
      if (this.V.length === j) this.V.push(0);
      this.V[j] += 1;
    }
  }
}

function dirichLogProb(priorList: number[], data: CompressedRowData): number {
  let K = data.K;
  let total = 0.0;
  
  for (let k = 0; k < K; k++) {
    for (let i = 0; i < data.U[k].length; i++) {
      total += data.U[k][i]*Math.log(priorList[k] + i);
    }
  }

  let sumPrior = priorList.reduce((a, b) => a + b, 0);
  for (let i = 0; i < data.V.length; i++) {
    total -= data.V[i] * Math.log(sumPrior + i);
  }

  return total;
}

function priorGradient(priorList: number[], data: CompressedRowData): number[] {
  let K = data.K;
  let sumPrior = sum(priorList);

  let termToSubtract = 0;
  for (let i = 0; i < data.V.length; i++) {
    termToSubtract += data.V[i] / (sumPrior + i);
  }

  let retVal: number[] = new Array(K).fill(0);
  for (let j = 0; j < K; j++) {
    for (let i = 0; i < data.U[j].length; i++) {
      retVal[j] += data.U[j][i] / (priorList[j] + i);
    }
  }

  for (let k = 0; k < K; k++) {
    retVal[k] -= termToSubtract;
  }

  return retVal;
}

function sum(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0);
}

function priorHessianConst(priorList: number[], data: CompressedRowData): number {
  return sum(data.V.map((v, i) => v / Math.pow(sum(priorList) + i, 2)));
}

function priorHessianDiag(priorList: number[], data: CompressedRowData): number[] {
  let K = data.U.length;
  let retVal: number[] = new Array(K).fill(0);
  
  for (let k = 0; k < K; k++) {
    for (let i = 0; i < data.U[k].length; i++) {
      retVal[k] -= data.U[k][i] / Math.pow(priorList[k] + i, 2);
    }
  }

  return retVal;
}

// Compute the next value to try here
// http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf (eq 18)
//https://www.researchgate.net/profile/Max-Sklar/publication/370927162_Algorithms_for_Multivariate_Newton-Raphson_for_Optimization/links/64698c0170202663165fd82a/Algorithms-for-Multivariate-Newton-Raphson-for-Optimization.pdf
function getPredictedStep(hConst: number, hDiag: number[], gradient: number[]): number[] {
  let K = gradient.length;
  let numSum = sum(gradient.map((g, k) => g / hDiag[k]));
  let denSum = sum(hDiag.map(h => 1.0 / h));
  let b = numSum / ((1.0 / hConst) + denSum);
  return gradient.map((g, k) => (b - g) / hDiag[k]);
}

// Uses the diagonal hessian on the log-alpha values
// https://www.researchgate.net/profile/Max-Sklar/publication/370927162_Algorithms_for_Multivariate_Newton-Raphson_for_Optimization/links/64698c0170202663165fd82a/Algorithms-for-Multivariate-Newton-Raphson-for-Optimization.pdf
function getPredictedStepAlt(hConst: number, hDiag: number[], gradient: number[], alphas: number[]): number[] {
  let x = gradient.map((grad, i) => grad + alphas[i] * hDiag[i]);
  let Z = 1.0 / hConst + sum(alphas.map((alpha, i) => alpha / x[i]));
  let S = sum(alphas.map((alpha, i) => alpha * gradient[i] / x[i]));
  return gradient.map((grad, i) => (S / Z - grad) / x[i]);
}

//The priors and data are global, so we don't need to pass them in
function getTotalLoss(trialPriors: number[], data: CompressedRowData): number {
  return -1 * dirichLogProb(trialPriors, data);
}

function predictStepUsingHessian(gradient: number[], priors: number[], data: CompressedRowData): number[] {
  let totalHConst = priorHessianConst(priors, data);
  let totalHDiag = priorHessianDiag(priors, data);
  return getPredictedStep(totalHConst, totalHDiag, gradient);
}

function predictStepLogSpace(gradient: number[], priors: number[], data: CompressedRowData): number[] {
  let totalHConst = priorHessianConst(priors, data);
  let totalHDiag = priorHessianDiag(priors, data);
  return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors);
}

// Returns whether it's a good step, and the loss
function testTrialPriors(trialPriors: number[], data: CompressedRowData): number {
  if (trialPriors.some(alpha => alpha <= 0)) {
  return Infinity;
  }

  return getTotalLoss(trialPriors, data);
}

function sqVectorSize(v: number[]): number {
  return sum(v.map(x => x ** 2));
}

export function findDirichletPriors(data: CompressedRowData, iterations: number): number[] {
  let priors = Array.from({length: data.K}, () => 1.0 / data.K);

  // Let the learning begin!!
  // Only step in a positive direction, get the current best loss.
  let currentLoss = getTotalLoss(priors, data);

  let gradientToleranceSq = 2 ** -10;
  let learnRateTolerance = 2 ** -20;

  let count = 0;
  while(count < iterations) {
    count += 1;

    // Get the data for taking steps
    let gradient = priorGradient(priors, data);
    let gradientSize = sqVectorSize(gradient); 
    console.debug(`Iteration: ${count} Loss: ${currentLoss} ,Priors: ${priors}, Gradient Size: ${gradientSize}`);

    if (gradientSize < gradientToleranceSq) {
      console.debug("Converged with small gradient");
      return priors;
    }

    let trialStep = predictStepUsingHessian(gradient, priors, data);

    // First, try the second order method
    let trialPriors = priors.map((prior, i) => prior + trialStep[i]);

    // TODO: Check for taking such a small step that the loss change doesn't register (essentially converged)
    //  Fix by ending

    let loss = testTrialPriors(trialPriors, data);
    if (loss < currentLoss) {
      currentLoss = loss;
      priors = trialPriors;
      continue;
    }

    trialStep = predictStepLogSpace(gradient, priors, data);
    trialPriors = priors.map((prior, i) => {
      try {
        return prior * Math.exp(trialStep[i]);
      } catch (e) {
        return prior;
      }
    });

    loss = testTrialPriors(trialPriors, data);

    // Step in the direction of the gradient until there is a loss improvement
    let learnRate = 1.0;
    while (loss > currentLoss) {
      learnRate *= 0.9;
      trialPriors = priors.map((prior, i) => prior + gradient[i]*learnRate);
      loss = testTrialPriors(trialPriors, data);
    }

    if (learnRate < learnRateTolerance) {
      console.debug("Converged with small learn rate");
      return priors;
    }

    currentLoss = loss;
    priors = trialPriors;
  }

  console.debug("Reached max iterations");
  return priors;
}