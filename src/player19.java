import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;
import java.lang.Math;

import org.ejml.data.Matrix64F;

public class player19 implements ContestSubmission {
	public static final int DIM = 10;

	public static final int RECOMB_MEAN = 0, RECOMB_DISCRETE = 1,
			RECOMB_RANDOM = 2;

	Random rnd_;
	ContestEvaluation evaluation_;
	Integer population_;
	Integer generation_;
	Integer lambda_;// number# of offspring
	int limit_;
	double glr_;// global learning rate
	double llr_;// local learning rate
	double beta_;// turning angle
	int algIndex_;
	double[] var_;// store the best
	boolean mm_, rg_, sp_;
	Matrix64F M;

	public player19() {
		rnd_ = new Random();
		var_ = new double[10];
		for (int i = 0; i < 10; i++)
			var_[i] = 1;
	}

	@Override
	public void setSeed(long seed) {
		// Set seed of algorithms random process
		rnd_.setSeed(seed);
	}

	@Override
	public void setEvaluation(ContestEvaluation evaluation) {
		// Set evaluation problem used in the run
		evaluation_ = evaluation;

		// Get evaluation properties
		Properties props = evaluation.getProperties();
		// Property keys depend on specific evaluation
		mm_ = Boolean.parseBoolean(props.getProperty("Multimodal"));
		rg_ = Boolean.parseBoolean(props.getProperty("Regular"));
		sp_ = Boolean.parseBoolean(props.getProperty("Separable"));
		limit_ = (int) Double.parseDouble(props.getProperty("Evaluations"));
		// Do something with property values, e.g. specify relevant settings of
		// your
		// algorithm
		population_ = (int) Math.round(Math.sqrt(limit_)) / 8;
		generation_ = ((int) Math.floor(limit_) - population_)
				/ (population_ * 6);
		glr_ = 1 / Math.sqrt(2 * limit_);
		llr_ = 1 / Math.sqrt(2 * Math.sqrt(limit_));
		lambda_ = population_ * 6;
		beta_ = 0.087266462599716;
		algIndex_ = 0;

	}

	@Override
	public void run() {
		evolution(mm_, rg_, sp_);
	}

	private void evolution(boolean mm, boolean rg, boolean sp) {
		// TODO
		if (!mm)
			golden();
		else if (rg)
			SaDE();
		else
			NSaDE();
	}

	
	// basic sampling, some novel method in the end
	private double[][] sampling(int population) {
		double[][] g = new double[population][DIM];
		for (int i = 0; i < population; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
		}
		return g;
	}
	
	
	private void ES() {
		int strategy = DIM * (DIM + 3) / 2;
		double[][] g = ES_sampling();
		double[][] gnext = new double[lambda_][strategy];
		for (int i = 0; i < generation_; i++) {
			gnext = ES_recombination(g, strategy, lambda_, population_,
					RECOMB_MEAN);
			gnext = ES_mutation(gnext);
			g = ES_selection(gnext);
		}
	}
	
	private double[][] ES_sampling() {
		int strategy = DIM * (DIM + 3) / 2;
		double[][] g = new double[population_][strategy];

		for (int i = 0; i < population_; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
			for (int j = DIM; j < 2 * DIM; j++) {
				g[i][j] = 0.2;
			}
			for (int j = 2 * DIM; j < strategy; j++) {
				g[i][j] = 0;
			}
		}

		return g;
	}

	private double[][] ES_recombination(double[][] g, int length, int lambda,
			int population, int method) {
		int[] index = new int[2];
		double[][] gnext = new double[lambda][length];
		for (int i = 0; i < lambda; i++) {
			index[0] = rnd_.nextInt(population);
			index[1] = rnd_.nextInt(population);
			double ratio = rnd_.nextDouble();
			for (int j = 0; j < length; j++) {
				switch (method) {
				case RECOMB_MEAN:
					gnext[i][j] = 0.5 * (g[index[0]][j] + g[index[1]][j]);
					break;
				case RECOMB_DISCRETE:
					if (rnd_.nextDouble() > 0.5)
						gnext[i][j] = g[index[0]][j];
					else
						gnext[i][j] = g[index[1]][j];
					break;
				case RECOMB_RANDOM:
					gnext[i][j] = ratio * g[index[0]][j] + (1 - ratio)
							* g[index[1]][j];
					break;
				}
			}
		}
		return gnext;
	}

	private double[][] ES_mutation(double[][] gnext) {
		double sum;
		int strategy = DIM * (DIM + 3) / 2;
		double r = 0.0;
		double sigma = 0.1;
		double[][] cov = new double[DIM][DIM];
		double[][] L = new double[DIM][DIM];
		double[][] gtemp = new double[lambda_][strategy];
		double[] x = new double[DIM];
		for (int i = 0; i < lambda_; i++) {
			r = glr_ * rnd_.nextGaussian();
			for (int j = 0; j < DIM; j++) {
				x[j] = rnd_.nextGaussian();
			}
			for (int j = DIM; j < 2 * DIM; j++) {
				gtemp[i][j] = gnext[i][j] + r + llr_ * rnd_.nextGaussian();
				gtemp[i][j] = Math.max(gtemp[i][j], sigma);
				x[j - DIM] = x[j - DIM] * gtemp[i][j];// part of new algorithms
			}
			for (int j = 0; j < DIM; j++) {
				for (int k = j + 1; k < DIM; k++) {
					int index = 2 * DIM - 1 + (2 * DIM - 1 - j) * j / 2 + k - j;
					gtemp[i][index] = gnext[i][index] + beta_
							* rnd_.nextGaussian();
					if (gtemp[i][index] > Math.PI / 2)
						gtemp[i][index] -= Math.PI;
					else if (gtemp[i][index] <= -Math.PI / 2)
						gtemp[i][index] += Math.PI;
					x = rotate(x, j, k, gtemp[i][index]);// part of new

				}
			}
			for (int j = 0; j < DIM; j++) {
				gtemp[i][j] = x[j];
			}

			/*
			 * for (int j = 0; j < DIM; j++) { sum = 0; for (int k = 0; k < j +
			 * 1; k++) { sum += L[j][k] * x[k]; } gtemp[i][j] = gnext[i][j] +
			 * sum; }
			 */
		}
		return gtemp;
	}

	private double[] rotate(double[] g, int i, int j, double angle) {
		double[] gnext = new double[DIM];
		double[][] R = new double[DIM][DIM];
		R[i][i] = Math.cos(angle);
		R[j][j] = Math.cos(angle);
		R[i][j] = -Math.sin(angle);
		R[j][i] = Math.sin(angle);
		// gnext = multiply(R, g);
		return gnext;
	}

	/*
	 * private double[][] cholesky(double[][] cov) { RealMatrix C = new
	 * Array2DRowRealMatrix(cov); CholeskyDecomposition cho = new
	 * CholeskyDecomposition(C); RealMatrix L = cho.getL(); return L.getData();
	 * }
	 */
	private double[][] ES_selection(double[][] gnext) {
		int i, j, k;
		int index = 0;
		double[][] g = new double[population_][65];
		Double[] score = new Double[lambda_];
		Double temp = 0.0;

		for (i = 0; i < lambda_; i++) {
			score[i] = (Double) evaluation_.evaluate(Arrays.copyOfRange(
					gnext[i], 0, 10));
		}
		for (i = 0; i < population_; i++) {
			for (j = i + 1; j < lambda_; j++) {
				if (score[i] < score[j]) {
					temp = score[j];
					score[j] = score[i];
					score[i] = temp;
					index = j;
				}
			}
			g[i] = (double[]) gnext[index].clone();
		}
		return g;
	}

	
	
	// golden evolution
	private void golden() {
		double[] g = new double[DIM];
		int pop = 400;// population_/2 - 1;
		for (int i = 0; i < DIM; i++)
			g[i] = gd_trial(i, (pop) / DIM);
		for (int i = 0; i < DIM; i++)
			g[i] = gd_trial(i, (pop) / DIM);
		pop = 500;
		for (int i = 0; i < DIM; i++)
			g[i] = gd_trial(i, (pop) / DIM);
		evaluation_.evaluate(g);
	}
	
	private double gd_trial(int dim, int population) {
		int i, j, k;
		double best = 0;
		double bestscore = -999;
		double result = 0.0;
		double[][] g = new double[3][2];
		double[][] gnext = new double[4][2];
		g[0][0] = -5;
		g[0][1] = gd_evaluate(dim, g[0][0]);
		g[1][0] = 20 / (1 + Math.sqrt(5)) - 5;
		g[1][1] = gd_evaluate(dim, g[1][0]);
		g[2][0] = 5;
		g[2][1] = gd_evaluate(dim, g[2][0]);
		for (i = 0; i < population - 5; i++) {
			gnext = gd_recomb(g, dim);
			g = gd_select(gnext);
		}
		gnext = gd_recomb(g, dim);
		best = (gnext[1][0] + gnext[2][0]) / 2;
		var_[dim] = best;
		// if (gnext[1][1]>gnext[2][1]) best = gnext[1][0];
		// else best = gnext[2][0];
		best = (gnext[0][0] + gnext[3][0]) / 2;
		return best;
	}

	private double[][] gd_select(double[][] gnext) {
		double[][] g = new double[3][2];
		if (gnext[1][1] < gnext[2][1]) {
			g[0] = gnext[1].clone();
			g[1] = gnext[2].clone();
			g[2] = gnext[3].clone();
		} else {
			g[0] = gnext[0].clone();
			g[1] = gnext[1].clone();
			g[2] = gnext[2].clone();
		}
		return g;
	}

	private double[][] gd_recomb(double[][] g, int dim) {
		double[][] gnext = new double[4][2];
		gnext[0] = g[0].clone();
		gnext[3] = g[2].clone();
		double gnew = g[0][0] + g[2][0] - g[1][0];
		if (gnew < g[1][0]) {
			gnext[2] = g[1].clone();
			gnext[1][0] = gnew;
			gnext[1][1] = gd_evaluate(dim, gnew);
		} else {
			gnext[1] = g[1].clone();
			gnext[2][0] = gnew;
			gnext[2][1] = gd_evaluate(dim, gnew);
		}
		return gnext;
	}

	private double gd_evaluate(int dim, double g) {
		double score = 0;
		double[] var = var_;
		// for(int i=1;i<10;i++) var[i] = -1;
		var[dim] = g;
		score = (Double) evaluation_.evaluate(var);
		return score;
	}

	
	// for mm seperable functions(not in those 3)
	private double trialRg(int dim, int population) {
		int i, j, k;
		double best = 0;
		double bestscore = -999;
		double result = 0.0;
		int pop = (int) Math.round(Math.sqrt((double) population));
		int gen = population / pop;
		double[][] g = new double[pop][DIM];
		double[][] gnext = new double[pop][DIM];
		for (i = 0; i < pop; i++) {
			for (j = 0; j < DIM; j++) {
				g[i][j] = 1;
				gnext[i][j] = 1;
			}
			g[i][dim] = rnd_.nextDouble() * 10 - 5;
		}

		double c = 0.95;// constant for mutation changing rate
		double[] ps = new double[pop];//
		double[] s = new double[pop];
		double[] sigma = new double[pop];
		Double[] score = new Double[pop];
		for (i = 0; i < pop; i++) {
			sigma[i] = 0.1;
			score[i] = (Double) evaluation_.evaluate(g[i]);
		}

		double rtemp = 0.0;
		double tempb = 0.0;

		for (i = 0; i < gen - 1; i++) {
			for (j = 0; j < pop; j++) {
				gnext[j][dim] = g[j][dim] + sigma[j] * rnd_.nextGaussian();

				tempb = (Double) evaluation_.evaluate(gnext[j]);
				if (score[j] >= tempb) {
					s[j] = 0;
				} else {
					if (bestscore < tempb) {
						bestscore = tempb;
						best = gnext[j][dim];
					}
					// System.out.println("enter");
					s[j] = 1;
					rtemp = gnext[j][dim];
					g[j][dim] = rtemp;
					score[j] = tempb;
				}
				ps[j] = ps[j] + s[j];
				if (ps[j] / i > 0.205 && sigma[j] > 0.01) {
					sigma[j] = sigma[j] * c;
				} else if (ps[j] / i < 0.195 && sigma[j] < 2) {
					sigma[j] = sigma[j] / c;
				}
			}
		}
		return best;
	}


	// benchmark score
	private void Rnd() {
		double[][] g = sampling(population_);
		for (int i = 0; i < population_; i++)
			evaluation_.evaluate(g[i]);
	}

	// private void CMA_ES() {
	// // set parameters (quite a lot...)
	// int lambda = 100;// lambda
	// int mu = lambda / 2;// mu
	// double mu2 = (double) lambda / 2;// mu'
	// double[] weight = new double[mu];// w
	// double[] weight2 = new double[mu];// w'
	// double sum = 0.0;
	// for (int i = 0; i < mu; i++) {
	// weight2[i] = Math.log(mu2 + 0.5) - Math.log(i);
	// sum += weight2[i];
	// }
	// double sum2 = 0.0;
	// for (int i = 0; i < mu; i++) {
	// weight[i] = weight2[i] / sum;
	// sum2 += Math.pow(weight[i], 2);
	// }
	// double mu_eff = 1 / sum2;
	// double c_sigma = (mu_eff + 2) / (DIM + mu_eff + 5);
	// double d_sigma = 1 + 2
	// * Math.max(0, Math.sqrt((mu_eff - 1) / (DIM + 1)) - 1)
	// + c_sigma;
	// double c_c = (4 + mu_eff / DIM) / (DIM + 4 + 2 * mu_eff / DIM);
	// double c_1 = 2 / (Math.pow(DIM + 1.3, 2) + mu_eff);
	// double alpha_mu = 2;
	// double c_mu = Math.min(1 - c_1, alpha_mu * (mu_eff - 2 + 1 / mu_eff)
	// / (Math.pow(DIM + 2, 2) + alpha_mu * mu_eff / 2));
	// double E_norm = Math.sqrt(DIM)
	// * (1 - 0.25 / DIM + 1 / (21 * Math.pow(DIM, 2)));
	//
	// // initialization
	// double[] p_sigma = new double[DIM];
	// double p_norm = 0;
	// double[] p_c = new double[DIM];
	// double h_sigma = 0;
	// //double[][] g = sampling(population_);
	// RealMatrix C = MatrixUtils.createRealIdentityMatrix(DIM);
	// RealVector mean = new ArrayRealVector(DIM);
	// double sigma = 3;
	//
	// // evolution
	//
	// for (int i = 0; i < generation_; i++) {
	// g = CMA_sort(g);
	// yw = CMA_mean(g, weight, mu);
	// for (int j = 0; j < DIM; j++) {
	// mean[j] += sigma * yw[j];
	// p_sigma[j] = (1 - c_sigma) * p_sigma[j]
	// + Math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * yw[j];
	// }
	// p_norm = 0;// TODO
	// sigma = sigma * Math.exp(c_sigma / d_sigma * (p_norm / E_norm - 1));
	// if (p_norm / Math.sqrt(1 - Math.pow(1 - c_sigma, 2 * (i + 1))) < (1.4 + 2
	// / (DIM + 1))
	// * E_norm)
	// h_sigma = 1;
	//
	// }
	//
	// RealVector[] x = new ArrayRealVector[lambda];
	// RealVector[] y = new ArrayRealVector[lambda];
	// RealVector[] z = new ArrayRealVector[lambda];
	// for (int i = 0; i < generation_; i++) {
	// // Compute z_k
	// for (int k = 0; k < lambda; k++) {
	// for (int j = 0; j < DIM; j++) {
	// z[k].append(rnd_.nextGaussian());
	// }
	// }
	//
	// // Decomposite C
	// EigenDecomposition decomp = new EigenDecomposition(C);
	// RealMatrix B = decomp.getV();
	// RealMatrix D = decomp.getD();
	// for (int j = 0; j < DIM; j++) {
	// D.setEntry(j, j, Math.sqrt(D.getEntry(j, j)));
	// }
	//
	// // Compute y_k and x_k
	// for (int k = 0; k < lambda; k++) {
	// y[k] = B.multiply(D).operate(x[k]);
	// x[k] = y[k].mapMultiply(sigma).add(mean);
	// }
	//
	// }
	//
	// }
	//
	// private double[][] CMA_sample(int lambda, int mu, double m, double sigma,
	// double[][] cov) {
	// double[][] g = new double[lambda][DIM];
	// RealMatrix B, C, D, BT;
	// EigenDecomposition c, d;
	// C = new Array2DRowRealMatrix(cov);
	// c = new EigenDecomposition(C);
	// B = c.getV();
	// D = c.getD();
	// BT = c.getVT();
	// return g;
	// }
	//
	// private double[][] CMA_sort(double[][] g) {
	// Arrays.sort(g, new Comparator<double[]>() {
	// @Override
	// public int compare(double[] a, double[] b) {
	// return Double.compare(b[0], a[0]);
	// }
	// });
	// return g;
	// }
	//
	// private RealVector[] CMA_sort(RealVector[] g) {
	// Arrays.sort(g, new Comparator<RealVector>() {
	// @Override
	// public int compare(RealVector a, RealVector b) {
	// return Double.compare(a.getEntry(DIM), b.getEntry(DIM));
	// }
	// });
	// return g;
	// }
	//
	// private double[] CMA_mean(double[][] g, double[] weight, int mu) {
	// double[] m = new double[DIM];
	// double[] sum = new double[DIM];
	// int l = g.length;
	// if (mu > l)
	// throw new RuntimeException("mu > l");
	//
	// for (int i = 0; i < DIM; i++) {
	// sum[i] = 0;
	// for (int j = 0; j < mu; j++)
	// sum[i] += weight[j] * g[i][j];
	// m[i] = sum[i] / l;
	// }
	// return m;
	// }

	
	
	// Differential Evolution
	private void DE() {
		// set parameters
		double CR = 0.5;// also try 0.9 and 1
		double F = 0.5;// initial, can be further increased
		int population = 100;
		int generation = limit_ / population - 1;

		// initialization
		double randr = 0.0;
		int randi = 0;
		int a, b, c;
		double[][] g = sampling(population);
		double[] score = new double[population];
		for (int i = 0; i < population; i++)
			score[i] = (Double) evaluation_.evaluate(g[i]);
		double[] y = new double[DIM];
		double score_neo = 0;
		for (int i = 0; i < generation; i++) {
			for (int j = 0; j < population; j++) {
				score_neo = 0;
				do {
					a = rnd_.nextInt(population);
				} while (a == j);

				do {
					b = rnd_.nextInt(population);
				} while (b == j || b == a);

				do {
					c = rnd_.nextInt(population);
				} while (c == j || c == a || c == b);

				randi = rnd_.nextInt(DIM);
				for (int k = 0; k < DIM; k++) {
					randr = rnd_.nextDouble();
					if (randi == k || randr < CR) {
						y[k] = g[a][k] + F * (g[b][k] - g[c][k]);
						y[k] = Math.max(y[k], -5);
						y[k] = Math.min(y[k], 5);
					} else {
						y[k] = g[j][k];
					}
				}
				score_neo = (Double) evaluation_.evaluate(y);
				if (score_neo > score[j]) {
					g[j] = y.clone();
					score[j] = score_neo;
				}
			}
		}
	}

	// Self-adaptive DE
	private void SaDE() {
		// set parameters
		double[] CR;
		double CRm = 0.5;
		double CRd = 0.1;
		double F = 0.5;
		double Fm = 0.5;// initial, can be further increased
		double Fd = 0.2;// deviation

		int population = 100;
		int generation = limit_ / population - 1;
		int lp = 50;// learning period

		// initialization
		int CRsuc = 1;
		double CRsum = 0.5;
		double randr = 0.0;
		double[] bestX = new double[DIM];
		double best = -100.0;
		int gen = 0;
		int ns_1 = 1, ns_2 = 1, nf_1 = 1, nf_2 = 1;// strategy selection counter
		double p_1 = 0.5, p_2 = 0.5;// strategy selection probability
		int randi = 0;
		int r1,r2,r3,r4,r5;// random index
		int st = 1;// current strategy

		double[][] g = neo_sampling(population);
		double[] score = new double[population];
		CR = new double[population];
		for (int i = 0; i < population; i++) {
			score[i] = (Double) evaluation_.evaluate(g[i]);
			if (score[i] > best) {
				best = score[i];
				bestX = g[i].clone();
			}
		}
		double[] y = new double[DIM];
		double score_neo = 0;

		// start evolution
		for (int i = 0; i < generation; i++) {
			// reset several parameter
			if (i % 5 == 0) {
				if (i % 25 == 0) {
					CRm = CRsum / CRsuc;
					CRsum = 0.0;
					CRsuc = 0;
				}
				for (int j = 0; j < population; j++) {
					CR[j] = CRm + rnd_.nextGaussian() * CRd;
					CR[j] = Math.max(0.1, CR[j]);
				}
			}
			if (i % lp == 0) {
				p_1 = 1.2 * ns_1 * (ns_2 + nf_2)
						/ (ns_1 * (ns_2 + nf_2) + ns_2 * (ns_1 + nf_1));
				p_2 = 1 - p_1;
				ns_1 = 0;
				ns_2 = 0;
				nf_1 = 0;
				nf_2 = 0;
			}

			for (int j = 0; j < population; j++) {
				// set F, location undecided
				F = Fm + rnd_.nextGaussian() * Fd;
				F = Math.max(0.01, F);
				score_neo = 0;
				do {
					r1 = rnd_.nextInt(population);
				} while (r1 == j);

				do {
					r2 = rnd_.nextInt(population);
				} while (r2 == j || r2 == r1);

				do {
					r3 = rnd_.nextInt(population);
				} while (r3 == j || r3 == r1 || r3 == r2);
				/*
				do {
					r4 = rnd_.nextInt(population);
				} while (r4 == j || r4 == r1 || r4 == r2 || r4 == r3);
				
				do {
					r5 = rnd_.nextInt(population);
				} while (r5 == j || r5 == r1 || r5 == r2 || r5 == r3 || r5 == r4);
				*/
				if (rnd_.nextDouble() <= p_1) {
					st = 1;
				} else {
					st = 2;
				}
				randi = rnd_.nextInt(DIM);
				for (int k = 0; k < DIM; k++) {
					randr = rnd_.nextDouble();
					if (randi == k || randr < CR[j]) {
						if (st == 1) {
							y[k] = g[r1][k] + F * (g[r2][k] - g[r3][k]);
						} else {
							y[k] = g[j][k] + F * (bestX[k] - g[j][k]) + F
									* (g[r1][k] - g[r2][k]);
						}
						y[k] = Math.max(y[k], -5);
						y[k] = Math.min(y[k], 5);
					} else {
						y[k] = g[j][k];
					}
				}

				score_neo = (Double) evaluation_.evaluate(y);
				if (score_neo > score[j]) {
					g[j] = y.clone();
					score[j] = score_neo;
					// update counter
					if (st == 1)
						ns_1++;
					else
						ns_2++;

					CRsuc++;
					CRsum += CR[j];

					if (score[j] > best) {
						best = score[j];
						bestX = g[j].clone();
					}
				} else {
					if (st == 1)
						nf_1++;
					else
						nf_2++;
				}
			}
			gen++;
		}
	}

	// TODO
	// neo Self-adaptive DE
	private void NSaDE() {
		// set parameters
		double[][] CR;
		double[] CRm = { 0.5, 0.5, 0.5, 0.5 };
		double CRd = 0.1;
		int num_st = 4;// #number of strategies
		double Fm = 0.5;// initial, can be further increased
		double Fd = 0.2;// deviation
		double sigma = 0.01;//

		int population = 100;
		int generation = limit_ / population - 1;
		int lp = 50;// learning period

		// initialization
		int[] CRsuc = { 1, 1, 1, 1 };
		double[] CRsum = { 0.5, 0.5, 0.5, 0.5 };
		double randr = 0.0;
		double[] bestX = new double[DIM];
		double best = -100.0;
		int gen = 0;
		double[] F = new double[population];
		double[] K = new double[population];
		int[] ns = { 1, 1, 1, 1 };// strategy selection counter
		int[] nf = { 1, 1, 1, 1 };// strategy selection counter
		double[] p = new double[num_st];// strategy
										// selection
										// probability
		double[] s = new double[num_st];
		double s_sum;
		int randi = 0;
		int r1, r2, r3, r4, r5;// random index
		int st = 1;// current strategy

		double[][] g = neo_sampling(population);
		double[] score = new double[population];
		CR = new double[population][num_st];
		for (int i = 0; i < population; i++) {
			score[i] = (Double) evaluation_.evaluate(g[i]);
			if (score[i] > best) {
				best = score[i];
				bestX = g[i].clone();
			}
		}
		double[] y = new double[DIM];
		double score_neo = 0;

		// start evolution
		for (int i = 0; i < generation; i++) {
			// reset several parameter
			if (i % (lp / 5) == 0) {
				if (i % lp == 0) {
					for (int k = 0; k < num_st; k++) {
						CRm[k] = CRsum[k] / CRsuc[k];
						CRsum[k] = 0.0;
						CRsuc[k] = 0;
					}

					// update strategies probability
					s_sum = 0;
					for (int k = 0; k < num_st; k++) {
						s[k] = (double) ns[k] / (ns[k] + nf[k]) + sigma;
						s_sum += s[k];
					}

					for (int k = 0; k < num_st; k++) {
						p[k] = s[k] / s_sum;
						ns[k] = 0;
						nf[k] = 0;
					}
				}
				for (int j = 0; j < population; j++) {
					for (int k = 0; k < num_st; k++) {
						CR[j][k] = CRm[k] + rnd_.nextGaussian() * CRd;
						CR[j][k] = Math.max(0.1, CR[j][k]);
					}
					F[j] = Fm + rnd_.nextGaussian() * Fd;
					F[j] = Math.max(0.01, F[j]);
					K[j] = rnd_.nextDouble();
				}
			}

			for (int j = 0; j < population; j++) {
				// set F, location decided

				score_neo = 0;
				do {
					r1 = rnd_.nextInt(population);
				} while (r1 == j);

				do {
					r2 = rnd_.nextInt(population);
				} while (r2 == j || r2 == r1);

				do {
					r3 = rnd_.nextInt(population);
				} while (r3 == j || r3 == r1 || r3 == r2);

				do {
					r4 = rnd_.nextInt(population);
				} while (r4 == j || r4 == r1 || r4 == r2 || r4 == r3);

				do {
					r5 = rnd_.nextInt(population);
				} while (r5 == j || r5 == r1 || r5 == r2 || r5 == r3
						|| r5 == r4);

				randr = rnd_.nextDouble();
				if (randr <= p[0]) {
					st = 1;
				} else if (randr <= p[0] + p[1]) {
					st = 2;
				} else if (randr <= p[0] + p[1] + p[2]) {
					st = 3;
				} else {
					st = 4;
				}

				randi = rnd_.nextInt(DIM);
				for (int k = 0; k < DIM; k++) {
					if (st != 4) {
						randr = rnd_.nextDouble();
						if (randi == k || randr < CR[j][st - 1]) {
							if (st == 1) {
								y[k] = g[r1][k] + F[j] * (g[r2][k] - g[r3][k]);
							} else if (st == 2) {
								y[k] = g[j][k] + F[j] * (bestX[k] - g[j][k])
										+ F[j] * (g[r1][k] - g[r2][k]) + F[j]
										* (g[r3][k] - g[r4][k]);
							} else {
								y[k] = g[r1][k] + F[j] * (g[r2][k] - g[r3][k])
										+ F[j] * (g[r4][k] - g[r5][k]);
							}
							y[k] = Math.max(y[k], -5);
							y[k] = Math.min(y[k], 5);
						} else {
							y[k] = g[j][k];
						}
					} else {
						y[k] = g[j][k] + K[j] * (g[r1][k] - g[j][k]) + F[j]
								* (g[r2][k] - g[r3][k]);
					}
				}

				score_neo = (Double) evaluation_.evaluate(y);
				if (score_neo > score[j]) {
					g[j] = y.clone();
					score[j] = score_neo;

					// update counter
					ns[st - 1]++;
					CRsuc[st - 1]++;
					CRsum[st - 1] += CR[j][st - 1];

					if (score[j] > best) {
						best = score[j];
						bestX = g[j].clone();
					}
				} else {
					nf[st - 1]++;
				}
			}
			gen++;
		}
	}

	
	// 2 stages self-adapted DE
	private void SSaDE() {
		// set parameters
		double F_l = 0.1;
		double F_u = 0.9;

		int population = 100;
		int generation = limit_ / population - 1;
		double tao_1 = 0.1;
		double tao_2 = 0.1;

		// initialization
		double[] CR = new double[population];
		double[] F = new double[population];
		double CR1, F1;
		double randr = 0.0;
		double[] bestX = new double[DIM];
		double best = -100.0;
		int gen = 0;

		int randi = 0;
		int a, b, c;
		double[][] g = sampling(population);
		double[] score = new double[population];
		for (int i = 0; i < population; i++) {
			score[i] = (Double) evaluation_.evaluate(g[i]);
			CR[i] = 0.6;
			F[i] = 0.6;
			if (score[i] > best) {
				best = score[i];
				bestX = g[i].clone();
			}
		}
		double[] y = new double[DIM];
		double score_neo = 0;
		for (int i = 0; i < generation; i++) {
			for (int j = 0; j < population; j++) {
				if (rnd_.nextDouble() < tao_1)
					F1 = F_l + rnd_.nextDouble() * F_u;
				else
					F1 = F[j];

				if (rnd_.nextDouble() < tao_2)
					CR1 = rnd_.nextDouble();
				else
					CR1 = CR[j];

				score_neo = 0;
				do {
					a = rnd_.nextInt(population);
				} while (a == j);

				do {
					b = rnd_.nextInt(population);
				} while (b == j || b == a);

				do {
					c = rnd_.nextInt(population);
				} while (c == j || c == a || c == b);

				randi = rnd_.nextInt(DIM);
				for (int k = 0; k < DIM; k++) {
					randr = rnd_.nextDouble();
					if (randi == k || randr < CR1) {
						if (10 - best > 1 && gen <= 600) {
							y[k] = g[a][k] + F1 * (g[b][k] - g[c][k]);
						} else {
							y[k] = g[j][k] + F1 * (g[b][k] - g[c][k]) + F1
									* (bestX[k] - g[j][k]);
						}
						y[k] = Math.max(y[k], -5);
						y[k] = Math.min(y[k], 5);
					} else {
						y[k] = g[j][k];
					}
				}
				score_neo = (Double) evaluation_.evaluate(y);
				if (score_neo > score[j]) {
					g[j] = y.clone();
					score[j] = score_neo;
					F[j] = F1;
					CR[j] = CR1;
					if (score[j] > best) {
						best = score[j];
						bestX = g[j].clone();
					}
				}
			}
			gen++;
			// if(gen == 600) System.out.println(Double.toString(best));
			// if (10 - best < 1 || gen == 600)
			// break;
		}
	}

	// partical swarm optimization
	private void PSO() {
		// set parameters
		double weight = 0.4;// weight
		double vmax = 2;// vmax
		double c_1 = 1.0;
		double c_2 = 1.0;
		int population = 60;
		int generation = limit_ / population - 1;

		// initialization

		double[][] x = sampling(population);
		double[][] v = new double[population][DIM];
		double[][] xbest = new double[population][DIM];
		double[] score = new double[population];
		double[] best = new double[population];
		score[0] = (Double) evaluation_.evaluate(x[0]);
		best[0] = score[0];
		xbest[0] = x[0].clone();
		double gbest = score[0];
		double[] gxbest = x[0].clone();
		for (int i = 1; i < population; i++) {
			score[i] = (Double) evaluation_.evaluate(x[i]);
			best[i] = score[i];
			xbest[i] = x[i].clone();
			if (score[i] > gbest) {
				gbest = score[i];
				gxbest = x[i].clone();
			}
			for (int j = 0; j < DIM; j++) {
				v[i][j] = rnd_.nextGaussian() * 2;
			}
		}
		double rnd1, rnd2;
		double neoscore = 0;
		for (int g = 0; g < generation; g++) {
			for (int i = 0; i < population; i++) {
				rnd1 = rnd_.nextDouble();
				rnd2 = rnd_.nextDouble();
				for (int j = 0; j < DIM; j++) {
					v[i][j] += weight * v[i][j];
					v[i][j] += c_1 * rnd1 * (xbest[i][j] - x[i][j]);
					v[i][j] += c_2 * rnd2 * (gxbest[j] - x[i][j]);
					v[i][j] += 0.01 * rnd_.nextGaussian();
					v[i][j] = Math.min(v[i][j], 5 - x[i][j]);
					v[i][j] = Math.max(v[i][j], -(x[i][j] + 5));
					v[i][j] = Math.min(v[i][j], vmax);
					v[i][j] = Math.max(v[i][j], -vmax);
					x[i][j] += v[i][j];
				}
				neoscore = (Double) evaluation_.evaluate(x[i]);
				if (neoscore > best[i]) {
					best[i] = neoscore;
					xbest[i] = x[i].clone();
					if (neoscore > gbest) {
						gbest = neoscore;
						gxbest = x[i].clone();
					}
				}
			}
		}

	}

	// self adaptive partical swarm optimization
	private void SaPSO() {
		// set parameters
		double vmax = 2;// vmax
		double c_1 = 1.25;
		double c_2 = 1.25;
		int population = 50;
		int generation = limit_ / population - 1;
		double[] weight = new double[population];// weight
		double tao = 0.1;

		// initialization

		double[][] x = sampling(population);
		double[][] v = new double[population][DIM];
		double[][] xbest = new double[population][DIM];
		double[] score = new double[population];
		double[] best = new double[population];
		score[0] = (Double) evaluation_.evaluate(x[0]);
		best[0] = score[0];
		xbest[0] = x[0].clone();
		double gbest = score[0];
		double[] gxbest = x[0].clone();
		for (int i = 1; i < population; i++) {
			score[i] = (Double) evaluation_.evaluate(x[i]);
			best[i] = score[i];
			xbest[i] = x[i].clone();
			if (score[i] > gbest) {
				gbest = score[i];
				gxbest = x[i].clone();
			}
			for (int j = 0; j < DIM; j++) {
				v[i][j] = rnd_.nextGaussian() * 2;
			}
			weight[i] = 0.4;
		}
		double rnd1, rnd2;
		double neoscore = 0.0;
		double neo_weight = 0.0;
		for (int g = 0; g < generation; g++) {
			for (int i = 0; i < population; i++) {

				rnd1 = rnd_.nextDouble();
				rnd2 = rnd_.nextDouble();

				if (rnd_.nextDouble() < tao)
					neo_weight = 0.3 + rnd_.nextDouble() * 0.7;
				else
					neo_weight = weight[i];

				for (int j = 0; j < DIM; j++) {
					v[i][j] += neo_weight * v[i][j];
					v[i][j] += c_1 * rnd1 * (xbest[i][j] - x[i][j]);
					v[i][j] += c_2 * rnd2 * (gxbest[j] - x[i][j]);
					v[i][j] += 0.01 * rnd_.nextGaussian();

					v[i][j] = Math.min(v[i][j], 5 - x[i][j]);
					v[i][j] = Math.max(v[i][j], -(x[i][j] + 5));
					v[i][j] = Math.min(v[i][j], vmax);
					v[i][j] = Math.max(v[i][j], -vmax);

					x[i][j] += v[i][j];// update location
				}
				neoscore = (Double) evaluation_.evaluate(x[i]);
				if (neoscore > best[i]) {
					best[i] = neoscore;
					xbest[i] = x[i].clone();
					weight[i] = neo_weight;
					if (neoscore > gbest) {
						gbest = neoscore;
						gxbest = x[i].clone();
					}
				}
			}
		}

	}

	// new sampling method
	private double[][] opp_sampling(int population) {
		double[][] g = new double[population][DIM];
		for (int i = 0; i < population / 2; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
		}
		for (int i = population / 2; i < population; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = -g[i - population / 2][j];
			}
		}
		return g;
	}
	
	private double[][] neo_sampling(int population) {
		double[][] g = new double[population][DIM];
		for (int i = 0; i < population / 2; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
		}
		for (int i = population / 2; i < population; i++) {
			for (int j = 0; j < DIM; j++) {
				if (g[i - population / 2][j] > 0) {
					g[i][j] = (g[i - population / 2][j] - 5) / 2;
				} else if (g[i - population / 2][j] < 0) {
					g[i][j] = (g[i - population / 2][j] + 5) / 2;
				} else {
					g[i][j] = rnd_.nextDouble() * 10 - 5;
				}
				
			}
		}
		return g;
	}
	
	public static void main(String[] args) {
	}
}
