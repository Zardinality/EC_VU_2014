import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;
import java.util.Comparator;
import java.lang.Math;

import org.apache.commons.math3.linear.*;

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

	public player19() {
		rnd_ = new Random();
		var_ = new double[10];
		for (int i = 0; i < 10; i++)
			var_[i] = 1;
	}

	@Override
	public void setSeed(long seed) {
		// Set seed of algortihms random process
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
		// Do sth with property values, e.g. specify relevant settings of your
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
			SSaDE();
		else
			DE();
	}

	private double[][] sampling(int population) {
		int i, j, k;
		double[][] g = new double[population][DIM];
		for (i = 0; i < population; i++) {
			for (j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
		}
		return g;
	}

	private double[][] samplingES() {
		int i, j;
		double[][] g = new double[population_][65];

		for (i = 0; i < population_; i++) {
			for (j = 0; j < 10; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
			for (j = 10; j < 20; j++) {
				g[i][j] = 0.2;
			}
			for (j = 20; j < 65; j++) {
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
		double r = 0.0;
		double sigma = 0.01;
		double[][] cov = new double[10][10];
		double[][] L = new double[10][10];
		double[][] gtemp = new double[lambda_][65];
		double[] x = new double[10];
		for (int i = 0; i < lambda_; i++) {
			r = glr_ * rnd_.nextGaussian();
			for (int j = 10; j < 20; j++) {
				gtemp[i][j] = gnext[i][j] + r + llr_ * rnd_.nextGaussian();
				gtemp[i][j] = Math.max(gtemp[i][j], sigma);
			}
			for (int j = 20; j < 65; j++) {
				gtemp[i][j] = gnext[i][j] + beta_ * rnd_.nextGaussian();
				if (gtemp[i][j] > Math.PI / 4)
					gtemp[i][j] -= Math.PI / 2;
				else if (gtemp[i][j] < -Math.PI / 4)
					gtemp[i][j] += Math.PI / 2;
			}
			for (int j = 0; j < DIM; j++) {
				for (int k = j; k < DIM; k++) {
					if (j != k) {
						cov[j][k] = Math.abs(Math.pow(gtemp[i][DIM + j], 2)
								- Math.pow(gtemp[i][DIM + k], 2))
								* Math.sin(2 * gtemp[i][19 + (19 - j) * j / 2
										+ k - j]);
						cov[k][j] = Math.abs(Math.pow(gtemp[i][DIM + j], 2)
								- Math.pow(gtemp[i][DIM + k], 2))
								* Math.sin(2 * gtemp[i][19 + (19 - j) * j / 2
										+ k - j]);
					} else {
						cov[j][k] = Math.pow(gtemp[i][DIM + j], 2);
					}
				}
			}
			L = cholesky(cov);
			for (int j = 0; j < DIM; j++) {
				x[j] = rnd_.nextGaussian();
			}
			for (int j = 0; j < DIM; j++) {
				sum = 0;
				for (int k = 0; k < j + 1; k++) {
					sum += L[j][k] * x[k];
				}
				gtemp[i][j] = gnext[i][j] + sum;
			}
		}
		return gtemp;
	}

	private double[][] cholesky(double[][] cov) {
		RealMatrix C = new Array2DRowRealMatrix(cov);
		CholeskyDecomposition cho = new CholeskyDecomposition(C);
		RealMatrix L = cho.getL();
		return L.getData();
	}
	
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


	
	private double trial(int dim, int population) {
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

	private void golden() {
		double[] g = new double[DIM];
		int pop = 400;// population_/2 - 1;
		for (int i = 0; i < DIM; i++)
			g[i] = trial(i, (pop) / DIM);
		for (int i = 0; i < DIM; i++)
			g[i] = trial(i, (pop) / DIM);
		pop = 500;
		for (int i = 0; i < DIM; i++)
			g[i] = trial(i, (pop) / DIM);
		evaluation_.evaluate(g);
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

	
	// can't understand now......
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

	private void ES() {
		double[][] g = samplingES();
		double[][] gnext = new double[lambda_][65];
		for (int i = 0; i < generation_; i++) {
			gnext = ES_recombination(g, 65, lambda_, population_, RECOMB_MEAN);
			gnext = ES_mutation(gnext);
			g = ES_selection(gnext);
		}
	}

	private void Rnd() {
		double[][] g = sampling(population_);
		for (int i = 0; i < population_; i++)
			evaluation_.evaluate(g[i]);
	}

//	private void CMA_ES() {
//		// set parameters (quite a lot...)
//		int lambda = 100;// lambda
//		int mu = lambda / 2;// mu
//		double mu2 = (double) lambda / 2;// mu'
//		double[] weight = new double[mu];// w
//		double[] weight2 = new double[mu];// w'
//		double sum = 0.0;
//		for (int i = 0; i < mu; i++) {
//			weight2[i] = Math.log(mu2 + 0.5) - Math.log(i);
//			sum += weight2[i];
//		}
//		double sum2 = 0.0;
//		for (int i = 0; i < mu; i++) {
//			weight[i] = weight2[i] / sum;
//			sum2 += Math.pow(weight[i], 2);
//		}
//		double mu_eff = 1 / sum2;
//		double c_sigma = (mu_eff + 2) / (DIM + mu_eff + 5);
//		double d_sigma = 1 + 2
//				* Math.max(0, Math.sqrt((mu_eff - 1) / (DIM + 1)) - 1)
//				+ c_sigma;
//		double c_c = (4 + mu_eff / DIM) / (DIM + 4 + 2 * mu_eff / DIM);
//		double c_1 = 2 / (Math.pow(DIM + 1.3, 2) + mu_eff);
//		double alpha_mu = 2;
//		double c_mu = Math.min(1 - c_1, alpha_mu * (mu_eff - 2 + 1 / mu_eff)
//				/ (Math.pow(DIM + 2, 2) + alpha_mu * mu_eff / 2));
//		double E_norm = Math.sqrt(DIM)
//				* (1 - 0.25 / DIM + 1 / (21 * Math.pow(DIM, 2)));
//
//		// initialization
//		double[] p_sigma = new double[DIM];
//		double p_norm = 0;
//		double[] p_c = new double[DIM];
//		double h_sigma = 0;
//		double[][] g = sampling(population_);
//		RealMatrix B, C, D, BT;
//		EigenDecomposition c, d;
//		double[] yw = new double[DIM];
//		double[] mean = new double[DIM];
//		double sigma = 3;
//		
//		
//		for (int i = 0; i < generation_; i++) {
//			g = CMA_sort(g);
//			yw = CMA_mean(g, weight, mu);
//			for (int j = 0; j < DIM; j++) {
//				mean[j] += sigma * yw[j];
//				p_sigma[j] = (1 - c_sigma) * p_sigma[j]
//						+ Math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * yw[j];
//			}
//			p_norm = 0;// TODO
//			sigma = sigma * Math.exp(c_sigma / d_sigma * (p_norm / E_norm - 1));
//			if (p_norm / Math.sqrt(1 - Math.pow(1 - c_sigma, 2 * (i + 1))) < (1.4 + 2 / (DIM + 1))
//					* E_norm)
//				h_sigma = 1;
//
//		}
//	}
//
//	private double[][] CMA_sample(int lambda, int mu, double m, double sigma,
//			double[][] cov) {
//		double[][] g = new double[lambda][DIM];
//		RealMatrix B, C, D, BT;
//		EigenDecomposition c, d;
//		C = new Array2DRowRealMatrix(cov);
//		c = new EigenDecomposition(C);
//		B = c.getV();
//		D = c.getD();
//		BT = c.getVT();
//		return g;
//	}
//
//	private double[][] CMA_sort(double[][] g) {
//		Arrays.sort(g, new Comparator<double[]>() {
//			@Override
//			public int compare(double[] a, double[] b) {
//				return Double.compare(b[0], a[0]);
//			}
//		});
//		return g;
//	}
//
//	private RealVector[] CMA_sort(RealVector[] g) {
//		Arrays.sort(g, new Comparator<RealVector>() {
//			@Override
//			public int compare(RealVector a, RealVector b) {
//				return Double.compare(a.getEntry(DIM), b.getEntry(DIM));
//			}
//		});
//		return g;
//	}
//
//	private double[] CMA_mean(double[][] g, double[] weight, int mu) {
//		double[] m = new double[DIM];
//		double[] sum = new double[DIM];
//		int l = g.length;
//		if (mu > l)
//			throw new RuntimeException("mu > l");
//
//		for (int i = 0; i < DIM; i++) {
//			sum[i] = 0;
//			for (int j = 0; j < mu; j++)
//				sum[i] += weight[j] * g[i][j];
//			m[i] = sum[i] / l;
//		}
//		return m;
//	}

	// Differential Evolution
	private void DE() {
		// set parameters
		double CR = 0.6;// also try 0.9 and 1
		double F = 0.6;// initial, can be further increased
		int population = 500;
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

	private void SaDE() {
		// set parameters
		double CR = 0.5;// also try 0.9 and 1
		double F = 0.5;// initial, can be further increased
		double F_l = 0.1;
		double F_u = 0.9;

		int population = 100;
		int generation = limit_ / population - 1;
		double tao_1 = 0.1;
		double tao_2 = 0.1;

		// initialization
		double randr = 0.0;
		// double rand1,rand2,rand3,rand4;
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
				if (rnd_.nextDouble() < tao_1)
					F = F_l + rnd_.nextDouble() * F_u;
				if (rnd_.nextDouble() < tao_2)
					CR = rnd_.nextDouble();
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
		double best = 0.0;
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
						y[k] = g[a][k] + F1 * (g[b][k] - g[c][k]);
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
			if (10 - best < 0.00001)
				break;
		}
		//System.out.println(Integer.toString(gen));
		
		for (int i = 0; i < population; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = bestX[j] + 0.1 * rnd_.nextGaussian();
			}
		}
		for (int i = gen; i < generation; i++) {
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
				do
					a = rnd_.nextInt(population);
				while (a == j);

				do
					b = rnd_.nextInt(population);
				while (b == j || b == a);

				do
					c = rnd_.nextInt(population);
				while (c == j || c == a || c == b);

				randi = rnd_.nextInt(DIM);
				for (int k = 0; k < DIM; k++) {
					randr = rnd_.nextDouble();
					if (randi == k || randr < CR1) {
						y[k] = g[a][k] + F1 * (g[b][k] - g[c][k]);
						// + F1 * (bestX[k] - g[a][k]);
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
		}
	}
	
	//Matrix operator
	
	// return C = A * B
    private double[][] multiply(double[][] A, double[][] B) {
        int mA = A.length;
        int nA = A[0].length;
        int mB = B.length;
        int nB = A[0].length;
        if (nA != mB) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] C = new double[mA][nB];
        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nB; j++)
                for (int k = 0; k < nA; k++)
                    C[i][j] += (A[i][k] * B[k][j]);
        return C;
    }

    // matrix-vector multiplication (y = A * x)
    private double[] multiply(double[][] A, double[] x) {
        int m = A.length;
        int n = A[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += (A[i][j] * x[j]);
        return y;
    }

    // vector-matrix multiplication (y = x^T A)
    private double[] multiply(double[] x, double[][] A) {
        int m = A.length;
        int n = A[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += (A[i][j] * x[i]);
        return y;
    }
	
    // return a random m-by-n matrix with values between 0 and 1
    private double[][] random(int m, int n) {
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = rnd_.nextDouble();
        return C;
    }
    
    // return a random m-by-n matrix with values between 0 and 1
    private double[][] random(int m, int n, double sigma) {
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = rnd_.nextGaussian() * sigma;
        return C;
    }

    // return n-by-n identity matrix I
    private double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++)
            I[i][i] = 1;
        return I;
    }

    // return x^T y
    private double dot(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // return C = A^T
    private double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[j][i] = A[i][j];
        return C;
    }

    // return C = A + B
    private double[][] add(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    // return C = A - B
    private double[][] subtract(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }
}
