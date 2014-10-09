import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;
import java.lang.Math;
import java.util.Comparator;
import java.lang.reflect.Array;

import org.ejml.simple.*;

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
	double best_; // store the best in CMA_ES

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

		// Do something with property values

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
		// SimpleMatrix test = new SimpleMatrix(DIM, 1);
		if (!mm)
			//golden();
			SAA();
		else if (rg)
			SaDE();
		else {
			do {
				MVMO();
			} while (limit_ > 10000);
		}
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

		// SimpleMatrix RR = new SimpleMatrix(R);
		// SimpleMatrix gg = new SimpleMatrix(DIM, 1, true, g);
		// SimpleMatrix ggnext = RR.mult(gg);
		// gnext = ggnext.getMatrix().getData();

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
		int pop = limit_ / 5 - 1;// population_/2 - 1;
		for (int i = 0; i < DIM; i++)
			g[i] = gd_trial(i, (pop) / DIM);
		pop = limit_ - limit_ / 5 - 1;
		for (int i = 0; i < DIM; i++)
			g[i] = gd_trial(i, (pop) / DIM);
		// pop = 500;
		// for (int i = 0; i < DIM; i++)
		// g[i] = gd_trial(i, (pop) / DIM);
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

	private void CMA_ES_RS() {
		int lambda = 55;
		while (limit_ * 2 > (int) lambda
				* (100 + 50 * Math.pow((DIM + 3), 2) / Math.sqrt(lambda))) {
			CMA_ES(lambda);
			// lambda = lambda * 2;
		}
	}

	private void CMA_ES(int lambda) {
		// Set parameters
		// - Selection and Recombination
		int generation = (int) (100 + 50 * Math.pow((DIM + 3), 2)
				/ Math.sqrt(lambda) / 2);
		int mu = lambda / 2; //
		double mu_p = (double) lambda / 2; // mu'
		double[] w = new double[mu]; // w
		double[] w_p = new double[mu]; // w'
		double sum = 0.0;
		double endDiff = 1e-8;
		for (int i = 0; i < mu; i++) {
			w_p[i] = Math.log(mu_p + 0.5) - Math.log(i + 1);
			sum += w_p[i];
		}
		double sum_p = 0.0;
		for (int i = 0; i < mu; i++) {
			w[i] = w_p[i] / sum;
			sum_p += Math.pow(w[i], 2);
		}

		// Set parameters
		// - Step-size control
		double mu_eff = 1 / sum_p;
		double c_sigma = (mu_eff + 2) / (DIM + mu_eff + 5);
		double d_sigma = 1 + 2
				* Math.max(0, Math.sqrt((mu_eff - 1) / (DIM + 1)) - 1)
				+ c_sigma;

		// Set parameters
		// - Covariance matrix adaptation
		double c_c = (4 + mu_eff / DIM) / (DIM + 4 + 2 * mu_eff / DIM);
		double c_1 = 2 / (Math.pow(DIM + 1.3, 2) + mu_eff);
		double alpha_mu = 2;
		double c_mu = Math.min(1 - c_1, alpha_mu * (mu_eff - 2 + 1 / mu_eff)
				/ (Math.pow(DIM + 2, 2) + alpha_mu * mu_eff / 2));

		// double E_norm = Math.sqrt(DIM)
		// * (1 - 0.25 / DIM + 1 / (21 * Math.pow(DIM, 2)));

		// Initialization
		SimpleMatrix p_sigma = new SimpleMatrix(DIM, 1);
		// double p_norm = 0;
		SimpleMatrix p_c = new SimpleMatrix(DIM, 1);
		// double h_sigma = 0;
		// double[][] g = sampling(population_);
		SimpleMatrix C = SimpleMatrix.identity(DIM);
		SimpleMatrix m = new SimpleMatrix(DIM, 1);
		double sigma = 3;
		double chiN = Math.sqrt(DIM)
				* (1 - 1 / (4 + DIM) + 1 / (21 * DIM * DIM));

		double[] best_score = new double[generation];

		for (int g = 0;; g++) {
			// Sample new population of search points
			SimpleMatrix[] x = new SimpleMatrix[lambda];
			SimpleMatrix[] y = new SimpleMatrix[lambda];
			SimpleMatrix[] z = new SimpleMatrix[lambda];

			SimpleMatrix B = new SimpleMatrix(DIM, DIM);
			SimpleMatrix D = new SimpleMatrix(DIM, DIM);
			evd_matrix(C, B, D);

			if (g == 0) {
				double[][] tmp = neo_sampling(lambda);
				for (int k = 0; k < lambda; k++) {
					x[k] = new SimpleMatrix(DIM, 1, true, tmp[k]);
				}
			} else {
				for (int k = 0; k < lambda; k++) {
					x[k] = new SimpleMatrix(DIM, 1);
					y[k] = new SimpleMatrix(DIM, 1);
					z[k] = new SimpleMatrix(DIM, 1);
					for (int i = 0; i < DIM; i++) {
						z[k].set(i, 0, rnd_.nextGaussian());
						y[k] = B.mult(D).mult(z[k]);
						x[k] = m.plus(sigma, y[k]);
					}
					// x[k].print();
				}
			}

			// Selection and recombination
			SimpleMatrix y_w;
			SimpleMatrix m_bak = m.copy();
			if (g == 0) {
				best_ = 0;
			} else {
				best_ = best_score[g - 1];
			}
			CMA_sort(x, lambda);

			best_score[g] = best_;
			m.set(0);
			for (int i = 0; i < mu; i++) {
				// x[i].print();
				m = m.plus(w[i], x[i]);
				y[i] = x[i].minus(m_bak).divide(sigma);
			}
			y_w = m.minus(m_bak).divide(sigma);

			// Step-size control
			SimpleMatrix C_nsqrt = B.mult(D.invert()).mult(B.transpose());
			p_sigma = p_sigma.scale(1 - c_sigma).plus(
					C_nsqrt.mult(y_w).scale(
							Math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)));
			sigma = sigma
					* Math.exp(c_sigma / d_sigma * (p_sigma.normF() / chiN - 1));

			// Covariance matrix adaptation
			int h_sigma;
			if (p_sigma.normF()
					/ Math.sqrt(1 - Math.pow(1 - c_sigma, 2 * (g + 1))) < (1.4 + 2 / (DIM + 1))
					* chiN)
				h_sigma = 1;
			else
				h_sigma = 0;
			double delta_h_sigma = (1 - h_sigma) * c_c * (2 - c_c);
			p_c = p_c.scale(1 - c_c).plus(
					y_w.scale(h_sigma * Math.sqrt(c_c * (2 - c_c) * mu_eff)));
			SimpleMatrix y_sqrsum = new SimpleMatrix(DIM, DIM);
			for (int i = 0; i < mu; i++) {
				y_sqrsum = y[i].mult(y[i].transpose()).scale(w[i]);
			}
			// y_sqrsum.print();
			C = C.scale(1 - c_1 - c_mu)
					.plus(p_c.mult(p_c.transpose())
							.plus(C.scale(delta_h_sigma)).scale(c_1))
					.plus(c_mu, y_sqrsum);

			if (g >= 20 && best_score[g] - best_score[g - 20] < endDiff
					&& limit_ * 2 >= (int) lambda
					* (100 + 50 * Math.pow((DIM + 3), 2) / Math.sqrt(lambda))) {
					break;
			}
			if (g + 1 >= generation && limit_ * 2 >= (int) lambda
					* (100 + 50 * Math.pow((DIM + 3), 2) / Math.sqrt(lambda))){
				break;
			}
		}
	}

	private void evd_matrix(SimpleMatrix C, SimpleMatrix B, SimpleMatrix D) {
		// C.print();
		SimpleEVD EVD_C = C.eig();
		int n = EVD_C.getNumberOfEigenvalues();
		for (int i = 0; i < n; i++) {
			B.setColumn(i, 0, EVD_C.getEigenVector(i).getMatrix().getData());
			D.set(i, i, EVD_C.getEigenvalue(i).getReal());
		}
	}

	@SuppressWarnings("unchecked")
	private void CMA_sort(SimpleMatrix[] x, int lambda) {

		class fitComparator implements Comparator {
			@Override
			public final int compare(Object a, Object b) {
				double diff = ((SimpleMatrix) a).get(DIM, 0)
						- ((SimpleMatrix) b).get(DIM, 0);
				if (diff == 0)
					return 0;
				else if (diff < 0)
					return 1;
				else
					return -1;
			}
		}

		best_ = 0;
		SimpleMatrix[] x_fit = new SimpleMatrix[lambda];
		for (int i = 0; i < lambda; i++) {
			double[] gene = x[i].getMatrix().getData();
			x_fit[i] = new SimpleMatrix(DIM + 1, 1);
			x_fit[i].setColumn(0, 0, gene);
			x_fit[i].set(DIM, 0, (Double) evaluation_.evaluate(gene));
			if (x_fit[i].get(DIM, 0) > best_) {
				best_ = x_fit[i].get(DIM, 0);
			}
			limit_--;
		}
		Arrays.sort(x_fit, new fitComparator());
		for (int i = 0; i < lambda; i++) {
			x[i].setColumn(0, 0, x_fit[i].extractMatrix(0, DIM, 0, 1)
					.getMatrix().getData());
		}
	}

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
		int r1, r2, r3, r4, r5;// random index
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
		double Fd = 0.3;// deviation
		double sigma = 0.01;//

		int population = 100;
		int generation = (limit_ / population - 1);
		int lp = 50;// learning period

		// initialization
		double randr = 0.0;
		double[] bestX = new double[DIM];
		double best = -100.0;
		int gen = 0;
		double[] F = new double[population];
		double K = 0.5;
		// double[] K = new double[population];

		// setting memory
		int[][] ns_mem = new int[lp][num_st];
		int[][] nf_mem = new int[lp][num_st];
		int[][] CR_suc_mem = new int[lp][num_st];
		double[][] CR_mem = new double[lp][num_st];
		int[] CRsuc = { 1, 1, 1, 1 };
		double[] CRsum = { 0.5, 0.5, 0.5, 0.5 };
		int[] ns = { 1, 1, 1, 1 };// strategy selection counter
		int[] nf = { 1, 1, 1, 1 };// strategy selection counter
		double[] p = { 0.25, 0.25, 0.25, 0.25 };// strategy
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
			limit_--;
			if (score[i] > best) {
				best = score[i];
				bestX = g[i].clone();
			}
		}
		double[] y = new double[DIM];
		double score_neo = 0;
		double[] gen_score = new double[generation];
		for (int i = 0; i < generation; i++) {
			gen_score[i] = best;
		}
		// start evolution
		for (int i = 0; i < generation; i++) {

			if (i >= lp) {
				for (int j = 0; j < num_st; j++) {
					CRsuc[j] = 0;
					CRsum[j] = 0;
					ns[j] = 0;
					nf[j] = 0;
					for (int k = 0; k < lp; k++) {
						CRsuc[j] += CR_suc_mem[k][j];
						CRsum[j] += CR_mem[k][j];
						ns[j] += ns_mem[k][j];
						nf[j] += nf_mem[k][j];
					}
				}

				s_sum = 0;
				for (int k = 0; k < num_st; k++) {
					s[k] = (double) ns[k] / (ns[k] + nf[k]) + sigma;
					s_sum += s[k];
				}

				for (int k = 0; k < num_st; k++) {
					p[k] = s[k] / s_sum;
					CRm[k] = CRsum[k] / CRsuc[k];
					ns_mem[i % lp][k] = 0;
					nf_mem[i % lp][k] = 0;
					CR_suc_mem[i % lp][k] = 1;
					CR_mem[i % lp][k] = 0.5;
				}

				for (int j = 0; j < population; j++) {
					for (int k = 0; k < num_st; k++) {
						// CR[j][k] = CRm[k] + Math.tan(Math.PI *
						// (rnd_.nextDouble() - 0.5)) * CRd;
						CR[j][k] = CRm[k] + rnd_.nextGaussian() * CRd;
						// CR[j][k] = Math.max(0.1, CR[j][k]);
					}
				}
			} else {
				for (int j = 0; j < population; j++) {
					for (int k = 0; k < num_st; k++) {
						// CR[j][k] = CRm[k] + Math.tan(Math.PI *
						// (rnd_.nextDouble() - 0.5)) * CRd;
						CR[j][k] = CRm[k] + rnd_.nextGaussian() * CRd;
						CR[j][k] = Math.max(0.1, CR[j][k]);
						ns_mem[i % lp][k] = 0;
						nf_mem[i % lp][k] = 0;
						CR_suc_mem[i % lp][k] = 1;
						CR_mem[i % lp][k] = 0.5;
					}
				}
			}
			K = rnd_.nextDouble();

			for (int j = 0; j < population; j++) {
				// set F, location decided
				F[j] = Fm + rnd_.nextGaussian() * Fd;
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
					st = 1; // DE/rand/1/bin
				} else if (randr <= p[0] + p[1]) {
					st = 2; // DE/rand-to-best/2/bin
				} else if (randr <= p[0] + p[1] + p[2]) {
					st = 3; // DE/rand/2/bin
				} else {
					st = 4; // DE/current-to-rand/1/bin
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
						y[k] = g[j][k] + K * (g[r1][k] - g[j][k]) + F[j]
								* (g[r2][k] - g[r3][k]);
					}
				}

				score_neo = (Double) evaluation_.evaluate(y);
				limit_--;
				if (score_neo > score[j]) {
					g[j] = y.clone();
					score[j] = score_neo;

					// update counter
					ns_mem[i % lp][st - 1]++;
					CR_suc_mem[i % lp][st - 1]++;
					CR_mem[i % lp][st - 1] += CR[j][st - 1];

					if (score[j] > best) {
						best = score[j];
						bestX = g[j].clone();
					}
				} else {
					nf_mem[i % lp][st - 1]++;
				}
			}
			gen++;
			gen_score[i] = best;
			if (gen > 100 && gen_score[i] - gen_score[i - 20] < 0.00000001) {
				break;
			}
		}
	}

	// Differential Evolution
	private void Cauchy_DE() {
		// set parameters

		int population = 100;
		int generation = limit_ / population - 1;
		double[] CR = new double[population];
		double[] F = new double[population];
		double[] CR_mem = new double[population];
		double[] F_mem = new double[population];
		double F_avg;
		double CR_avg;
		double F_sum;
		double CR_sum;

		// initialization
		double randr = 0.0;
		int randi = 0;
		int a, b, c;
		int counter;
		double[][] g = sampling(population);
		double[] score = new double[population];
		for (int i = 0; i < population; i++) {
			score[i] = (Double) evaluation_.evaluate(g[i]);
			limit_--;
			CR[i] = 0.9;
			F[i] = 0.5;
		}
		double[] y = new double[DIM];
		double score_neo = 0;
		for (int i = 0; i < generation; i++) {
			counter = 0;
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
					if (randi == k || randr < CR[j]) {
						y[k] = g[a][k] + F[j] * (g[b][k] - g[c][k]);
						y[k] = Math.max(y[k], -5);
						y[k] = Math.min(y[k], 5);
					} else {
						y[k] = g[j][k];
					}
				}
				score_neo = (Double) evaluation_.evaluate(y);
				limit_--;
				if (score_neo > score[j]) {
					g[j] = y.clone();
					score[j] = score_neo;
					F_mem[counter] = F[j];
					CR_mem[counter] = CR[j];
					counter++;
				}
			}
			if (counter > 0) {
				F_sum = 0;
				CR_sum = 0;
				for (int k = 0; k < counter; k++) {
					F_sum += F_mem[k];
					CR_sum += CR_mem[k];
				}
				F_avg = F_sum / counter;
				CR_avg = CR_sum / counter;
				for (int j = 0; j < population; j++) {
					F[j] = Math.tan(Math.PI * (rnd_.nextDouble() - 0.5)) * 0.1
							+ F_avg;
					CR[j] = Math.tan(Math.PI * (rnd_.nextDouble() - 0.5)) * 0.1
							+ CR_avg;
					F[j] = Math.max(F[j], 0.1);
					F[j] = Math.min(F[j], 1);
					CR[j] = Math.max(CR[j], 0);
					CR[j] = Math.min(CR[j], 1);
				}
			}

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

	// TODO
	// Mean-Variance Mapping Optimization
	@SuppressWarnings("unused")
	private void MVMO() {

		// set parameter
		int lambda = 20;
		double gamma = 1;
		double f_s_ini = 0.1;
		double f_s_final = 20;
		double f_s = f_s_ini;
		double f_s_star = f_s;
		double d_0 = 0.25;
		double m_ini = DIM / 6;
		double m_final = DIM / 2;
		double alpha_LS_min = 0.23;
		double s_d = 75;
		double k_d = 0.0505 / DIM + 1.0;
		int lsTrail = 200;
		int i_max = Math.min(100000, limit_-100);//limit_ - 10;

		// initial
		double x_sum = 0, x_var = 0;
		double[] score = new double[lambda];
		double[] mean = new double[DIM];
		double[] shape = new double[DIM];
		double[] d_factor = new double[DIM];
		int m, m_star;
		// int index = 0;
		double alpha;
		double[][] archive = new double[lambda][DIM + 1];

		// avg, si, di, si1, si2
		double[][] param = new double[5][DIM];

		double score_neo = 0;
		double[] x = new double[DIM];
		double[][] first = neo_sampling(lambda);
		double h_x, h0, h1, x_star;

		for (int i = 0; i < lambda; i++) {
			archive[i] = Arrays.copyOf(normalize(first[i]), DIM + 1);
			archive[i][DIM] = (Double) evaluation_.evaluate(first[i]);
			limit_--;
		}
		Arrays.sort(archive, new java.util.Comparator<double[]>() {
			public int compare(double[] a, double[] b) {
				return Double.compare(b[DIM], a[DIM]);
			}
		});

		for (int i = 0; i < DIM; i++) {
			param[2][i] = 1;
		}

		int i = 0;
		while (i < i_max) {

			alpha = (double) i / i_max;

			if (rnd_.nextDouble() < gamma && alpha > alpha_LS_min) {
				double[] temp = ls(denormalize(x),0.1,lsTrail);
				i=i+lsTrail;
				x = Arrays.copyOfRange(temp, 0, DIM);
				score_neo = temp[DIM];
				gamma = 0;
			} else {
				score_neo = (Double) evaluation_.evaluate(denormalize(x));
				limit_--;
				i++;
			}

			// update archive
			if (score_neo > archive[lambda - 1][DIM]) {
				archive[lambda - 1] = Arrays.copyOf(x, DIM + 1);
				archive[lambda - 1][DIM] = score_neo;
				Arrays.sort(archive, new java.util.Comparator<double[]>() {
					public int compare(double[] a, double[] b) {
						return Double.compare(b[DIM], a[DIM]);
					}
				});
				for (int j = 0; j < DIM; j++) {
					x_sum = 0;
					x_var = 0;
					for (int k = 0; k < lambda; k++) {
						x_sum += archive[k][j];
					}
					param[0][j] = x_sum / lambda;

					for (int k = 0; k < lambda; k++) {
						x_var += Math.pow(archive[k][j] - param[0][j], 2);
					}

					if (param[1][j] > 20) {
						param[1][j] += 1;
					}

					if (x_var == 0) {
						param[3][j] = param[1][j];// si1
						param[4][j] = param[1][j];// si2
						if (param[1][j] < s_d) {
							s_d = s_d * k_d;
							param[3][j] = s_d;
						} else if (param[1][j] > s_d) {
							s_d = s_d / k_d;
							param[3][j] = s_d;
						}

					} else {
						param[1][j] = -Math.log((x_var) / lambda) * f_s;
						param[3][j] = param[1][j];// si1
						param[4][j] = param[1][j];// si2
						if (param[1][j] > 0) {
							double delta_d = (1 + d_0) + 2 * d_0
									* (rnd_.nextDouble() - 0.5);
							if (param[1][j] > param[2][j]) {
								param[2][j] = param[2][j] * delta_d;
							} else {
								param[2][j] = param[2][j] / delta_d;
							}
							if (rnd_.nextDouble() > 0.5) {
								param[3][j] = param[1][j];
								param[4][j] = param[2][j];
							} else {
								param[3][j] = param[2][j];
								param[4][j] = param[1][j];
							}
						}
					}

				}

			}

			// gen offspring
			m_star = (int) Math.round(m_ini - Math.pow(alpha, 2)
					* (m_ini - m_final));
			m = (int) Math.round(m_final + rnd_.nextDouble()
					* (m_star - m_final));

			int[] index = getRandomPermutation(DIM);
			// index += m;
			for (int j = 0; j < m; j++) {
				int k = index[j];
				// int k = (j + index)%10;
				x_star = rnd_.nextDouble();
				h_x = MVMO_h(param[0][k], param[3][k], param[4][k], x_star);
				h0 = MVMO_h(param[0][k], param[3][k], param[4][k], 0);
				h1 = MVMO_h(param[0][k], param[3][k], param[4][k], 1);
				x[k] = h_x + (1 - h1 + h0) * x_star - h0;
			}
			for (int j = m; j < DIM; j++) {
				int k = index[j];
				// int k = (j + index)%10;
				x[k] = archive[0][k];
			}

			f_s_star = f_s_ini + Math.pow(alpha, 2) * (f_s_final - f_s_ini);
			f_s = f_s_star * (1 + rnd_.nextDouble());

		}

	}

	private double[] normalize(double[] a) {
		int l = Array.getLength(a);
		double[] b = new double[l];
		for (int i = 0; i < l; i++) {
			b[i] = (a[i] + 5) / 10;
		}
		return b;
	}

	private double[] denormalize(double[] a) {
		int l = Array.getLength(a);
		double[] b = new double[l];
		for (int i = 0; i < l; i++) {
			b[i] = a[i] * 10 - 5;
		}
		return b;
	}

	private double MVMO_h(double avg, double s1, double s2, double x) {
		return avg * (1 - Math.exp(-x * s1)) + (1 - avg)
				* Math.exp(-(1 - x) * s2);
	}

	private int[] getRandomPermutation(int length) {

		// initialize array and fill it with {0,1,2...}
		int[] array = new int[length];
		for (int i = 0; i < array.length; i++)
			array[i] = i;

		for (int i = 0; i < length; i++) {

			// randomly chosen position in array whose element
			// will be swapped with the element in position i
			// note that when i = 0, any position can chosen (0 thru length-1)
			// when i = 1, only positions 1 through length -1
			// NOTE: r is an instance of java.util.Random
			int ran = i + rnd_.nextInt(length - i);

			// perform swap
			int temp = array[i];
			array[i] = array[ran];
			array[ran] = temp;
		}
		return array;
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
		int half = (int) Math.round((double) population / 2);
		for (int i = 0; i < half; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
		}
		for (int i = half; i < population; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = -g[i - population / 2][j];
			}
		}
		return g;
	}

	private double[][] neo_sampling(int population) {
		double[][] g = new double[population][DIM];
		int half = (int) Math.round((double) population / 2);
		for (int i = 0; i < half; i++) {
			for (int j = 0; j < DIM; j++) {
				g[i][j] = rnd_.nextDouble() * 10 - 5;
			}
		}
		for (int i = half; i < population; i++) {
			for (int j = 0; j < DIM; j++) {
				if (g[i - half][j] > 0) {
					g[i][j] = g[i - half][j] / 3 - 10 / 3;
				} else if (g[i - half][j] < 0) {
					g[i][j] = g[i - half][j] / 3 + 10 / 3;
				} else {
					g[i][j] = rnd_.nextDouble() * 10 - 5;
				}
			}
		}
		return g;
	}
	
	private void SAA() {
		double[] saa_test = new double[10];
		for (int i = 0; i < 10; i++) {
			saa_test[i] = rnd_.nextDouble() * 10 - 5;
			// System.out.println(saa_test[i]);
		}
		sa(saa_test);
		//ls(saa_test,0.1,150);
		saa_test[0] = 0;
	}
	
	private double[] ls(double[] x, double r, int gen){
		double best = (double) evaluation_.evaluate(x);
		limit_--;
		double decay_scale = 0.999;
		double[] g = new double[DIM+1];
		double[] y = new double[DIM]; 
		double tol = 1e-8;
		double nextbest = -10000;
		double score;
		
		for (int i = 0; i < gen; i++) {
			
			for(int j = 0;j<DIM;j++){
				y[j] = x[j] + (rnd_.nextDouble()-0.5)*r*10;
			}
			score = (double) evaluation_.evaluate(y);
			limit_--;
			if(score>best){
				nextbest = best;
				best = score;
				x = y.clone();
				//r = r * decay_scale;
			}
			
			if(Math.abs(best - nextbest)<tol){
				break;
			}
		}
		g = Arrays.copyOf(x,DIM+1);
		g[DIM] = best;
		return g;
	}
	

	public void sa(double[] x)
	{
		int markovlength = 300;
		double decay_scale = 0.95;
		double step_factor = 0.2;
		double temperature = 500;
		double tolerance = 1e-8;
		double[] pre, next, prebest, best;
		double acceptpoints = 0.0;
		int i;
		double pre_s = -999,next_s = -999,best_s = -999, prebest_s = -999;
		Random rnd_ = new Random();

		pre = new double[10];
		next = new double[10];
		prebest = new double[10];
		best = new double[10];

		for (i = 0; i < 10; i++) {
			pre[i] = -x[i];
			prebest[i] = pre[i];
			best[i] = pre[i];
			// System.out.println(max[i]);
		}

		do {
			temperature = temperature * decay_scale;
			acceptpoints = 0.0;

			for (int trials = 0; trials < markovlength; trials++) {
				// System.out.println("trials is: " + trials);
				do {
					for (i = 0; i < 10; i++) {
						next[i] = pre[i] + step_factor * 5
								* (rnd_.nextDouble() - 0.5);
						// System.out.println("next" + i +" is " + next[i]);
					}

				}while(!within_boundary(next));
				
				next_s = (double)evaluation_.evaluate(next);
				//best_s = (double)evaluation_.evaluate(best);
				//pre_s = (double)evaluation_.evaluate(pre);
				
				if(next_s > best_s)
				{
					for(i=0;i<10;i++)
					{

						prebest[i] = best[i];
						best[i] = next[i];
					}
					prebest_s = best_s;
					best_s = next_s;
				}

				if (pre_s - next_s < 0) {
					for (i = 0; i < 10; i++) {
						pre[i] = next[i];
					}
					pre_s = next_s;
					acceptpoints++;
				} else {
					double change = (next_s - pre_s);
					// System.out.println(change + " " +
					// (double)evaluation_.evaluate(next) + " " +
					// (double)evaluation_.evaluate(pre));
					if (Math.exp(change) / temperature > rnd_.nextDouble()) {
						for (i = 0; i < 10; i++) {
							pre[i] = next[i];
						}
						pre_s = next_s;
						acceptpoints++;
					}
				}
			}

		} while(Math.abs(best_s) - Math.abs(pre_s) > tolerance);

	}

	private boolean within_boundary(double[] x) {
		for (int i = 0; i < 10; i++) {
			if (x[i] <= -5 || x[i] >= 5) {
				return false;
			}
		}
		return true;
	}
	
	public static void main(String[] args) {
	}
}
