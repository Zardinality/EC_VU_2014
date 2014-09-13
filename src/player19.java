import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.util.Arrays;
import java.lang.Math;

public class player19 implements ContestSubmission
{
        public static final int DIM = 10;
        
	Random rnd_;
	ContestEvaluation evaluation_;
	Integer population_;
	Integer generation_;
	Integer lambda_;//number# of offspring
	double glr_;//global learning rate
	double llr_;//local learning rate
	double beta_;//turning angle
	int algIndex_;
        
        boolean mm_, rg_, sp_;
	
	
	public player19()
	{
		rnd_ = new Random();
	}
	
	
	@Override
	public void setSeed(long seed)
	{
		// Set seed of algortihms random process
		rnd_.setSeed(seed);
	}
	
	@Override
	public void setEvaluation(ContestEvaluation evaluation)
	{
		// Set evaluation problem used in the run
		evaluation_ = evaluation;
		
		// Get evaluation properties
		Properties props = evaluation.getProperties();
		// Property keys depend on specific evaluation
		mm_ = Boolean.parseBoolean(props.getProperty("Multimodal"));
		rg_ = Boolean.parseBoolean(props.getProperty("Regular"));
		sp_ = Boolean.parseBoolean(props.getProperty("Separable"));
		double limit = Double.parseDouble(props.getProperty("Evaluations"));
		// Do sth with property values, e.g. specify relevant settings of your algorithm
		population_ = (int)Math.round(Math.sqrt(limit))/8;
		generation_ = ((int)Math.floor(limit)-population_)/(population_*6);
		glr_ = 1/Math.sqrt(2 * limit);
		llr_ = 1/Math.sqrt(2 * Math.sqrt(limit));
		lambda_ = population_ * 6;
		beta_ = 0.087266462599716;
		algIndex_ = 0;
 		if(sp_)
		{
		
			population_ = (int)Math.floor(limit);
			algIndex_ = 1;
		}
		else if(rg_)
		{
			population_ = 10;
			lambda_ = 70;
			generation_ = ((int)Math.floor(limit)-population_)/lambda_;
			algIndex_ = 3; 
		}
		else algIndex_ = 3;
 		
	}
	
	@Override
	public void run()
	{
//		int i,j,k;//iterator
//		
//		if(algIndex_ == 1)
//		{
//			double[] g = separable(true);
//			evaluation_.evaluate(g);
//			return;
//		}
//		else if(algIndex_ == 2)
//		{
//			double[] g = separable(false);
//			evaluation_.evaluate(g);
//			return;
//		}
//		else if(algIndex_ == 3)
//		{
//			double[][] g = samplingES();
//			double[][] gnext = new double[lambda_][65];
//			for(i = 0; i<generation_; i++)
//			{
//				gnext = recombination(g);
//				gnext = mutation(gnext);
//				g = selection(gnext);
//			}
//			return;
//		}
                double[] g = evolution(mm_, rg_, sp_);
                evaluation_.evaluate(g);
	}
	
	private double[][] sampling()
	{
		int i,j,k;
		double[][] g = new double[population_][10]; 
		for(i=0;i<population_;i++)
		{
			for(j=0;j<10;j++)
			{
				g[i][j] = rnd_.nextDouble()*10-5;
			}
		}
		return g;
	}
	
	private double[][] samplingES()
	{
		int i,j,k;
		double[][] g = new double[population_][65]; 
		
		for(i=0;i<population_;i++)
		{
			for(j=0;j<10;j++)
			{
				g[i][j] = rnd_.nextDouble()*10-5;
			}
			for(j = 10; j < 20; j++)
			{
				g[i][j] = 0.1;
			}
			for(j = 20; j < 65; j++)
			{
				g[i][j] = 0;
			}
		}
		
		return g;
	}
	
	private double[][] recombination(double[][] g)// mean-value recombine
	{
		int i,j,k;
		int[] index = new int[2];
		double[][] gnext = new double[lambda_][65];
		for(i = 0; i < lambda_; i++)
		{
			index[0] = rnd_.nextInt(population_);
			index[1] = rnd_.nextInt(population_);
			for(j = 0; j < 65; j++)
			{
				gnext[i][j] = 0.5 * (g[index[0]][j]+g[index[1]][j]);
			}
		}
		return gnext;
	}
	
	private double[][] recombinationD(double[][] g)//discrete recombine
	{
		int i,j,k;
		int[] index = new int[2];
		double[][] gnext = new double[lambda_][65];
		for(i = 0; i < lambda_; i++)
		{
			index[0] = rnd_.nextInt(population_);
			index[1] = rnd_.nextInt(population_);
			for(j = 0; j < 65; j++)
			{
				if(rnd_.nextDouble() > 0.5) gnext[i][j] = g[index[0]][j];
				else gnext[i][j] = g[index[1]][j];
			}
		}
		return gnext;
	}
	

	private double[][] recombinationRND(double[][] g)//
	{
		int i,j,k;
		int[] index = new int[2];
		double ratio = 0.0;
		double[][] gnext = new double[lambda_][65];
		for(i = 0; i < lambda_; i++)
		{
			index[0] = rnd_.nextInt(population_);
			index[1] = rnd_.nextInt(population_);
			ratio = rnd_.nextDouble();
			for(j = 0; j < 65; j++)
			{
				gnext[i][j] = ratio * g[index[0]][j] + (1 - ratio) * g[index[1]][j];
			}
		}
		return gnext;
	}
	
	private double[][] mutation(double[][] gnext)
	{
		int i,j,k;
		double sum;
		double r = 0.0;
		double[][] cov = new double[10][10];
		double[][] L = new double[10][10];
		double[][] gtemp = new double[lambda_][65];
		double[] x = new double[10];
		for(i = 0; i < lambda_; i++)
		{
			r = glr_ * rnd_.nextGaussian();
			for(j = 10; j < 20; j++)
			{
				gtemp[i][j] = gnext[i][j] + r + llr_ * rnd_.nextGaussian();
			}
			for(j = 20; j < 65; j++)
			{
				gtemp[i][j] = gnext[i][j] + beta_ * rnd_.nextGaussian();
			}
			for(j = 0; j < 10; j++)
			{
				for(k = 0; k < 10; k++)
				{
					if(j > k)
					{
						cov[j][k] = -0.5 * (gtemp[i][10+j] * gtemp[i][10+j] - gtemp[i][10+k] * gtemp[i][10+k]) * Math.tan(2 * gtemp[i][19+(19-k)*k/2+j-k]);
					}
					else if(j < k)
					{
						cov[j][k] = 0.5 * (gtemp[i][10+j] * gtemp[i][10+j] - gtemp[i][10+k] * gtemp[i][10+k]) * Math.tan(2 * gtemp[i][19+(19-j)*j/2+k-j]);
					}
					else
					{
						cov[j][k] = gtemp[i][10+j] * gtemp[i][10+j];
					}
				}
			}
			L = Cholesky.cholesky(cov);
			for(j = 0; j < 10; j++)
			{
				x[j] = rnd_.nextGaussian();
			}
			for(j = 0; j < 10; j++)
			{
				sum = 0;
				for(k = 0; k < j+1; k++)
				{
					sum += L[j][k] * x[k];
				}
				gtemp[i][j] = gnext[i][j] + sum;
			}
			//for(j = 0; j < 10; j++)
			//{
			//	gtemp[i][j] = gnext[i][j] + gtemp[i][j+10] * rnd_.nextGaussian();
			//}
		}
		return gtemp;
	}
	
	
	private double[][] selection(double[][] gnext)
	{
		int i,j,k;
		int index = 0;
		double[][] g = new double[population_][65];
		Double[] score = new Double[lambda_];
		Double temp = 0.0;
		
		for(i = 0; i < lambda_; i++)
		{
			score[i] = (Double)evaluation_.evaluate(Arrays.copyOfRange(gnext[i], 0, 10));
		}
		for(i = 0; i < population_; i++)
		{
			for(j = i + 1; j < lambda_; j++)
			{
				if(score[i]<score[j])
				{
					temp = score[j];
					score[j] = score[i];
					score[i] = temp;
					index = j;
				}
			}
			g[i] = (double[])gnext[index].clone();
		}
		return g;
	}
	
        private double[] evolution(boolean mm, boolean rg, boolean sp)
        {
            double[] g = new double[DIM];
            if (mm)
                {
                    g = func1();
                }
                else if (rg)
                {
                    g = func3();
                }
                else
                {
                    g = func2();
                }
            return g;
        }
        
	private double[] separable(boolean rg)
	{
		int i,j,k;
		double[] g = new double[10];
		for(i = 0; i < 10; i++)
		{
			if(rg)
			{
				g[i] = trialRg(i,(population_ - 1)/10);
			}
			else
			{
				g[i] = trial(i,(population_ - 1)/10);
			}
		}
		return g;
	}
	
	private double trial(int dim, int population)
	{
		int i,j,k;
		double best = 0;
		double bestscore = -999;
		double result = 0.0;
		double[] g = new double[DIM];
		for(i = 0; i < DIM; i++)
		{
			g[i] = 1;
		}
		for(i = 0; i < population; i++)
		{
			g[dim] = rnd_.nextDouble() * 10 - 5;
			//g[dim] = rnd_.nextDouble() * 10 * (i + 1) / population - 5;
			result = (Double)evaluation_.evaluate(g);
			if(bestscore < result)
			{
				bestscore = result;
				best = g[dim];
			}
		}
		return best;
	}
	
	private double trialRg(int dim, int population)
	{
		int i,j,k;
		double best = 0;
		double bestscore = -999;
		double result = 0.0;
		int pop = (int)Math.round(Math.sqrt((double)population));
		int gen = population / pop;
		double[][] g = new double[pop][DIM];
		double[][] gnext = new double[pop][DIM];
		for(i = 0; i < pop; i++)
		{
			for(j = 0; j < DIM; j++)
			{
				g[i][j] = 1;
				gnext[i][j] = 1;
			}
			g[i][dim] = rnd_.nextDouble() * 10 - 5;
		}
		
		double c = 0.95;//constant for mutation changing rate
		double[] ps = new double[pop];//
		double[] s = new double[pop];
		double[] sigma = new double[pop];
		Double[] score = new Double[pop];
		for(i = 0; i < pop; i++)
		{
			sigma[i] = 0.1;
			score[i] = (Double)evaluation_.evaluate(g[i]);
		}
		
		double rtemp = 0.0;
		double tempb = 0.0;
		
		for(i = 0; i < gen - 1; i++)
		{
			for(j = 0; j < pop; j++)
			{	
				gnext[j][dim] = g[j][dim] + sigma[j] * rnd_.nextGaussian();
				
				tempb = (Double)evaluation_.evaluate(gnext[j]);
				if(score[j] >= tempb)
				{
					s[j] = 0;
				}
				else
				{
					if(bestscore < tempb)
					{
						bestscore = tempb;
						best = gnext[j][dim];
					}
					//System.out.println("enter");
					s[j] = 1;
					rtemp = gnext[j][dim];
					g[j][dim] = rtemp;
					score[j] = tempb;
				}
				ps[j] = ps[j] + s[j];
				if(ps[j]/i > 0.205 && sigma[j] > 0.01)
				{
					sigma[j] = sigma[j] * c;
				}
				else if(ps[j]/i < 0.195 && sigma[j] < 2)
				{
					sigma[j] = sigma[j] / c;
				}
			}
		}
		return best;
	}
	
        private double[] func1()
        {
            
        }
        
        private double[] func2()
        {
            
        }
        
        private double[] func3(int dim, int population)
	{
		int i,j,k;
		double best = 0;
		double bestscore = -999;
		double result = 0.0;
		double[] g = new double[DIM];
		for(i = 0; i < DIM; i++)
		{
			g[i] = 1;
		}
		for(i = 0; i < population; i++)
		{
			g[dim] = rnd_.nextDouble() * 10 - 5;
			//g[dim] = rnd_.nextDouble() * 10 * (i + 1) / population - 5;
			result = (Double)evaluation_.evaluate(g);
			if(bestscore < result)
			{
				bestscore = result;
				best = g[dim];
			}
		}
		return g;
	}
 	
	public Double getResult()
	{
		return evaluation_.getFinalResult();
	} 
	
}