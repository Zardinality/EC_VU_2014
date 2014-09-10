import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.lang.Math;


public class player19 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
	Integer population_;
	Integer generation_;
	Integer lambda_;//number# of offspring
	double glr_;//global learning rate
	double llr_;//local learning rate
	double beta_;
	
	
	public player19()
	{
		rnd_ = new Random();
	}
	
	//@Override
	public void setSeed(long seed)
	{
		// Set seed of algortihms random process
		rnd_.setSeed(seed);
	}
	
	//@Override
	public void setEvaluation(ContestEvaluation evaluation)
	{
		// Set evaluation problem used in the run
		evaluation_ = evaluation;
		
		// Get evaluation properties
		Properties props = evaluation.getProperties();
		// Property keys depend on specific evaluation
		boolean mm = Boolean.parseBoolean(props.getProperty("Multimodal"));
		boolean rg = Boolean.parseBoolean(props.getProperty("Regular"));
		boolean sp = Boolean.parseBoolean(props.getProperty("Separable"));
		double limit = Double.parseDouble(props.getProperty("Evaluations"));
		// Do sth with property values, e.g. specify relevant settings of your algorithm
		population_ = (int)Math.round(Math.sqrt(limit));
		generation_ = (int)Math.floor(limit)/(population_);
		glr_ = 1/Math.sqrt(limit);
		llr_ = 1/Math.sqrt(2 * limit);
		lambda_ = population_ * 7;
		beta_ = 0.087266462599716;
	}
	
	//@Override
	public void run()
	{
		// Evaluating your results
		// E.g. evaluating a series of true/false predictions
		// boolean pred[] = ...
		int i,j,k;//iterator
		
		//first generation
		double g[][] = sampling(); 

		double gnext[][] = new double[population_][10];
		double gtemp[][] = new double[population_][10];
		
		
		double c = 0.95;//constant for mutation changing rate
		double ps[] = new double[population_];
		double s[] = new double[population_];
		double sigma[] = new double[population_];
		Double score[] = new Double[population_];
		for(i = 0; i<population_; i++)
		{
			sigma[i] = 1;
			score[i] = (Double)evaluation_.evaluate(g[i]);
		}
		
		Double rtemp = 0.0;
		Double tempb = new Double(0);
		//double rate = 0.0
		
		
		for(i=0;i<generation_-1;i++)
		{
			for(j=0;j<population_;j++)
			{
				g[j] = new double[10];
				System.arraycopy(g[j], 0, gtemp[j], 0, 10);
			}
			//System.out.println(Double.toString(evaluation_.getFinalResult()));
			for(j = 0; j < population_; j++)
			{	
				//rtemp = sigma[j] * rnd_.nextGaussian();
				for(k = 0; k < 10; k++)
				{
					gnext[j][k] = gtemp[j][k] + sigma[j] * rnd_.nextGaussian();
				}
				tempb = (Double)evaluation_.evaluate(gnext[j]);
				//System.out.println(Double.toString(tempb));
				if(score[j] >= tempb)
				{
					//System.out.println(Double.toString(tempb));
					s[j] = 0;
				}
				else
				{
					//System.out.println("enter");
					s[j] = 1;
					g[j] = gnext[j];
					score[j] = tempb;
				}
				ps[j] = ps[j] + s[j];
				if(ps[j]/i > 0.205 && sigma[j] > 0.1)
				{
					sigma[j] = sigma[j] * c;
				}
				else if(ps[j]/i < 0.195 && sigma[j] < 5)
				{
					sigma[j] = sigma[j] / c;
				}
			}
					
		}
		
	}
	
	private double[][] sampling()
	{
		int i,j,k;
		double g[][] = new double[population_][10]; 
		if(population_>1024)
		{
			for(i=0;i<population_;i++)
			{
				for(j=0;j<10;j++)
				{
					g[i][j] = rnd_.nextDouble()*10-5;
				}
			}
		}
		else
		{
			for(i=0;i<population_;i++)
			{
				for(j=0;j<10;j++)
				{
					g[i][j] = rnd_.nextDouble()*10-5;
				}
			}
		}
		return g;
	}
	/*
	private double[][] recombination(double[][] g)//
	{
		int i,j,k;
		int[] index = new int[2];
		double[][] gnext = new double[lambda_][10];
		double[][] snext = new double[lambda_][10];
		for(i = 0; i < lambda_; i++)
		{
			index[1] = rnd_.nextInt(population_);
			index[2] = rnd_.nextInt(population_);
			for(j = 0; j < 65; j++)
			{
				gnext[i][j] = 0.5 * (g[index[1]][j]+g[index[2]][j]);
			}
		}
		return gnext;
	}
	
	private double[][] mutation(double[][] gnext)
	{
		int i,j,k;
		double r = 0.0;
		double[][] cov = new double[10][10];
		double[][] gtemp = new double[lambda_][10];
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
					if(j < k)
					{
						cov[j][k] = 0.5 * (gtemp[i][10+j] * gtemp[i][10+j] - gtemp[i][10+k] * gtemp[i][10+k]) * Math.tan(2 * gtemp[i][j*9+k-1]);
					}
					else if(j > k)
					{
						cov[j][k] = 0.5 * (gtemp[i][10+j] * gtemp[i][10+j] - gtemp[i][10+k] * gtemp[i][10+k]) * Math.tan(2 * gtemp[i][j*9+k]);
					}
					else
					{
						cov[j][k] = gtemp[i][10+j] * gtemp[i][10+j];
					}
				}
			}
			for(j = 0; j < 10; j++)
			{
                
			}
		}
		return gtemp;
	}
	
	private double[][] selection(double[][] gnext)
	{
		int i,j,k;
		int index = 0;
		double g[][] = new double[population_][65];
		Double[] score = new Double[lambda_];
		Double temp = 0.0;
		
		for(i = 0; i < lambda_; i++)
		{
			score[i] = (Double)evaluation_.evaluate(gnext[i]);
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
	
	*/
	
	public Double getResult()
	{
		return evaluation_.getFinalResult();
	}
}
