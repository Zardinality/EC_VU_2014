import java.util.Properties;
import java.util.Random;

import org.vu.contest.ContestEvaluation;

import java.lang.Math;


public class FletcherEvaluation implements ContestEvaluation{
	
	
	private final static int EVALS_LIMIT_ = 500000;
	private double best_;
	private int evaluations_;
	
	private String multimodal_ = "true";
	private String regular_ = "false";
	private String separable_ = "false";
	private String evals_ = Integer.toString(EVALS_LIMIT_);
	private Random rnd_;
	double[] alpha_;
	int[][] a_;
	int[][] b_;
	
	public FletcherEvaluation()
	{
		best_ = -100;
		evaluations_ = 0;
		rnd_ = new Random();
		rnd_.setSeed(0);
		alpha_ = new double[10];
		a_ = new int[10][10];
		b_ = new int[10][10];
		for(int i=0;i<10;i++)
		{
			alpha_[i] = rnd_.nextDouble()*2*Math.PI-Math.PI;
			for(int j=0;j<10;j++)
			{
				a_[i][j] = rnd_.nextInt(201)-100;
				b_[i][j] = rnd_.nextInt(201)-100;
			}
		}
	}
	
	private double func(double[] x, double[] alpha, int[][] a, int[][] b)
	{
		double f = 0;
		for(int i=0;i<10;i++)
		{
			f += Math.pow((func_A(a[i],b[i],alpha) - func_B(a[i],b[i],x)), 2);
		}
		return f;
	}
	private double func_A(int[] a,int[] b,double[] alpha)
	{
		double sum = 0;
		for(int j = 0;j<10;j++)
		{	
			sum += (a[j]*Math.sin(alpha[j]) + b[j]*Math.cos(alpha[j]));
		}
		return sum;
	}
	
	private double func_B(int[] a,int[] b,double[] x)
	{
		double sum = 0;
		for(int j = 0;j<10;j++)
		{
			sum += (a[j]*Math.sin(x[j]/5*Math.PI) + b[j]*Math.cos(x[j]/5*Math.PI));
		}
		return sum;
	}
	
	@Override
	public Object evaluate(Object result) 
	{
		// Check argument
		if(!(result instanceof double[])) throw new IllegalArgumentException();
		double ind[] = (double[]) result;
		if(ind.length!=10) throw new IllegalArgumentException();
			
		if(evaluations_>EVALS_LIMIT_) return null;

		double f = 0;
		f = 10 - func(ind,alpha_,a_,b_);
		if(f>best_) best_ = f;
		evaluations_++;
		
		return new Double(f);
	}
	
	@Override
	public Object getData(Object arg0) 
	{
		return null;
	}

	@Override
	public double getFinalResult() 
	{
		return best_;
	}
	
	@Override
	public Properties getProperties() 
	{
		Properties props = new Properties();
		props.put("Multimodal", multimodal_);
		props.put("Regular", regular_);
		props.put("Separable", separable_);
		props.put("Evaluations", evals_);
		return props;
	}
}
