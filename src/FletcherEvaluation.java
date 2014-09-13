import java.util.Properties;
import java.util.Random;

import org.vu.contest.ContestEvaluation;

import java.lang.Math;


public class FletcherEvaluation implements ContestEvaluation{
	
	
	private final static int EVALS_LIMIT_ = 10000;
	private double best_;
	private int evaluations_;
	
	private String multimodal_ = "true";
	private String regular_ = "false";
	private String separable_ = "false";
	private String evals_ = Integer.toString(EVALS_LIMIT_);
	
	public FletcherEvaluation()
	{
		best_ = 100;
		evaluations_ =0;
	}
	
	private double func(double[] x)
	{
		double f=0;
		Random rnd = new Random();
		for(int i=0;i<10;i++)
		{
			int a = rnd.nextInt(200)-100;
			int b = rnd.nextInt(200)-100;
			double alpha = rnd.nextDouble()*2*Math.PI-Math.PI;
			
			f += Math.pow((func_A(a,b,alpha) - func_B(a,b,x[i])), 2);
			
		}
		return f;
	}
	private double func_A(int a,int b,double alpha)
	{
		
		double sum = 0;
		for(int j = 0;j<10;j++)
		{	
			sum += (a*Math.sin(alpha) + b*Math.cos(alpha));
		}
		return sum;
	}
	
	private double func_B(int a,int b,double x)
	{
		double sum = 0;
		for(int j = 0;j<10;j++)
		{
			sum += (a*Math.sin(x/5*Math.PI) + b*Math.cos(x/5*Math.PI));
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
		f = 10 - func(ind);
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
