import java.io.File;
import java.io.FileNotFoundException;
import java.util.Properties;
import java.util.Scanner;

import org.vu.contest.ContestEvaluation;

// This is an example evaluation. It is based on the standard sphere function. It is a maximization problem with a maximum of 10 for 
//  	vector x=0.
// The sphere function itself is for minimization with minimum at 0, thus fitness is calculated as Fitness = 10 - 10*(f-fbest), 
//  	where f is the result of the sphere function
// Base performance is calculated as the distance of the expected fitness of a random search (with the same amount of available
//	evaluations) on the sphere function to the function minimum, thus Base = E[f_best_random] - ftarget. Fitness is scaled
//	according to this base, thus Fitness = 10 - 10*(f-fbest)/Base
public class SchwefelEvaluation implements ContestEvaluation 
{
	// Evaluations budget
	private final static int EVALS_LIMIT_ = 500000;
	// The base performance. It is derived by doing random search on the sphere function (see function method) with the same
	//  amount of evaluations
	
	// Best fitness so far
	private double best_;
	// Evaluations used so far
	private int evaluations_;
	
	// Properties of the evaluation
	private String multimodal_ = "true";
	private String regular_ = "false";
	private String separable_ = "false";
	private String evals_ = Integer.toString(EVALS_LIMIT_);
	
	final double INF = 1.0e99;
	final double EPS = 1.0e-14;
	final double E  = 2.7182818284590452353602874713526625;
	final double PI = 3.1415926535897932384626433832795029;
	
	double[] OShift,M,y,z,x_bound;
	int ini_flag=0,n_flag,func_flag;

	public SchwefelEvaluation() throws FileNotFoundException
	{
		best_ = -100;
		evaluations_ = 0;
		
		int nx = 10;
		
		int cf_num=10,i;
		
		y=new double[nx];
		z=new double[nx];
		x_bound=new double[nx];
		for (i=0; i<nx; i++)
			x_bound[i]=100.0;

		if (!(nx==2||nx==5||nx==10||nx==20||nx==30||nx==40||nx==50||nx==60||nx==70||nx==80||nx==90||nx==100))
		{
			System.out.println("\nError: Test functions are only defined for D=2,5,10,20,30,40,50,60,70,80,90,100.");
		}
		
		File fpt = new File("input_data/M_D"+nx+".txt");//* Load M data *
		Scanner input = new Scanner(fpt);
		if (!fpt.exists())
		{
		    System.out.println("\n Error: Cannot open input file for reading ");
		}
		 
		M=new double[cf_num*nx*nx]; 
		
		for (i=0; i<cf_num*nx*nx; i++)
		{
				M[i]=input.nextDouble();
		}
		input.close();
		

		
		fpt=new File("input_data/shift_data.txt");
		input = new Scanner(fpt);
		if (!fpt.exists())
		{
			System.out.println("\n Error: Cannot open input file for reading ");
		}
		OShift=new double[nx*cf_num];
		for(i=0;i<cf_num*nx;i++)
		{
			OShift[i]=input.nextDouble();
		}
		input.close();
		
		n_flag=nx;
		ini_flag=1;
	}

	void shiftfunc (double[] x, double[] xshift, int nx,double[] Os)
	{
		int i;
	    for (i=0; i<nx; i++)
	    {
	        xshift[i]=x[i]-Os[i];
	    }
	}

	void rotatefunc (double[] x, double[] xrot, int nx,double[] Mr)
	{
		int i,j;
	    for (i=0; i<nx; i++)
	    {
	        xrot[i]=0;
				for (j=0; j<nx; j++)
				{
					xrot[i]=xrot[i]+x[j]*Mr[i*nx+j];
				}
	    }
	}
	
	// The standard sphere function. It has one minimum at 0.
	double schwefel_func (double[] x, double f, int nx, double[] Os,double[] Mr,int r_flag) /* Schwefel's  */
	{
	    int i;
		double tmp;
		for (i = 0; i < nx; i++) {
			x[i] *= 20;
		}
		shiftfunc(x, y, nx, Os);
		for (i=0; i<nx; i++)//shrink to the orginal search range
	    {
	        y[i]*=1000/100;
	    }
		if (r_flag==1)
		rotatefunc(y, z, nx, Mr);
		else
	    for (i=0; i<nx; i++)
			z[i]=y[i];

		for (i=0; i<nx; i++)
			y[i] = z[i]*Math.pow(10.0,1.0*i/(nx-1)/2.0);

		for (i=0; i<nx; i++)
			z[i] = y[i]+4.209687462275036e+002;
		
	    f=0;
	    for (i=0; i<nx; i++)
		{
			if (z[i]>500)
			{
				f-=(500.0-(z[i]%500))*Math.sin(Math.pow(500.0-(z[i]%500),0.5));
				tmp=(z[i]-500.0)/100;
				f+= tmp*tmp/nx;
			}
			else if (z[i]<-500)
			{
				f-=(-500.0+(Math.abs(z[i])%500))*Math.sin(Math.pow(500.0-(Math.abs(z[i])%500),0.5));
				tmp=(z[i]+500.0)/100;
				f+= tmp*tmp/nx;
			}
			else
				f-=z[i]*Math.sin(Math.pow(Math.abs(z[i]),0.5));
	    }
	    f=4.189828872724338e+002*nx+f;
	    
	    return f;
	}
	
	@Override
	public Object evaluate(Object result) 
	{
		// Check argument
		if(!(result instanceof double[])) throw new IllegalArgumentException();
		double ind[] = (double[]) result;
		if(ind.length!=10) throw new IllegalArgumentException();
		
		if(evaluations_>EVALS_LIMIT_) return null;
		
		// Transform function value (sphere is minimization).
		// Normalize using the base performance
		double f = 0;
		f = Math.exp(-schwefel_func(ind, f, 10, OShift, M, 1) / 100) * 10;
		//System.out.println(f);
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
