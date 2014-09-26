import java.util.Properties;
import org.vu.contest.ContestEvaluation;
import java.lang.Math;

// This is an example evaluation. It is based on the standard sphere function. It is a maximization problem with a maximum of 10 for 
//  	vector x=0.
// The sphere function itself is for minimization with minimum at 0, thus fitness is calculated as Fitness = 10 - 10*(f-fbest), 
//  	where f is the result of the sphere function
// Base performance is calculated as the distance of the expected fitness of a random search (with the same amount of available
//	evaluations) on the sphere function to the function minimum, thus Base = E[f_best_random] - ftarget. Fitness is scaled
//	according to this base, thus Fitness = 10 - 10*(f-fbest)/Base
public class AckleyEvaluation implements ContestEvaluation {
	// Evaluations budget
	private final static int EVALS_LIMIT_ = 50000;
	// The base performance. It is derived by doing random search on the sphere
	// function (see function method) with the same
	// amount of evaluations
	private final static double BASE_ = 11.5356;
	// The minimum of the sphere function
	private final static double ftarget_ = 0;

	// Best fitness so far
	private double best_;
	// Evaluations used so far
	private int evaluations_;

	// Properties of the evaluation
	private String multimodal_ = "true";
	private String regular_ = "true";
	private String separable_ = "false";
	private String evals_ = Integer.toString(EVALS_LIMIT_);

	public AckleyEvaluation() {
		best_ = -100;
		evaluations_ = 0;
	}

	// The standard sphere function. It has one minimum at 0.
	private double func1(double[] x) {
		double sum = 0;
		for (int i = 0; i < 10; i++)
			sum += Math.pow((x[i] + 1), 2);
		return sum;
	}

	private double func2(double[] x) {
		double sum = 0;
		for (int i = 0; i < 10; i++)
			sum += Math.cos(2 * Math.PI * (x[i] + 1));
		return sum;
	}

	@Override
	public Object evaluate(Object result) {
		// Check argument
		if (!(result instanceof double[]))
			throw new IllegalArgumentException();
		double ind[] = (double[]) result;
		if (ind.length != 10)
			throw new IllegalArgumentException();

		if (evaluations_ > EVALS_LIMIT_)
			return null;

		// Transform function value (sphere is minimization).
		// Normalize using the base performance
		double f = -20 * Math.exp(-0.2 * Math.sqrt(0.1 * func1(ind)))
				- Math.exp(0.1 * func2(ind)) + 20 + Math.exp(1);
		double score = Math.exp(-f) * 10;
		if (score > best_)
			best_ = score;
		evaluations_++;

		return new Double(score);
	}

	@Override
	public Object getData(Object arg0) {
		return null;
	}

	@Override
	public double getFinalResult() {
		return best_;
	}

	@Override
	public Properties getProperties() {
		Properties props = new Properties();
		props.put("Multimodal", multimodal_);
		props.put("Regular", regular_);
		props.put("Separable", separable_);
		props.put("Evaluations", evals_);
		return props;
	}
}
