import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Properties;
import java.lang.Math;

public class Submission implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
	Integer population_;
	Integer generation_;
	
	
	public Submission()
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
		boolean mm = Boolean.parseBoolean(props.getProperty("Multimodal"));
		boolean rg = Boolean.parseBoolean(props.getProperty("Regular"));
		boolean sp = Boolean.parseBoolean(props.getProperty("Separable"));
		double limit = Double.parseDouble(props.getProperty("Evaluations"));
		// Do sth with property values, e.g. specify relevant settings of your algorithm
		
	}
	
	@Override
	public void run()
	{
		// Evaluating your results
		// E.g. evaluating a series of true/false predictions
		// boolean pred[] = ...
		double pred[][] = new double[population_][10]; 
		for(int i=0;i<population_;i++){
			for(int j=0;j<10;j++){
				pred[i][j] = rnd_.nextDouble()*10-5;
			}
		}
		Double score[][] = new Double[population_][10];
		for(int i=0;i<population_;i++){
			for(int j=0;j<10;j++){
				score[i][j] = (Double)evaluation_.evaluate(pred);
			}
		}
		
	}
}
