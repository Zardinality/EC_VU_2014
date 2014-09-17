public class Main {

	public static void main(String[] args) {
		
		int trailnum = 10;
		int i,j;
		double[] result = new double[trailnum];
		double sum = 0;
		double score = 0;
		sum = 0;
		score = 0;
		for(i = 0; i < trailnum; i++)
		{
			SphereEvaluation eva = new SphereEvaluation();
			//AckleyEvaluation eva = new AckleyEvaluation();
			//FletcherEvaluation eva = new FletcherEvaluation();
			player19 test = new player19();
			test.setSeed(i);
			test.setEvaluation(eva);
			test.run();
			result[i] = eva.getFinalResult();
			sum += result[i];
		}
		score = sum/trailnum;
		System.out.println(Double.toString(score));
		sum = 0;
		score = 0;
		for(i = 0; i < trailnum; i++)
		{
			AckleyEvaluation eva = new AckleyEvaluation();
			player19 test = new player19();
			test.setSeed(i);
			test.setEvaluation(eva);
			test.run();
			result[i] = eva.getFinalResult();
			sum += result[i];
		}
		score = sum/trailnum;
		System.out.println(Double.toString(score));
		sum = 0;
		score = 0;
		for(i = 0; i < trailnum; i++)
		{
			FletcherEvaluation eva = new FletcherEvaluation();
			player19 test = new player19();
			test.setSeed(i);
			test.setEvaluation(eva);
			test.run();
			result[i] = eva.getFinalResult();
			sum += result[i];
		}
		score = sum/trailnum;
		System.out.println(Double.toString(score));
	}

}
