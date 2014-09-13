public class Main {

	public static void main(String[] args) {
		
		int trailnum = 20;
		int i,j;
		double[] result = new double[trailnum];
		double sum = 0;
		double score = 0;
		for(i = 0; i < trailnum; i++)
		{
			//SphereEvaluation eva = new SphereEvaluation();
			AckleyEvaluation eva = new AckleyEvaluation();
			player19 test = new player19();
			test.setSeed(i);
			test.setEvaluation(eva);
			test.run();
			result[i] = test.getResult();
			sum += result[i];
		}
		score = sum/trailnum;
		System.out.println(Double.toString(score));
	}

}
